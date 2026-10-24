# ---------------------------------------------------------------------------
# A DiffusionFactor whose inversion is the deterministic implicit solver (`ImplicitProx`).
#
# `invert`, `factor_message` and `local_free_energy` in `factor.jl` already route through
# `_run_prox`, which dispatches here. This file adds what RED-Diff could not offer: the solver's
# report, and the factor-level backward pass. See `implicit_factor.md`.
# ---------------------------------------------------------------------------


"""
    implicit_solution(f::DiffusionFactor, polarity, inputs, π, ps, st) -> (ImplicitSolution, st)

Run the factor's implicit inversion and return the whole [`ImplicitSolution`](@ref): the full
state, the residual, and the `converged` / `stable` flags. `invert` and `factor_message` return
only the target block as a `DiracBelief`; this is the call to make when the report matters.

The reference configuration is [`assemble_state`](@ref)`(f, inputs, π)`: inputs on their
channels, and the prior `π`'s point (if any) on the others. It doubles as the **warm start**,
which is how a message schedule reuses the previous belief: as in `ImplicitLayers`' DEQ factor,
the prior seeds the solver. Whether it also *pulls* the solution is decided by the polarity's
precision on the unobserved channels — `default_precision(Unobserved()) == 1`, an anchor; set it
to 0 for a pure conditional.

For a [`ProxDM`](@ref) inversion the solver is [`prox_infer`](@ref) instead, with the same
problem and warm start.
"""
function implicit_solution(f::DiffusionFactor, p::LenticulumCore.Polarity, inputs, π, ps, st)
    _require_implicit(f)
    z₀, ρ = _implicit_problem(f, p, inputs, π)
    return _factor_solve(f.prox, f, z₀, ρ, ps, st)
end

_factor_solve(prox::ImplicitProx, f::DiffusionFactor, z₀, ρ, ps, st) =
    implicit_infer(_implicit_model(f), z₀, ρ, ps, st; z_init = z₀, _solver_kwargs(prox)...)

_implicit_model(f::DiffusionFactor) = ImplicitDiffusion(f.predictor, f.prox.nodes; λ = f.prox.λ, batched = f.prox.batched)

_require_implicit(f::DiffusionFactor) = f.prox isa Union{ImplicitProx,ProxDM} || throw(ArgumentError(
    "this DiffusionFactor inverts with $(typeof(f.prox)); construct it with prox = ImplicitProx(nodes) or ProxDM(prox)"))

function _implicit_problem(f::DiffusionFactor, p::LenticulumCore.Polarity, inputs, π)
    z₀ = assemble_state(f, inputs, π, p)
    ρ, hard = precision_vector(f, p)
    ρ = copy(ρ)
    ρ[hard] .= Inf
    return (z₀, ρ)
end

"""
    implicit_factor_pullback(f, polarity, inputs, π, z̄, ps, st)
        -> (inputs = NamedTuple, prior = π̄, ps = ps̄, precisions = NamedTuple)

The backward pass of the factor's inversion, by the adjoint of
[`implicit_pullback`](@ref). `z̄` is the cotangent of a downstream loss with respect to the
inferred state, given **per channel** as a `NamedTuple` (e.g. `(y = ȳ,)`; channels left out
have zero cotangent).

Returns the cotangents with respect to
- each channel's incoming point (`inputs`): the clamped value for an observed channel, the
  anchor for a soft one;
- the prior's point (`prior`): sums the blocks initialized from `π`, zero when incoming
  points override it, and `nothing` if `π` carries no point;
- the predictor's parameters (`ps`);
- each channel's precision (`precisions`; zero for hard-clamped channels, whose precision is
  infinite and not a differentiable quantity).

Throws if the solve did not converge, as `implicit_pullback` does. For a [`ProxDM`](@ref)
inversion the adjoint is [`prox_pullback`](@ref), and `ps` is `(predictor = 0, prox = ps̄)`: the
inversion does not read the prior's parameters.
"""
function implicit_factor_pullback(f::DiffusionFactor{names}, p::LenticulumCore.Polarity,
                                  inputs, π, z̄::NamedTuple, ps, st) where {names}
    _require_implicit(f)
    z₀, ρ = _implicit_problem(f, p, inputs, π)
    sol, st = _factor_solve(f.prox, f, z₀, ρ, ps, st)
    rs = blockranges(f)
    zbar = zeros(Float64, statedim(f))
    for (k, v) in pairs(z̄)
        haskey(rs, k) || throw(ArgumentError("channel :$k is not a channel of this DiffusionFactor"))
        zbar[getfield(rs, k)] .= _vec(v)
    end
    b = _factor_pullback(f.prox, f, sol, z₀, ρ, zbar, ps, st)
    blocks_in = NamedTuple{names}(map(nm -> b.z₀[getfield(rs, nm)], names))
    blocks_ρ = NamedTuple{names}(map(nm -> sum(b.ρ[getfield(rs, nm)]), names))   # one precision per channel
    unobs = LenticulumCore.unobserved_channels(p)
    prior_point = _point(π)
    prior_cot = prior_point === nothing ? nothing : zeros(eltype(b.z₀), length(prior_point))
    if prior_cot !== nothing
        for nm in unobs
            # assemble_state reads π only when this channel has no incoming point.
            _point(_get(inputs, nm)) === nothing || continue
            prior_cot .+= b.z₀[getfield(rs, nm)]
        end
    end
    return (inputs = blocks_in, prior = prior_cot, ps = b.ps, precisions = blocks_ρ)
end

_factor_pullback(prox::ImplicitProx, f::DiffusionFactor, sol, z₀, ρ, z̄, ps, st) =
    implicit_pullback(_implicit_model(f), sol, z₀, ρ, z̄, ps, st; linsolve = prox.linsolve)

# --- Gaussian messages --------------------------------------------------------------------

# The Laplace posterior on the target channel; as a message (`likelihood = true`) the target's
# own anchor N(z₀, ρ⁻²), which the solver multiplied in, is divided out again. In canonical form
# that is a subtraction, Λ - diag(ρ²) and η - ρ²z₀, exact for the Laplace approximation, and the
# result may be improper (a likelihood that constrains only some directions). The diffusion
# model's own prior cannot be divided out (`implicit_factor.md` §5).
function _target_belief(prox::ImplicitProx, f::DiffusionFactor, p, inputs, π, ps, st, target; likelihood::Bool)
    prox.message === :point && return invoke(_target_belief, Tuple{Any,DiffusionFactor,Any,Any,Any,Any,Any,Any},
                                             prox, f, p, inputs, π, ps, st, target; likelihood)
    z₀, ρ = _implicit_problem(f, p, inputs, π)
    m = _implicit_model(f)
    r = getfield(blockranges(f), target)
    if prox.message === :mixture
        mix, st = implicit_mixture(m, z₀, ρ, ps, st; coords = r, linsolve = prox.linsolve,
                                   tol = prox.tol, maxiters = prox.maxiters, step = prox.step)
        likelihood || return (mix, st)
        D = ρ[r] .^ 2
        all(iszero, D) && return (mix, st)
        comps = [LenticulumCore.GaussianBelief(c.η .- D .* z₀[r], Matrix(c.Λ) .- Diagonal(D)) for c in mix.components]
        all(LenticulumCore.isproper, comps) || throw(ArgumentError(
            "anchor removal produced an improper mixture component; normalized mixture weights are undefined"))
        logw = [w + LenticulumCore.logpartition(msg) - LenticulumCore.logpartition(post)
                for (w, msg, post) in zip(mix.logw, comps, mix.components)]
        return (LenticulumCore.MixtureBelief(Val(:log), comps, logw), st)
    end
    sol, st = implicit_infer(m, z₀, ρ, ps, st; z_init = z₀, _solver_kwargs(prox)...)
    (sol.converged && sol.stable) || throw(ArgumentError(
        "a Gaussian message needs a converged, stable answer (converged = $(sol.converged), stable = $(sol.stable))"))
    post = laplace_belief(m, sol, ρ, ps, st; coords = r, linsolve = prox.linsolve)
    likelihood || return (post, st)
    D = ρ[r] .^ 2
    return (LenticulumCore.GaussianBelief(post.η .- D .* z₀[r], Matrix(post.Λ) .- Diagonal(D)), st)
end

