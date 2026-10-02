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
"""
function implicit_solution(f::DiffusionFactor, p::LenticulumCore.Polarity, inputs, π, ps, st)
    _require_implicit(f)
    z₀, ρ = _implicit_problem(f, p, inputs, π)
    m = _implicit_model(f)
    return implicit_infer(m, z₀, ρ, ps, st; z_init = z₀, tol = f.prox.tol,
                          maxiters = f.prox.maxiters, step = f.prox.step)
end

_implicit_model(f::DiffusionFactor) = ImplicitDiffusion(f.predictor, f.prox.nodes; λ = f.prox.λ)

_require_implicit(f::DiffusionFactor) = f.prox isa ImplicitProx || throw(ArgumentError(
    "this DiffusionFactor inverts with $(typeof(f.prox)); construct it with prox = ImplicitProx(nodes)"))

function _implicit_problem(f::DiffusionFactor, p::LenticulumCore.Polarity, inputs, π)
    z₀ = assemble_state(f, inputs, π)
    ρ, hard = precision_vector(f, p)
    ρ = copy(ρ)
    ρ[hard] .= Inf
    return (z₀, ρ)
end

"""
    implicit_factor_pullback(f, polarity, inputs, π, z̄, ps, st)
        -> (inputs = NamedTuple, ps = ps̄, precisions = NamedTuple)

The backward pass of the factor's inversion, by the adjoint of
[`implicit_pullback`](@ref). `z̄` is the cotangent of a downstream loss with respect to the
inferred state, given **per channel** as a `NamedTuple` (e.g. `(y = ȳ,)`; channels left out
have zero cotangent).

Returns the cotangents with respect to
- each channel's incoming point (`inputs`): the clamped value for an observed channel, the
  anchor for a soft one;
- the predictor's parameters (`ps`);
- each channel's precision (`precisions`; zero for hard-clamped channels, whose precision is
  infinite and not a differentiable quantity).

Throws if the solve did not converge, as `implicit_pullback` does.
"""
function implicit_factor_pullback(f::DiffusionFactor{names}, p::LenticulumCore.Polarity,
                                  inputs, π, z̄::NamedTuple, ps, st) where {names}
    _require_implicit(f)
    z₀, ρ = _implicit_problem(f, p, inputs, π)
    m = _implicit_model(f)
    sol, st = implicit_infer(m, z₀, ρ, ps, st; z_init = z₀, tol = f.prox.tol,
                             maxiters = f.prox.maxiters, step = f.prox.step)
    rs = blockranges(f)
    zbar = zeros(Float64, statedim(f))
    for (k, v) in pairs(z̄)
        haskey(rs, k) || throw(ArgumentError("channel :$k is not a channel of this DiffusionFactor"))
        zbar[getfield(rs, k)] .= _vec(v)
    end
    b = implicit_pullback(m, sol, z₀, ρ, zbar, ps, st)
    blocks_in = NamedTuple{names}(map(nm -> b.z₀[getfield(rs, nm)], names))
    blocks_ρ = NamedTuple{names}(map(nm -> sum(b.ρ[getfield(rs, nm)]), names))   # one precision per channel
    return (inputs = blocks_in, ps = b.ps, precisions = blocks_ρ)
end
