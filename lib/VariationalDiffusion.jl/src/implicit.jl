# ---------------------------------------------------------------------------
# A diffusion model as an implicit learner: a relation R_θ ⊂ Z, inference by root-finding on
# a residual field, and backpropagation by the adjoint (implicit function theorem).
#
# The residual on the free coordinates F (those with finite precision ρ_i) is
#
#     r(z) = g_θ(z) + ρ² ⊙ (z - z₀),     g_θ(z) = Σ_k w_k λ_{t_k} (ε_θ(α_k z + σ_k ε_k, t_k) - ε_k)
#
# — RED-Diff's stop-gradient field (Mardani et al., Prop. 2) made DETERMINISTIC by fixing the
# nodes (t_k, ε_k, w_k). Coordinates with ρ_i = ∞ are hard clamps, z_i = z₀_i.
#
# For an exact score, g is the gradient of a smoothed negative log-density, so the root is a
# P²-metric proximal point of that density (`Implicit Diffusion Learners.md` §3). For a learned
# network g need not be a gradient; nothing below assumes it is.
#
# Backward pass (`Backpropagation through Implicit Inference.md`):
#     state     r_F(z★; z₀, ρ, θ) = 0
#     adjoint   J_FFᵀ λ = -∂ℓ/∂z_F,                 J = ∂r/∂z
#     gradient  θ̄ = λᵀ ∂r_F/∂θ,  x̄_C = J_FCᵀ λ,  z̄₀_F = -ρ²⊙λ,  ρ̄_F = 2ρ⊙(z★ - z₀)⊙λ
#
# See `implicit.md`.
# ---------------------------------------------------------------------------

"""
    FieldNodes(t, ε, w)

The fixed quadrature nodes ``(t_k, \\varepsilon_k, w_k)`` that turn RED-Diff's stochastic
gradient into a deterministic vector field. Fixing them is *sample-average approximation*:
the field, its root and its Jacobian become ordinary deterministic objects, which is what the
implicit function theorem needs. Build them with [`field_nodes`](@ref) or
[`noisefree_nodes`](@ref).
"""
struct FieldNodes{T<:Real}
    t::Vector{T}
    ε::Matrix{T}
    w::Vector{T}
    function FieldNodes(t::Vector{T}, ε::Matrix{T}, w::Vector{T}) where {T}
        length(t) == size(ε, 2) == length(w) || throw(DimensionMismatch("one column of ε and one weight per time"))
        return new{T}(t, ε, w)
    end
end

"""
    field_nodes(rng, n; levels = range(0.002, 0.05; length = 8), samples = 4, antithetic = true)

`samples` noise draws at each noise level in `levels`, equally weighted. With `antithetic`
each draw ``\\varepsilon`` is paired with ``-\\varepsilon``: the control-variate term
``-\\sum w_k\\lambda_k\\varepsilon_k`` of the field then cancels **exactly**, so the field has
no constant tilt.

The levels matter more than anything else here. A relation is only recovered at noise scales
below its own curvature scale: the default stops at ``t = 0.05`` (``\\sigma_t\\approx0.16``).
RED-Diff's training-time range ``t\\in[t_{\\min},1]`` smooths a unit circle into a blob whose
density peaks at the centre (`Implicit Diffusion Learners.md` §5).
"""
function field_nodes(rng::AbstractRNG, n::Integer; levels = range(0.002, 0.05; length = 8),
                     samples::Integer = 4, antithetic::Bool = true)
    ts = Float64[]; cols = Vector{Float64}[]
    for t in levels, _ in 1:samples
        e = randn(rng, n)
        push!(ts, t); push!(cols, e)
        antithetic && (push!(ts, t); push!(cols, -e))
    end
    return FieldNodes(ts, reduce(hcat, cols), fill(1 / length(ts), length(ts)))
end

"""
    noisefree_nodes(n, t)

One noise level, no noise: ``g(z) = \\lambda_t\\,\\varepsilon_\\theta(\\alpha_t z, t)``. This is the
**deterministic relaxation**: a root satisfies ``\\hat x_0(\\alpha_t z) = z`` on the free
coordinates — a fixed point of the Tweedie denoiser, i.e. a deep equilibrium model whose layer
is the denoiser (`Deterministic Relaxation.md`).
"""
noisefree_nodes(n::Integer, t::Real) = FieldNodes([float(t)], zeros(n, 1), [1.0])

"""
    AbstractImplicitRelation

A learned relation that can be queried by root-finding: anything with a field
[`prior_field`](@ref) and its Jacobian [`prior_jacobian`](@ref). [`ImplicitDiffusion`](@ref) is one
diffusion model; [`ProductRelation`](@ref) pools several. The query functions
([`implicit_infer`](@ref), [`implicit_roots`](@ref), [`implicit_laplace`](@ref),
[`implicit_pullback`](@ref)) accept any of them.
"""
abstract type AbstractImplicitRelation end

"""
    ImplicitDiffusion(predictor, nodes; λ = 1.0, batched = false)

The relation ``R_\\theta = \\{z : r(z) = 0\\}`` defined by a noise predictor and a fixed node
set, with RED-Diff's weighting ``\\lambda_t = \\lambda\\sigma_t/\\alpha_t``.

With `batched = true` all nodes of one noise level go through the network in **one call**, as
the columns of an ``n\\times B`` matrix at a scalar time: one call per level instead of one per
node, for the field and for every derivative. The predictor must then accept a matrix input
(a Lux `Chain` of `Dense` layers does, with an `input` function that repeats the time embedding
across columns). The answers are the same either way (`implicit.md` §6).
"""
struct ImplicitDiffusion{P<:NoisePredictor,T<:Real} <: AbstractImplicitRelation
    predictor::P
    nodes::FieldNodes{T}
    λ::T
    batched::Bool
end
ImplicitDiffusion(pred::NoisePredictor, nodes::FieldNodes{T}; λ = 1.0, batched::Bool = false) where {T} =
    ImplicitDiffusion(pred, nodes, T(λ), batched)

_weight(m::ImplicitDiffusion, t) = m.λ * sigma(m.predictor.schedule, t) / alpha(m.predictor.schedule, t)
_node_input(m::ImplicitDiffusion, z, k) =
    perturb(m.predictor.schedule, z, m.nodes.t[k], view(m.nodes.ε, :, k))

# The nodes evaluated in one network call: one noise level each when batched, else one node.
function _groups(m::ImplicitDiffusion)
    m.batched || return [[k] for k in eachindex(m.nodes.t)]
    ts = m.nodes.t
    return [findall(==(u), ts) for u in unique(ts)]
end
# a vector for a single node (as the predictor has always been called), an n × B matrix when batched
_group_input(m::ImplicitDiffusion, z, grp) =
    m.batched ? perturb(m.predictor.schedule, z, m.nodes.t[first(grp)], m.nodes.ε[:, grp]) :
                _node_input(m, z, only(grp))

# coordinates as a sorted index vector: `:` (all), a mask, or indices
_coords(::Colon, n) = collect(1:n)
function _coords(c::AbstractVector{Bool}, n)
    length(c) == n || throw(DimensionMismatch("a coordinate mask needs $n entries, got $(length(c))"))
    return findall(c)
end
_coords(c, n) = collect(Int, c)

"""
    prior_field(m, z, ps, st) -> (g, st)

``g_\\theta(z) = \\sum_k w_k\\lambda_{t_k}\\bigl(\\varepsilon_\\theta(\\alpha_k z+\\sigma_k\\varepsilon_k,t_k)-\\varepsilon_k\\bigr)``.
"""
function prior_field(m::ImplicitDiffusion, z, ps, st)
    g = zero(float.(z))
    for grp in _groups(m)
        t = m.nodes.t[first(grp)]
        ε̂, st = epsilon(m.predictor, _group_input(m, z, grp), t, ps, st)
        g = g .+ _weight(m, t) .* ((reshape(ε̂, length(z), :) .- view(m.nodes.ε, :, grp)) * view(m.nodes.w, grp))
    end
    return (g, st)
end

"""
    prior_jacobian(m, z, ps, st; coords = :) -> Matrix

``\\partial g/\\partial z = \\sum_k w_k\\lambda_{t_k}\\alpha_{t_k}\\,\\partial_x\\varepsilon_\\theta``. Symmetric iff
the field is (locally) a gradient.

With `coords` (indices or a mask) only the block ``J_{II}`` on those coordinates is computed, at
a cost that grows with their number, not with the dimension (see
[`epsilon_jacobian_block`](@ref)). The solver, the Laplace approximation and the adjoint only
ever need the block on the free coordinates, so a query with ten free coordinates in a
794-dimensional state costs ten derivative passes per noise level, not 794.
"""
function prior_jacobian(m::ImplicitDiffusion, z, ps, st; coords = Colon())
    idx = _coords(coords, length(z))
    J = zeros(eltype(float(z)), length(idx), length(idx))
    isempty(idx) && return J
    for grp in _groups(m)
        t = m.nodes.t[first(grp)]
        c = _weight(m, t) * alpha(m.predictor.schedule, t)
        J .+= c .* epsilon_jacobian_block(m.predictor, _group_input(m, z, grp), t, ps, st, idx, m.nodes.w[grp])
    end
    return J
end

"""
    prior_vjp(m, z, w, ps, st) -> Vector

The vector–Jacobian product ``(\\partial g/\\partial z)^\\top w`` of the field, without forming the
Jacobian: one input VJP per noise level ([`epsilon_vjp_input`](@ref)). The adjoint uses it for
the cotangent of the clamped inputs, ``J_{FC}^\\top\\lambda``.
"""
function prior_vjp(m::ImplicitDiffusion, z, w, ps, st)
    v = zero(float.(z))
    for grp in _groups(m)
        t = m.nodes.t[first(grp)]
        c = _weight(m, t) * alpha(m.predictor.schedule, t)
        W = m.batched ? w * transpose(m.nodes.w[grp]) : w .* m.nodes.w[only(grp)]
        X̄ = epsilon_vjp_input(m.predictor, _group_input(m, z, grp), t, ps, st, W)
        v .+= c .* vec(sum(reshape(X̄, length(z), :); dims = 2))
    end
    return v
end

# any other relation: through its full Jacobian
prior_vjp(m::AbstractImplicitRelation, z, w, ps, st) = prior_jacobian(m, z, ps, st)' * w

"""
    implicit_residual(m, z, z₀, ρ, ps, st) -> (r, st)

``r = g_\\theta(z) + \\rho^2\\odot(z - z_0)`` on the free coordinates (finite `ρ`), zero on the
hard-clamped ones (``\\rho_i = \\infty``), where the constraint is ``z_i = z_{0,i}`` instead.
"""
function implicit_residual(m::AbstractImplicitRelation, z, z₀, ρ, ps, st)
    g, st = prior_field(m, z, ps, st)
    free = .!isinf.(ρ)
    r = zero(g)
    r[free] .= g[free] .+ ρ[free] .^ 2 .* (z[free] .- z₀[free])
    return (r, st)
end

"""
    ImplicitSolution

What inference returns: the state `z`, the residual norm on the free coordinates, the number
of iterations, and two flags.

- `converged` — the residual is below tolerance. If not, `z` is still returned (an *anytime*
  answer) but the implicit function theorem does not apply at it.
- `stable` — the symmetric part of ``J_{FF}`` is positive definite: a strict local minimum
  of the energy when the field is a gradient. A root that is not stable is a saddle or a
  maximum of the energy — on the zero set of ``r``, but not a member of the relation one wants.
"""
struct ImplicitSolution{V<:AbstractVector,T<:Real}
    z::V
    residual::T
    iters::Int
    converged::Bool
    stable::Bool
end

function _stable(Jff)
    isempty(Jff) && return true
    return isposdef(Symmetric((Jff .+ Jff') ./ 2))
end

"""
    implicit_infer(m, z₀, ρ, ps, st; z_init = z₀, tol = 1e-9, maxiters = 200, step = 0.05)
        -> (ImplicitSolution, st)

Find a stable root of the residual on the free coordinates, warm-started from `z_init`.

Each iteration takes a **Newton** step ``-J_{FF}^{-1}r_F`` with backtracking on ``\\|r\\|`` when the
symmetric part of ``J_{FF}`` is positive definite, and otherwise a **descent** step
``-\\texttt{step}\\cdot r_F`` along the field (energy descent when the field is a gradient). Newton
is only trusted in locally convex regions so that it is not drawn to saddles and maxima.

`z₀` holds the clamp values on hard coordinates and the soft targets elsewhere; `ρ` is the
precision vector, `Inf` for a hard clamp (cf. [`precision_vector`](@ref)).
"""
function implicit_infer(m::AbstractImplicitRelation, z₀, ρ, ps, st;
                        z_init = z₀, tol = 1e-9, maxiters::Integer = 200, step = 0.05)
    free = .!isinf.(ρ)
    z = float.(copy(z_init))
    z[.!free] .= z₀[.!free]
    ρ²F = ρ[free] .^ 2
    rnorm(zz) = norm(first(implicit_residual(m, zz, z₀, ρ, ps, st))[free])
    for it in 1:maxiters
        r, st = implicit_residual(m, z, z₀, ρ, ps, st)
        rF = r[free]
        nr = norm(rF)
        Jff = prior_jacobian(m, z, ps, st; coords = free) .+ Diagonal(ρ²F)
        if nr ≤ tol * (1 + norm(z))
            return (ImplicitSolution(z, nr, it - 1, true, _stable(Jff)), st)
        end
        if _stable(Jff)
            d = -(Jff \ rF)
            η = 1.0
            while η > 1e-8
                zn = copy(z); zn[free] .+= η .* d
                rnorm(zn) < (1 - 1e-4 * η) * nr && break
                η /= 2
            end
            z[free] .+= η .* d
        else
            z[free] .-= step .* rF
        end
    end
    r, st = implicit_residual(m, z, z₀, ρ, ps, st)
    Jff = prior_jacobian(m, z, ps, st; coords = free) .+ Diagonal(ρ²F)
    nr = norm(r[free])
    return (ImplicitSolution(z, nr, maxiters, nr ≤ tol * (1 + norm(z)), _stable(Jff)), st)
end

# ps-shaped arithmetic for the parameter cotangent
_axpy(a, x::NamedTuple, y::NamedTuple) = map((xi, yi) -> _axpy(a, xi, yi), x, y)
_axpy(a, x::AbstractArray, y::AbstractArray) = y .+ a .* x
_axpy(a, x::Real, y::Real) = y + a * x
_zero(x::NamedTuple) = map(_zero, x)
_zero(x::AbstractArray) = zero(x)
_zero(x::Real) = zero(x)
_hasparams(ps) = !(ps isa NamedTuple && isempty(ps))

"""
    implicit_pullback(m, sol, z₀, ρ, z̄, ps, st) -> (z₀ = z̄₀, ρ = ρ̄, ps = ps̄)

Reverse-mode derivative of ``z^\\star(z_0,\\rho,\\theta)`` by the adjoint method. Given the
cotangent `z̄` ``= \\partial\\ell/\\partial z^\\star`` of a downstream loss, solve

```math
J_{FF}^\\top\\lambda = -\\bar z_F
```

once, and read every gradient off ``\\lambda``:

| w.r.t. | cotangent |
|---|---|
| parameters ``\\theta`` | ``\\sum_k w_k\\lambda_{t_k}\\,(\\partial_\\theta\\varepsilon_\\theta(x_k,t_k))^\\top\\lambda`` (via [`epsilon_vjp_params`](@ref)) |
| hard-clamped inputs ``z_{0,C}`` | ``\\bar z_C + J_{FC}^\\top\\lambda`` |
| soft targets ``z_{0,F}`` | ``-\\rho_F^2\\odot\\lambda`` |
| precisions ``\\rho_F`` | ``2\\rho_F\\odot(z^\\star_F - z_{0,F})\\odot\\lambda`` |

**No noise has to be stored from the forward pass**: the nodes are part of the model, and
the Jacobians are evaluated at ``z^\\star`` afresh. Throws if `sol` did not converge, because the
implicit function theorem says nothing about a point that is not a root
(`Backpropagation through Implicit Inference.md` §6).
"""
function implicit_pullback(m::AbstractImplicitRelation, sol::ImplicitSolution, z₀, ρ, z̄, ps, st)
    sol.converged || throw(ArgumentError(
        "implicit_pullback at a non-converged state (residual $(sol.residual)): the implicit " *
        "function theorem does not apply. Re-solve, or use an unrolled/one-step surrogate."))
    z = sol.z
    free = .!isinf.(ρ)
    clamped = .!free
    Jff = prior_jacobian(m, z, ps, st; coords = free) .+ Diagonal(ρ[free] .^ 2)
    λF = -(Jff' \ z̄[free])
    λ = zero(float.(z)); λ[free] .= λF

    ps̄ = _params_cotangent(m, z, λ, ps, st)

    z̄₀ = zero(float.(z))
    z̄₀[free] .= .-(ρ[free] .^ 2) .* λF
    any(clamped) && (z̄₀[clamped] .= z̄[clamped] .+ prior_vjp(m, z, λ, ps, st)[clamped])   # J_FCᵀλ
    ρ̄ = zero(float.(z))
    ρ̄[free] .= 2 .* ρ[free] .* (z[free] .- z₀[free]) .* λF
    return (z₀ = z̄₀, ρ = ρ̄, ps = ps̄)
end

# (∂_θ g)ᵀ λ: the parameter part of the adjoint, one VJP per node (per noise level when batched)
function _params_cotangent(m::ImplicitDiffusion, z, λ, ps, st)
    ps̄ = _zero(ps)
    _hasparams(ps) || return ps̄
    for grp in _groups(m)
        t = m.nodes.t[first(grp)]
        X = _group_input(m, z, grp)
        if m.batched                    # the node weights go into the cotangent, one per column
            g = epsilon_vjp_params(m.predictor, X, t, ps, st, λ * transpose(m.nodes.w[grp]))
            ps̄ = _axpy(_weight(m, t), g, ps̄)
        else
            g = epsilon_vjp_params(m.predictor, X, t, ps, st, λ)
            ps̄ = _axpy(m.nodes.w[only(grp)] * _weight(m, t), g, ps̄)
        end
    end
    return ps̄
end

# --- As a factor's inversion ----------------------------------------------------

"""
    ImplicitProx(nodes; λ = 1.0, tol = 1e-9, maxiters = 200, step = 0.05, message = :point)

The deterministic implicit solver as a [`DiffusionFactor`](@ref)'s inversion, in place of
[`REDDiff`](@ref):

```julia
f = DiffusionFactor((x = 1, y = 1), pred; prox = ImplicitProx(field_nodes(rng, 2)))
```

Inversion is then [`implicit_infer`](@ref) on the factor's state space with the polarity's
precisions (`Inf` = hard clamp). Compared with RED-Diff it is deterministic, it reports
convergence and stability ([`implicit_solution`](@ref)), its free energy is deterministic, and it
has a backward pass ([`implicit_factor_pullback`](@ref)). `nodes` must have the factor's state
dimension.

`message` chooses what `invert` and `factor_message` return on the target channel:
- `:point` — a `DiracBelief` at the answer;
- `:gaussian` — a `GaussianBelief` from the Laplace approximation at the answer
  ([`laplace_belief`](@ref)): the posterior marginal from `invert`, and from `factor_message`
  the same with the target's own anchor divided out, so that a variable does not count its
  incoming belief twice. Use `λ = density_lambda(schedule, nodes)` for calibrated covariances.
"""
struct ImplicitProx{T<:Real}
    nodes::FieldNodes{T}
    λ::T
    tol::T
    maxiters::Int
    step::T
    message::Symbol
    function ImplicitProx(nodes::FieldNodes{T}, λ::T, tol::T, maxiters::Int, step::T, message::Symbol) where {T<:Real}
        message in (:point, :gaussian) || throw(ArgumentError("message must be :point or :gaussian, got :$message"))
        return new{T}(nodes, λ, tol, maxiters, step, message)
    end
end
ImplicitProx(nodes::FieldNodes{T}; λ = 1.0, tol = 1e-9, maxiters::Integer = 200, step = 0.05,
             message::Symbol = :point) where {T} =
    ImplicitProx(nodes, T(λ), T(tol), Int(maxiters), T(step), message)
