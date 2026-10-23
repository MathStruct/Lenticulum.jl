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
    field_nodes(rng, n; levels = range(0.002, 0.05; length = 8), samples = 4, antithetic = true,
                orthogonal = false)

`samples` noise draws at each noise level in `levels`, equally weighted. With `antithetic`
each draw ``\\varepsilon`` is paired with ``-\\varepsilon``: the control-variate term
``-\\sum w_k\\lambda_k\\varepsilon_k`` of the field then cancels **exactly**, so the field has
no constant tilt. With `orthogonal = true`, draws within each block of up to `n` samples at a
given noise level are QR-orthogonalised while preserving their radial ``\\chi_n`` norms, reducing
quadrature variance across spatial directions.

The levels matter more than anything else here. A relation is only recovered at noise scales
below its own curvature scale: the default stops at ``t = 0.05`` (``\\sigma_t\\approx0.16``).
RED-Diff's training-time range ``t\\in[t_{\\min},1]`` smooths a unit circle into a blob whose
density peaks at the centre (`Implicit Diffusion Learners.md` §5).
"""
function field_nodes(rng::AbstractRNG, n::Integer; levels = range(0.002, 0.05; length = 8),
                     samples::Integer = 4, antithetic::Bool = true, orthogonal::Bool = false)
    ts = Float64[]; cols = Vector{Float64}[]
    for t in levels
        if orthogonal
            rem = samples
            while rem > 0
                b = min(rem, n)
                G = randn(rng, n, b)
                radii = [norm(view(G, :, j)) for j in 1:b]
                Q = Matrix(LinearAlgebra.qr(G).Q)[:, 1:b]
                for j in 1:b
                    e = Q[:, j] .* radii[j]
                    push!(ts, t); push!(cols, e)
                    antithetic && (push!(ts, t); push!(cols, -e))
                end
                rem -= b
            end
        else
            for _ in 1:samples
                e = randn(rng, n)
                push!(ts, t); push!(cols, e)
                antithetic && (push!(ts, t); push!(cols, -e))
            end
        end
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

**Noise blocks.** For a predictor with `blocks` ([`NoisePredictor`](@ref)), `levels` gives each
block's noise level at the nodes: `nothing` for a block annealed with the nodes (at each node's
level), a number for a block held at that level at every node. A clamped block held at the
schedule's `tmin` is a clean condition, so the field on the free blocks is the smoothed
*conditional* score, ``\\nabla\\log p_t(z_F \\mid z_C)``, instead of the joint score at a common
level ([`conditional_levels`](@ref); `implicit.md` §9). The node input is then
``\\alpha_k \\odot z + \\sigma_k \\odot \\varepsilon_k`` with α and σ per coordinate, and the weight of coordinate
``i`` is ``\\lambda\\,(\\sigma_t^2/\\alpha_t^2)\\,\\alpha_i/\\sigma_i``, which is RED-Diff's ``\\lambda\\sigma_t/\\alpha_t`` on the
annealed blocks and keeps the field of an energy network a gradient.
"""
struct ImplicitDiffusion{P<:NoisePredictor,T<:Real,L} <: AbstractImplicitRelation
    predictor::P
    nodes::FieldNodes{T}
    λ::T
    batched::Bool
    levels::L
end
function ImplicitDiffusion(pred::NoisePredictor, nodes::FieldNodes{T}; λ = 1.0, batched::Bool = false,
                           levels = nothing) where {T}
    if levels !== nothing
        pred.blocks === nothing && throw(ArgumentError("levels need a predictor with noise blocks (NoisePredictor(...; blocks))"))
        length(levels) == nblocks(pred) || throw(DimensionMismatch("one level per noise block: $(length(levels)) for $(nblocks(pred))"))
        levels = Union{Nothing,T}[l === nothing ? nothing : T(l) for l in levels]
        all(l -> l === nothing || pred.schedule.tmin ≤ l ≤ 1, levels) ||
            throw(ArgumentError("block levels must lie in [tmin, 1]"))
    end
    return ImplicitDiffusion(pred, nodes, T(λ), batched, levels)
end

"""
    conditional_levels(pred::NoisePredictor, ρ; level = tmin) -> Vector

The block levels ([`ImplicitDiffusion`](@ref)'s `levels`) of a query with precisions `ρ`: every
block whose coordinates are all hard-clamped is held at `level` (by default the schedule's
`tmin`, a clean condition), the others are annealed (`nothing`).
"""
function conditional_levels(pred::NoisePredictor, ρ; level = pred.schedule.tmin)
    pred.blocks === nothing && throw(ArgumentError("conditional_levels needs a predictor with noise blocks"))
    return [all(isinf, ρ[b]) ? float(level) : nothing for b in pred.blocks.blocks]
end

_weight(m::ImplicitDiffusion, t) = m.λ * sigma(m.predictor.schedule, t) / alpha(m.predictor.schedule, t)

# the time of each block at a node of level t: annealed blocks at t, held ones at their level
_node_times(m::ImplicitDiffusion, t) = m.levels === nothing ? t : [l === nothing ? t : l for l in m.levels]

# the node input's α and σ and the field's weight at level t: numbers, or with block levels one
# per coordinate (weight λ σ_t²/α_t² · α_i/σ_i, which is λσ_t/α_t on the annealed blocks)
function _node_coefs(m::ImplicitDiffusion, t)
    s = m.predictor.schedule
    m.levels === nothing && return (alpha(s, t), sigma(s, t), _weight(m, t))
    τ = _node_times(m, t)
    own = m.predictor.blocks.owner
    a, σ = alpha.(Ref(s), τ)[own], sigma.(Ref(s), τ)[own]
    return (a, σ, (m.λ * sigma(s, t)^2 / alpha(s, t)^2) .* a ./ σ)
end

# the weight ⊗ α factor of the Jacobian block on idx: λ_i α_j
_rowcol(λc::Number, a::Number, idx) = λc * a
_rowcol(λc, a, idx) = λc[idx] .* a[idx]'

function _node_input(m::ImplicitDiffusion, z, k)
    a, σ, _ = _node_coefs(m, m.nodes.t[k])
    return a .* z .+ σ .* view(m.nodes.ε, :, k)
end

# The nodes evaluated in one network call: one noise level each when batched, else one node.
function _groups(m::ImplicitDiffusion)
    m.batched || return [[k] for k in eachindex(m.nodes.t)]
    ts = m.nodes.t
    return [findall(==(u), ts) for u in unique(ts)]
end
# a vector for a single node (as the predictor has always been called), an n × B matrix when batched
function _group_input(m::ImplicitDiffusion, z, grp)
    m.batched || return _node_input(m, z, only(grp))
    a, σ, _ = _node_coefs(m, m.nodes.t[first(grp)])
    return a .* z .+ σ .* m.nodes.ε[:, grp]
end

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
        _, _, λc = _node_coefs(m, t)
        ε̂, st = epsilon(m.predictor, _group_input(m, z, grp), _node_times(m, t), ps, st)
        g = g .+ λc .* ((reshape(ε̂, length(z), :) .- view(m.nodes.ε, :, grp)) * view(m.nodes.w, grp))
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
        a, _, λc = _node_coefs(m, t)
        J .+= _rowcol(λc, a, idx) .*
              epsilon_jacobian_block(m.predictor, _group_input(m, z, grp), _node_times(m, t), ps, st, idx, m.nodes.w[grp])
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
        a, _, λc = _node_coefs(m, t)
        W = m.batched ? (λc .* w) * transpose(m.nodes.w[grp]) : (λc .* w) .* m.nodes.w[only(grp)]
        X̄ = epsilon_vjp_input(m.predictor, _group_input(m, z, grp), _node_times(m, t), ps, st, W)
        v .+= a .* vec(sum(reshape(X̄, length(z), :); dims = 2))
    end
    return v
end

# any other relation: through its full Jacobian
prior_vjp(m::AbstractImplicitRelation, z, w, ps, st) = prior_jacobian(m, z, ps, st)' * w

"""
    prior_jvp(m, z, v, ps, st) -> Vector

The Jacobian-vector product ``(\\partial g/\\partial z)\\,v`` of the field, without forming the
Jacobian: one input JVP per noise level ([`epsilon_jvp_input`](@ref)), with every node's column
moved along the same `v`. The matrix-free solver ([`KrylovSolve`](@ref)) is built on it and on
[`prior_vjp`](@ref).
"""
function prior_jvp(m::ImplicitDiffusion, z, v, ps, st)
    out = zero(float.(z))
    for grp in _groups(m)
        t = m.nodes.t[first(grp)]
        a, _, λc = _node_coefs(m, t)
        V = m.batched ? repeat(a .* v, 1, length(grp)) : a .* v
        Ẋ = epsilon_jvp_input(m.predictor, _group_input(m, z, grp), _node_times(m, t), ps, st, V)
        out .+= λc .* (reshape(Ẋ, length(z), :) * view(m.nodes.w, grp))
    end
    return out
end

prior_jvp(m::AbstractImplicitRelation, z, v, ps, st) = prior_jacobian(m, z, ps, st) * v

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
    default_tolerance(m, z, ps) -> Real

The solvers' default tolerance for the relation `m` with parameters `ps` at states like `z`:
``\\max(10^{-9},\\ 30\\,\\epsilon)``, with ``\\epsilon`` the machine epsilon of the arithmetic the field is
computed in. On a device ([`NoisePredictor`](@ref)'s `device`) that is the parameters' type, since
the inputs are converted to it. A Float32 network's field has a rounding floor of 10 to 15 ``\\epsilon``
relative (measured on MNIST with 10 and with 784 free coordinates, `implicit.md` §11), so the
tolerance, about ``3.6\\cdot10^{-6}``, sits a factor of two to three above it. On the CPU, Julia
promotes Float32 parameters against Float64 states to Float64, so the tolerance stays ``10^{-9}``.
"""
default_tolerance(m, z, ps) = max(1e-9, 30 * _arith_eps(m, z, ps))

# the machine epsilon of the arithmetic the field is computed in
function _arith_eps(m, z, ps)
    Tp = _coarsest_type(ps)
    T = Tp === nothing ? float(eltype(z)) : _ondevice(m) ? Tp : promote_type(Tp, float(eltype(z)))
    return Float64(eps(T))
end
_coarsest_type(x::AbstractArray{T}) where {T<:AbstractFloat} = T
_coarsest_type(x::AbstractFloat) = typeof(x)
function _coarsest_type(x::Union{NamedTuple,Tuple})
    ts = filter(!isnothing, map(_coarsest_type, collect(values(x))))
    return isempty(ts) ? nothing : ts[argmax(map(eps, ts))]
end
_coarsest_type(x) = nothing
_ondevice(m::ImplicitDiffusion) = _ondevice(m.predictor)
_ondevice(m::AbstractImplicitRelation) = false

"""
    implicit_infer(m, z₀, ρ, ps, st; z_init = z₀, tol = default_tolerance(m, z₀, ps), maxiters = 200, step = 0.05,
                   linsolve = DenseSolve(), continuation = (), method = :auto,
                   radius = nothing) -> (ImplicitSolution, st)

Find a stable root of the residual on the free coordinates, warm-started from `z_init`.

Each iteration takes a **Newton** step ``-J_{FF}^{-1}r_F`` with backtracking on ``\\|r\\|`` when the
symmetric part of ``J_{FF}`` is positive definite, and otherwise a **descent** step
``-\\texttt{step}\\cdot r_F`` along the field (energy descent when the field is a gradient). Newton
is only trusted in locally convex regions so that it is not drawn to saddles and maxima.

`z₀` holds the clamp values on hard coordinates and the soft targets elsewhere; `ρ` is the
precision vector, `Inf` for a hard clamp (cf. [`precision_vector`](@ref)).

`linsolve` chooses the linear algebra: [`DenseSolve`](@ref) forms ``J_{FF}``;
[`KrylovSolve`](@ref) takes the Newton step by GMRES on Jacobian-vector products and tests
stability by Lanczos, without forming it (for many free coordinates).

`continuation = (c₁, c₂, …)` solves first with every node's noise level multiplied by `c₁`
(capped at 1), then by `c₂`, …, each solve started from the previous answer, and last with
`m` itself: a continuation from a smoother relation to the actual one. A smoother field has
fewer, wider basins, so a start far from every basin (a blank image, say) is carried into one
before the sharp field is solved. Each stage takes at most `maxiters` iterations and is used
whether or not it converged; `iters` counts all of them. For an [`ImplicitDiffusion`](@ref).

**Trust region.** `method = :trust_region` replaces the Newton-or-descent iteration by a
trust-region method on a merit function, which converges from far starts and in non-convex
regions where Newton-or-descent stalls (`implicit.md` §11). The default, `method = :auto`, uses
it when the relation has an energy and Newton-or-descent otherwise; `:newton` forces the latter.

- with an energy ([`EnergyNetwork`](@ref), [`GaussianMixtureEps`](@ref), products of them) the merit
  is the query's energy ``Q(z) = U(z) + \\tfrac12\\lVert\\rho_F(z_F - z_{0,F})\\rVert^2``, whose gradient is the
  residual and whose Hessian is ``\\operatorname{sym}J_{FF}``: every accepted step lowers ``Q``, so the
  iteration is not drawn to saddles or maxima;
- without one it is ``\\tfrac12\\lVert r_F\\rVert^2`` with the Gauss–Newton model (Levenberg–Marquardt),
  which can stop at a minimum of the residual that is not a root (then `converged` is false).

Each step minimises the quadratic model within a ball of radius ``\\Delta``: exactly from an
eigendecomposition with [`DenseSolve`](@ref), by Steihaug's truncated conjugate gradients on
Hessian-vector products with [`KrylovSolve`](@ref). ``\\Delta`` starts at `radius` (default
``0.1\\sqrt{|F|}``) and grows or shrinks with the ratio of actual to predicted decrease. `step`
is not used.

`tol` is relative, ``\\lVert r_F\\rVert \\le \\mathrm{tol}\\,(1 + \\lVert z\\rVert)``, and defaults to
[`default_tolerance`](@ref): ``10^{-9}`` in Float64, about ``3.6\\cdot10^{-6}`` for a Float32 network on a device.
"""
function implicit_infer(m::AbstractImplicitRelation, z₀, ρ, ps, st;
                        z_init = z₀, tol = nothing, maxiters::Integer = 200, step = 0.05,
                        linsolve::AbstractLinearSolve = DenseSolve(), continuation = (), method::Symbol = :auto,
                        radius = nothing)
    method in (:auto, :newton, :trust_region) ||
        throw(ArgumentError("method must be :auto, :newton or :trust_region, got :$method"))
    method === :auto && (method = _has_energy(m) ? :trust_region : :newton)
    tol === nothing && (tol = default_tolerance(m, z₀, ps))
    if !isempty(continuation)
        z, its = z_init, 0
        for c in continuation
            sc, st = implicit_infer(_rescaled(m, c), z₀, ρ, ps, st; z_init = z, tol, maxiters, step, linsolve, method, radius)
            z, its = sc.z, its + sc.iters
        end
        sol, st = implicit_infer(m, z₀, ρ, ps, st; z_init = z, tol, maxiters, step, linsolve, method, radius)
        return (ImplicitSolution(sol.z, sol.residual, sol.iters + its, sol.converged, sol.stable), st)
    end
    method === :trust_region && return _infer_trust_region(m, z₀, ρ, ps, st; z_init, tol, maxiters, linsolve, radius)
    free = .!isinf.(ρ)
    z = float.(copy(z_init))
    z[.!free] .= z₀[.!free]
    rnorm(zz) = norm(first(implicit_residual(m, zz, z₀, ρ, ps, st))[free])
    for it in 1:maxiters
        r, st = implicit_residual(m, z, z₀, ρ, ps, st)
        rF = r[free]
        nr = norm(rF)
        sys = _newton_system(linsolve, m, z, ρ, ps, st)
        if nr ≤ tol * (1 + norm(z))
            return (ImplicitSolution(z, nr, it - 1, true, sys.stable), st)
        end
        if sys.stable
            d = -sys.solve(rF)
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
    nr = norm(r[free])
    stable = _newton_system(linsolve, m, z, ρ, ps, st).stable
    return (ImplicitSolution(z, nr, maxiters, nr ≤ tol * (1 + norm(z)), stable), st)
end

# --- the trust-region method ---------------------------------------------------------
# A merit f on the free coordinates with gradient g and model Hessian H at z: the query energy
# (g = r_F, H = sym J_FF) when the relation has one, else ½‖r_F‖² (g = J_FFᵀ r_F, H = J_FFᵀ J_FF).
# Each step minimises gᵀp + ½pᵀHp over ‖p‖ ≤ Δ; Δ follows the ratio of actual to predicted
# decrease (Nocedal & Wright, Algorithm 4.1).
function _infer_trust_region(m, z₀, ρ, ps, st; z_init, tol, maxiters, linsolve, radius)
    free = .!isinf.(ρ)
    nF = count(free)
    z = float.(copy(z_init))
    z[.!free] .= z₀[.!free]
    T = eltype(z)
    energetic = _has_energy(m)
    resid(zz) = first(implicit_residual(m, zz, z₀, ρ, ps, st))[free]
    merit(zz, rF) = energetic ? _query_energy(m, zz, z₀, ρ, ps, st) : sum(abs2, rF) / 2
    Δ = T(radius === nothing ? 0.1 * sqrt(max(nF, 1)) : radius)
    εa = _arith_eps(m, z₀, ps)                         # the arithmetic's precision (Float32 on a device)
    rF = resid(z)
    f = merit(z, rF)
    for it in 0:maxiters
        nr = norm(rF)
        if nr ≤ tol * (1 + norm(z)) || it == maxiters || Δ < 10εa * (1 + norm(z))
            stable = _newton_system(linsolve, m, z, ρ, ps, st).stable
            return (ImplicitSolution(z, nr, it, nr ≤ tol * (1 + norm(z)), stable), st)
        end
        g, H = _tr_model(linsolve, m, z, ρ, ps, st, rF, energetic)
        p = _tr_step(linsolve, g, H, Δ)
        pred = -(dot(g, p) + dot(p, _apply_H(H, p)) / 2)
        zn = copy(z); zn[free] .+= p
        rn = resid(zn)
        fn = merit(zn, rn)
        # near a root the decrease drowns in the rounding of the merit (computed in the network's
        # precision, Float32 on a GPU): then a smaller residual decides
        noisy = pred ≤ 100 * εa * max(abs(f), one(T))
        ratio = noisy ? (norm(rn) < nr ? one(T) : zero(T)) : (f - fn) / pred
        np = norm(p)
        if ratio < 0.25
            Δ = np / 4
        elseif ratio > 0.75 && np ≥ 0.99Δ
            Δ = 2Δ
        end
        if ratio > 1e-4
            z, rF, f = zn, rn, fn
        end
    end
end

# the model's gradient and Hessian: a dense matrix, or an operator v ↦ Hv
function _tr_model(::DenseSolve, m, z, ρ, ps, st, rF, energetic)
    free = .!isinf.(ρ)
    J = prior_jacobian(m, z, ps, st; coords = free) .+ Diagonal(ρ[free] .^ 2)
    energetic && return (rF, Symmetric((J .+ J') ./ 2))
    return (J' * rF, Symmetric(J' * J))
end
function _tr_model(::KrylovSolve, m, z, ρ, ps, st, rF, energetic)
    op = _free_operators(m, z, ρ, ps, st)
    energetic && return (rF, op.S)
    return (op.Jt(rF), v -> op.Jt(op.J(v)))
end
_apply_H(H::AbstractMatrix, p) = H * p
_apply_H(H, p) = H(p)

# the exact step: p(λ) = -(H + λI)⁻¹g with λ ≥ max(0, -λ_min) and ‖p‖ ≤ Δ (Moré–Sorensen, from
# the eigendecomposition; in the hard case a move along the lowest eigenvector to the boundary)
function _tr_step(::DenseSolve, g, H::AbstractMatrix, Δ)
    E = LinearAlgebra.eigen(H)
    λs, V = E.values, E.vectors
    gt = V' * g
    pof(λ) = -(V * (gt ./ (λs .+ λ)))
    λmin = first(λs)
    if λmin > 0
        p = pof(zero(λmin))
        norm(p) ≤ Δ && return p
    end
    lo = max(zero(λmin), -λmin)
    hi = lo + norm(g) / Δ + abs(lo) + one(λmin)
    while norm(pof(hi)) > Δ
        hi *= 2
    end
    if !(norm(pof(lo + sqrt(eps(λmin)) * (1 + abs(lo)))) > Δ)    # hard case: g ⊥ lowest eigenvector
        p = pof(lo + sqrt(eps(λmin)) * (1 + abs(lo)))
        v = V[:, 1]
        τ = _to_boundary(p, v, Δ)
        return p .+ τ .* v
    end
    for _ in 1:100                                     # ‖p(λ)‖ decreases in λ: bisection
        mid = (lo + hi) / 2
        norm(pof(mid)) > Δ ? (lo = mid) : (hi = mid)
    end
    return pof(hi)
end

# Steihaug's truncated conjugate gradients: CG on Hp = -g until it leaves the ball, meets
# negative curvature (then to the boundary along that direction) or converges
function _tr_step(ls::KrylovSolve, g, H, Δ)
    p = zero(g)
    r = copy(g)
    d = -r
    gn = norm(g)
    gn == 0 && return p
    for _ in 1:ls.maxiters
        Hd = _apply_H(H, d)
        κ = dot(d, Hd)
        κ ≤ 0 && return p .+ _to_boundary(p, d, Δ) .* d
        α = dot(r, r) / κ
        pn = p .+ α .* d
        norm(pn) ≥ Δ && return p .+ _to_boundary(p, d, Δ) .* d
        rn = r .+ α .* Hd
        norm(rn) ≤ ls.rtol * gn && return pn
        d = -rn .+ (dot(rn, rn) / dot(r, r)) .* d
        p, r = pn, rn
    end
    return p
end

# τ ≥ 0 with ‖p + τd‖ = Δ
function _to_boundary(p, d, Δ)
    a, b, c = dot(d, d), 2dot(p, d), dot(p, p) - Δ^2
    return (-b + sqrt(max(b^2 - 4a * c, zero(a)))) / (2a)
end

# the relation with every node's level multiplied by c (at most 1): a smoother field
function _rescaled(m::ImplicitDiffusion, c::Real)
    c > 0 || throw(ArgumentError("continuation factors must be positive"))
    nodes = FieldNodes(min.(c .* m.nodes.t, one(eltype(m.nodes.t))), m.nodes.ε, m.nodes.w)
    return ImplicitDiffusion(m.predictor, nodes, m.λ, m.batched, m.levels)
end
_rescaled(m::AbstractImplicitRelation, c) = throw(ArgumentError("continuation is implemented for an ImplicitDiffusion"))

# The linear algebra of one iteration: is sym J_FF positive definite, and d ↦ J_FF⁻¹ d.
function _newton_system(::DenseSolve, m, z, ρ, ps, st)
    free = .!isinf.(ρ)
    Jff = prior_jacobian(m, z, ps, st; coords = free) .+ Diagonal(ρ[free] .^ 2)
    return (stable = _stable(Jff), solve = r -> Jff \ r)
end

function _newton_system(ls::KrylovSolve, m, z, ρ, ps, st)
    op = _free_operators(m, z, ρ, ps, st)
    solve(r) = first(_gmres(op.J, r; rtol = ls.rtol, maxiters = ls.maxiters, restart = ls.restart))
    return (stable = _lambda_min(op.S, op.n, ls) > 0, solve = solve)
end

# λ_F from J_FFᵀ λ_F = b
function _adjoint_solve(::DenseSolve, m, z, ρ, b, ps, st)
    free = .!isinf.(ρ)
    Jff = prior_jacobian(m, z, ps, st; coords = free) .+ Diagonal(ρ[free] .^ 2)
    return Jff' \ b
end

function _adjoint_solve(ls::KrylovSolve, m, z, ρ, b, ps, st)
    op = _free_operators(m, z, ρ, ps, st)
    λ, ok, k = _gmres(op.Jt, b; rtol = ls.rtol, maxiters = ls.maxiters, restart = ls.restart)
    ok || @warn "implicit_pullback: GMRES stopped after $k products without reaching rtol = $(ls.rtol); the gradient is approximate"
    return λ
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
    implicit_pullback(m, sol, z₀, ρ, z̄, ps, st; linsolve = DenseSolve()) -> (z₀ = z̄₀, ρ = ρ̄, ps = ps̄)

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

With `linsolve = KrylovSolve()` the adjoint system is solved by GMRES on vector-Jacobian
products, so ``J_{FF}`` is never formed; it then warns if GMRES stops short of its tolerance.
"""
function implicit_pullback(m::AbstractImplicitRelation, sol::ImplicitSolution, z₀, ρ, z̄, ps, st;
                           linsolve::AbstractLinearSolve = DenseSolve())
    sol.converged || throw(ArgumentError(
        "implicit_pullback at a non-converged state (residual $(sol.residual)): the implicit " *
        "function theorem does not apply. Re-solve, or use an unrolled/one-step surrogate."))
    z = sol.z
    free = .!isinf.(ρ)
    clamped = .!free
    λF = -_adjoint_solve(linsolve, m, z, ρ, z̄[free], ps, st)
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
        _, _, λc = _node_coefs(m, t)
        τ = _node_times(m, t)
        if m.batched                    # the node weights go into the cotangent, one per column
            g = epsilon_vjp_params(m.predictor, X, τ, ps, st, (λc .* λ) * transpose(m.nodes.w[grp]))
            ps̄ = _axpy(1, g, ps̄)
        else
            g = epsilon_vjp_params(m.predictor, X, τ, ps, st, λc .* λ)
            ps̄ = _axpy(m.nodes.w[only(grp)], g, ps̄)
        end
    end
    return ps̄
end

# --- As a factor's inversion ----------------------------------------------------

"""
    ImplicitProx(nodes; λ = 1.0, tol = 1e-9, maxiters = 200, step = 0.05, message = :point,
                 batched = false, linsolve = DenseSolve())

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

`batched` and `linsolve` are passed on to [`ImplicitDiffusion`](@ref) and to the solver: one
network call per noise level, and the matrix-free [`KrylovSolve`](@ref) for factors with many
free coordinates.
"""
struct ImplicitProx{T<:Real,L<:AbstractLinearSolve}
    nodes::FieldNodes{T}
    λ::T
    tol::T
    maxiters::Int
    step::T
    message::Symbol
    batched::Bool
    linsolve::L
    function ImplicitProx(nodes::FieldNodes{T}, λ::T, tol::T, maxiters::Int, step::T, message::Symbol,
                          batched::Bool, linsolve::L) where {T<:Real,L<:AbstractLinearSolve}
        message in (:point, :gaussian, :mixture) ||
            throw(ArgumentError("message must be :point, :gaussian, or :mixture, got :$message"))
        return new{T,L}(nodes, λ, tol, maxiters, step, message, batched, linsolve)
    end
end
ImplicitProx(nodes::FieldNodes{T}; λ = 1.0, tol = 1e-9, maxiters::Integer = 200, step = 0.05,
             message::Symbol = :point, batched::Bool = false,
             linsolve::AbstractLinearSolve = DenseSolve()) where {T} =
    ImplicitProx(nodes, T(λ), T(tol), Int(maxiters), T(step), message, batched, linsolve)

# the solver's keyword arguments, as the factor configures them
_solver_kwargs(p::ImplicitProx) = (tol = p.tol, maxiters = p.maxiters, step = p.step, linsolve = p.linsolve)
