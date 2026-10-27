# ---------------------------------------------------------------------------
# The two energies.
#
# AutoBayes (Def. 20) has one energy  l^c : X × ⟦c⟧ × Y -> [0,∞]  composed by ADDITION.
# Lenticulum keeps a vector energy    𝐥^c : X × ⟦c⟧ × Y -> K_c   composed by DIRECT SUM,
# plus a scalarisation σ_c : E_c -> ℝ recovering the paper's law.
#
#   linear σ   =>  σ ∘ compose == compose ∘ σ            (strict)
#   convex σ   =>  σ ∘ compose <= compose ∘ σ            (lax; gap = Jensen term)
#
# See `energy.md` and `Scalar and Multivariate Energy.md`.
# ---------------------------------------------------------------------------

# --- Energy spaces ---------------------------------------------------------

"""
    ScalarEnergySpace()

``E = \\mathbb{R}``, ``K = [0,\\infty)``. Recovers AutoBayes exactly when paired with
[`IdentityScalarisation`](@ref).
"""
struct ScalarEnergySpace <: AbstractEnergySpace end

"""
    EuclideanEnergySpace(n)

``E = \\mathbb{R}^n``. The space a vector residual ``r_\\theta`` lands in; `n` must match
the number of unobserved coordinates for the implicit function theorem to apply.
"""
struct EuclideanEnergySpace <: AbstractEnergySpace
    n::Int
end

"""
    GradedEnergySpace(parts::NamedTuple)

The direct sum ``E_G = \\bigoplus_{f} E_f`` graded by factor name.

This is what a composite factor's energy space *is*: composition does not add, it files the
summands away under their factors' names — exactly as an open model files intermediate
values into ``\\llbracket c \\rrbracket`` rather than integrating them out.
"""
struct GradedEnergySpace{names,T<:Tuple} <: AbstractEnergySpace
    parts::NamedTuple{names,T}
end

"""
    ⊕(a::AbstractEnergySpace, b::AbstractEnergySpace)
    oplus(a, b)

Direct sum of energy spaces. Named summands are merged; unnamed ones are wrapped under
`:left` / `:right`.
"""
function _check_disjoint_keys(a::NamedTuple, b::NamedTuple)
    common = intersect(keys(a), keys(b))
    isempty(common) || throw(ArgumentError(
        "cannot take direct sum ⊕ of graded objects with overlapping keys $(Tuple(common))"))
    return nothing
end

oplus(a::GradedEnergySpace, b::GradedEnergySpace) =
    (_check_disjoint_keys(a.parts, b.parts); GradedEnergySpace(merge(a.parts, b.parts)))
oplus(a::AbstractEnergySpace, b::AbstractEnergySpace) =
    GradedEnergySpace((; left = a, right = b))
const ⊕ = oplus

dimension(::ScalarEnergySpace) = 1
dimension(e::EuclideanEnergySpace) = e.n
dimension(e::GradedEnergySpace) = sum(dimension, values(e.parts); init = 0)

# --- Energy values ---------------------------------------------------------

"""
    GradedEnergy(parts::NamedTuple)

An element of a [`GradedEnergySpace`](@ref): the per-factor breakdown of a composite
energy, *before* it is collapsed to a number.

Per-factor loss attribution is not a logging feature bolted on afterwards — it is what the
composite energy is.
"""
struct GradedEnergy{names,T<:Tuple}
    parts::NamedTuple{names,T}
end

oplus(a::GradedEnergy, b::GradedEnergy) =
    (_check_disjoint_keys(a.parts, b.parts); GradedEnergy(merge(a.parts, b.parts)))
oplus(a, b) = GradedEnergy((; left = a, right = b))

Base.getindex(e::GradedEnergy, name::Symbol) = getfield(e.parts, name)
Base.keys(e::GradedEnergy{names}) where {names} = names

# Vector-space structure, needed because the loss 𝐅 = E[𝐥] - 𝐇 must live in E, and because
# the chain rule takes expectations of energies.
Base.:+(a::GradedEnergy{n}, b::GradedEnergy{n}) where {n} =
    GradedEnergy(NamedTuple{n}(map(+, values(a.parts), values(b.parts))))
Base.:-(a::GradedEnergy{n}, b::GradedEnergy{n}) where {n} =
    GradedEnergy(NamedTuple{n}(map(-, values(a.parts), values(b.parts))))
Base.:*(λ::Real, a::GradedEnergy{n}) where {n} =
    GradedEnergy(NamedTuple{n}(map(Base.Fix1(*, λ), values(a.parts))))
Base.:*(a::GradedEnergy, λ::Real) = λ * a
Base.zero(a::GradedEnergy{n}) where {n} = GradedEnergy(NamedTuple{n}(map(zero, values(a.parts))))

# --- Scalarisations --------------------------------------------------------

"""
    islinear(σ::AbstractScalarisation) -> Bool

Whether ``\\sigma`` commutes with expectation.

This trait is load-bearing, not decorative. With a linear `σ` the composite scalar loss may
be accumulated **eagerly** factor by factor, because
``\\sigma(\\mathbb{E}[\\cdot]) = \\mathbb{E}[\\sigma(\\cdot)]``. With a convex `σ` it must be
**deferred** until the vector loss is assembled, otherwise you silently compute
``\\mathbb{E}[\\sigma(\\mathbf{F})]`` when you asked for ``\\sigma(\\mathbb{E}[\\mathbf{F}])``.
The two differ by [`jensen_gap`](@ref).
"""
function islinear end

"""
    AbstractPointScalarisation <: AbstractScalarisation

A field of scalarisations depending on a factor, its point and parameters. Call
[`localise`](@ref) before [`scalarise`](@ref). It is not a fixed map on energy space,
so [`jensen_gap`](@ref) does not apply.
"""
abstract type AbstractPointScalarisation <: AbstractScalarisation end
islinear(::AbstractPointScalarisation) = false

"""
    localise(σ, f, x, a, y, ps, st) -> AbstractScalarisation

Return an ordinary scalarisation at this point, using the incoming state `st`.
Existing fixed scalarisations return themselves (`===`). Point-dependent subtypes
must implement this method. Graded scalarisations containing point-dependent parts
are rejected: each part needs its own factor and arguments, which cannot in general
be recovered from the enclosing factor's context (including tensor/composed factors).
Localise the parts with their own contexts before constructing a graded scalarisation.
"""
localise(σ::AbstractScalarisation, f, x, a, y, ps, st) = σ
localise(σ::AbstractPointScalarisation, f, x, a, y, ps, st) =
    throw(ArgumentError("$(typeof(σ)) must implement localise(σ, f, x, a, y, ps, st)"))

"""
    IdentityScalarisation()

``\\sigma = \\mathrm{id}`` on `ScalarEnergySpace`. Linear. Recovers AutoBayes' Definition 22
verbatim.
"""
struct IdentityScalarisation <: AbstractScalarisation end
islinear(::IdentityScalarisation) = true

"""
    WeightedSum(λ)

``\\sigma_\\lambda(e) = \\langle \\lambda, e\\rangle`` for `λ` in the dual cone. Linear.

This is where β-VAE weights, KL annealing schedules, curriculum weights and the precisions
``\\rho`` of `ImplicitREDDiff.md` live. Because the energy is kept as a vector, `λ` can be
changed *after* the graph is composed — with a scalar energy each `λ` is a different graph.
"""
struct WeightedSum{W} <: AbstractScalarisation
    λ::W
end
islinear(::WeightedSum) = true

"""
    SquaredNorm(M = nothing; check::Bool = true)

``\\sigma(e) = \\tfrac12 \\|e\\|_M^2``. **Convex, not linear.**

The natural scalarisation for a residual factor: ``r_\\theta(x) \\approx 0`` iff
``\\sigma(r_\\theta(x)) \\approx 0``. Its differential is ``\\mathrm{d}\\sigma(e) = Me``, so the
scalar gradient is ``J^\\top M r`` and the Gauss–Newton metric is ``J^\\top M J`` — both of
which need the Jacobian of the *vector* energy, which is exactly what is lost by
scalarising early.

When `check = true` (the default), validates that `M` is square, symmetric, and positive
semidefinite. Pass `check = false` when `M` is derived from an already-validated SPD
matrix (e.g. `inv(Symmetric(f.Q))`) or inside an AD trace.
"""
struct SquaredNorm{M} <: AbstractScalarisation
    M::M
    function SquaredNorm(M::Nothing; check::Bool = true)
        return new{Nothing}(nothing)
    end
    function SquaredNorm(M::AbstractMatrix; check::Bool = true)
        size(M, 1) == size(M, 2) ||
            throw(DimensionMismatch("SquaredNorm metric M must be square; got $(size(M))"))
        if check && M isa Diagonal
            all(x -> isreal(x) && real(x) >= 0, M.diag) ||
                throw(ArgumentError("SquaredNorm diagonal metric must be real and nonnegative"))
        elseif check && (!(M isa StridedMatrix) ||
                         M isa StridedMatrix{<:Union{AbstractFloat,Integer}})
            scale = maximum(abs, M; init = real(zero(eltype(M))))
            asym = maximum(abs, M .- M'; init = real(zero(eltype(M))))
            asym <= 1e-10 * (1 + scale) ||
                throw(ArgumentError("SquaredNorm metric M must be symmetric; max |M - Mᵀ| = $asym"))
            S = eltype(M) <: Real ? Symmetric(Matrix(float.(M))) : Hermitian(Matrix(float.(M)))
            if !isposdef(cholesky(S; check = false))
                evs = eigvals(S)
                minimum(evs; init = 0.0) >= -1e-10 * (1 + scale) ||
                    throw(ArgumentError(
                        "SquaredNorm metric M must be positive semidefinite; min eigenvalue = $(minimum(evs))"))
            end
        end
        return new{typeof(M)}(M)
    end
end
SquaredNorm(; check::Bool = true) = SquaredNorm(nothing; check = check)
islinear(::SquaredNorm) = false

"""
    SampsonNorm(; wrt = :y, Σ = nothing, λ = 0)

Point-dependent residual scalarisation: localises to `SquaredNorm(inv(A))`, where
`A = J * Σ * J' + λ * sampson_damping(f, x, a, y, ps, st)` and
`J = energy_jacobian(f, x, a, y, ps, st; wrt)`. `Σ = nothing` means identity noise
covariance in data space; otherwise it must be a finite real square PSD matrix.
`wrt` is `:x` or `:y`; damping `λ` must be finite and nonnegative.

Localisation throws `ArgumentError` unless `A` is numerically positive definite:
`λmin(A) > size(A,1) * eps(T) * λmax(A)`, where `T` is its floating element type.
There is no pseudoinverse. Identity damping is the default; gauge-invariant damping
requires a factor override of [`sampson_damping`](@ref). The reference finite-difference
Jacobian is not differentiable; factors may supply an exact Jacobian method.
"""
struct SampsonNorm{C,L<:Real} <: AbstractPointScalarisation
    wrt::Symbol
    Σ::C
    λ::L
    function SampsonNorm(; wrt::Symbol = :y, Σ = nothing, λ::Real = 0)
        wrt in (:x, :y) || throw(ArgumentError("wrt must be :x or :y; got :$wrt"))
        isfinite(λ) && λ >= 0 || throw(ArgumentError("SampsonNorm λ must be finite and nonnegative"))
        if Σ !== nothing
            Σ isa AbstractMatrix{<:Real} || throw(ArgumentError("SampsonNorm Σ must be a real matrix or nothing"))
            all(isfinite, Σ) || throw(ArgumentError("SampsonNorm Σ must be finite"))
            SquaredNorm(Σ)  # same symmetry/PSD checks and shape errors as SquaredNorm
        end
        return new{typeof(Σ),typeof(λ)}(wrt, Σ, λ)
    end
end

"""
    sampson_damping(f, x, a, y, ps, st) -> AbstractMatrix

Residual-space damping covariance for [`SampsonNorm`](@ref), by default the identity
of size `size(energy_jacobian(f, x, a, y, ps, st), 1)`. A factor with a residual gauge
`r → B*r` overrides this with a PSD matrix transforming as `R → B*R*B'`, e.g.
`ps.Θ * ps.Θ'` for `r = Θ*v(z)`. Identity damping is not invariant under general `B`.
"""
function sampson_damping(f, x, a, y, ps, st)
    r = first(energy(f, x, a, y, ps, st))
    return Matrix{float(eltype(r))}(LinearAlgebra.I, length(r), length(r))
end

function localise(σ::SampsonNorm, f, x, a, y, ps, st)
    J = energy_jacobian(f, x, a, y, ps, st; wrt = σ.wrt)
    if σ.Σ !== nothing
        size(σ.Σ, 1) == size(J, 2) || throw(DimensionMismatch(
            "SampsonNorm Σ must match the differentiated data dimension $(size(J, 2))"))
    end
    A = σ.Σ === nothing ? J * J' : J * σ.Σ * J'
    if !iszero(σ.λ)
        R = sampson_damping(f, x, a, y, ps, st)
        size(R) == size(A) || throw(DimensionMismatch("sampson_damping must have size $(size(A))"))
        all(isfinite, R) || throw(ArgumentError("sampson_damping must be finite"))
        SquaredNorm(R)  # reject asymmetric/indefinite damping, even if A would be SPD
        A = A + σ.λ * R
    end
    all(isfinite, A) || throw(ArgumentError("Sampson covariance must be finite"))
    S = Symmetric(A)
    evs = eigvals(S)
    T = eltype(evs)
    threshold = size(S, 1) * eps(T) * maximum(evs; init = zero(T))
    !isempty(evs) && minimum(evs) > threshold || throw(ArgumentError(
        "Sampson covariance must be numerically positive definite: λmin must exceed n*eps(T)*λmax = $threshold; use positive gauge-compatible damping"))
    # Inversion only after the covariance has passed the numerical SPD check.
    M = inv(S)
    all(isfinite, M) || throw(ArgumentError("Sampson precision is not finite after inversion"))
    return SquaredNorm(M; check = false)
end

"""
    GradedScalarisation(parts::NamedTuple)

``\\sigma_{dc}(e_c, e_d) = \\sigma_c(e_c) + \\sigma_d(e_d)``, the scalarisation induced on a
direct sum. Additivity across summands is automatic; linearity is inherited from the parts.
"""
struct GradedScalarisation{names,T<:Tuple} <: AbstractScalarisation
    parts::NamedTuple{names,T}
end
islinear(σ::GradedScalarisation) = all(islinear, values(σ.parts))

_haspoint(::AbstractScalarisation) = false
_haspoint(::AbstractPointScalarisation) = true
_haspoint(σ::GradedScalarisation) = any(_haspoint, values(σ.parts))
function localise(σ::GradedScalarisation, f, x, a, y, ps, st)
    _haspoint(σ) && throw(ArgumentError(
        "cannot localise a GradedScalarisation containing point-dependent parts: localise each part with its own factor and arguments first"))
    return σ
end

oplus(a::GradedScalarisation, b::GradedScalarisation) =
    (_check_disjoint_keys(a.parts, b.parts); GradedScalarisation(merge(a.parts, b.parts)))
oplus(a::AbstractScalarisation, b::AbstractScalarisation) =
    GradedScalarisation((; left = a, right = b))

"""
    scalarise(σ::AbstractScalarisation, e) -> Real

Apply ``\\sigma`` to a (possibly graded) energy.
"""
scalarise(::AbstractPointScalarisation, e) = throw(ArgumentError(
    "point-dependent scalarisation requires localise(σ, f, x, a, y, ps, st) before scalarise"))
scalarise(::IdentityScalarisation, e::Real) = e
function scalarise(::IdentityScalarisation, e::AbstractVector{<:Real})
    length(e) == 1 || throw(ArgumentError(
        "IdentityScalarisation requires a scalar or 1-element vector; got length $(length(e))"))
    return only(e)
end
scalarise(σ::WeightedSum, e) = sum(σ.λ .* e)
scalarise(σ::SquaredNorm{Nothing}, e) = sum(abs2, e) / 2
scalarise(σ::SquaredNorm, e) = dot_quadratic(σ.M, e) / 2
function scalarise(σ::GradedScalarisation{names}, e::GradedEnergy{names}) where {names}
    # Avoid sum(::Tuple; init), which has no working Zygote rule in the tested stack.
    parts = map(scalarise, values(σ.parts), values(e.parts))
    return isempty(parts) ? 0.0 : 0.0 + sum(parts)
end

dot_quadratic(M, e) = dot(e, M, e)

"""
    jensen_gap(σ, samples) -> Real

``\\mathbb{E}[\\sigma(e)] - \\sigma(\\mathbb{E}[e])`` over a collection of energy samples: the
exact discrepancy between the scalar chain rule (AutoBayes Theorem 23) and the multivariate
one.

Only defined for fixed scalarisations; point-dependent parts throw even for empty samples.
Zero for a linear `σ`; for `SquaredNorm()` it equals ``\\tfrac12 \\operatorname{tr}
\\operatorname{Cov}(e)``, the posterior variance of the upstream factor's loss. Report it —
it is the same kind of object as the mutual information that measures the laxness of the
tensor in AutoBayes' Remark 26.
"""
function jensen_gap(σ::AbstractScalarisation, samples)
    _haspoint(σ) && throw(ArgumentError(
        "jensen_gap requires a fixed scalarisation; localise at one fixed point first (a field of metrics has no such Jensen gap)"))
    islinear(σ) && return 0.0
    isempty(samples) && return 0.0
    mean_e = reduce(+, samples) * (1 / length(samples))
    mean_σ = sum(Base.Fix1(scalarise, σ), samples) / length(samples)
    return mean_σ - scalarise(σ, mean_e)
end

# --- Factor interface ------------------------------------------------------

# Two levels, which coincide unless a factor says otherwise (`energy.md` §"Two levels"):
#
#   energyspace / scalarisation                      the FREE energy 𝐅 = E[𝐥] - 𝐇: what composes
#                                                    and what the graph sums
#   pointwise_energyspace / pointwise_scalarisation  the POINTWISE energy 𝐥 = `energy(...)`
#
# AutoBayes has one space for both. A residual factor's pointwise energy is the signed residual
# r ∈ ℝᵏ, scalarised by a norm, while its free energy is graded (fit, complexity, negentropy);
# the two meet in E_b[σ(r)] = fit. Each of the four takes an optional polarity, defaulting to
# the polarity-free answer.

"""
    energyspace(factor[, polarity]) -> AbstractEnergySpace

The space ``E_c`` of `factor`'s vector **free energy** ``\\mathbf{F}^c``, the one that composes
by direct sum and that a graph's free energy is graded by. Defaults to `ScalarEnergySpace()`,
and to the polarity-free space when a polarity is given.

The pointwise energy returned by [`energy`](@ref) lives here too unless the factor says
otherwise with [`pointwise_energyspace`](@ref).
"""
energyspace(::AbstractLenticulumFactor) = ScalarEnergySpace()
energyspace(f::AbstractLenticulumFactor, polarity) = energyspace(f)

"""
    scalarisation(factor[, polarity]) -> AbstractScalarisation

The ``\\sigma_c`` that turns `factor`'s vector free energy into a number. Defaults to the
identity, which makes the factor behave exactly as in AutoBayes, and to the polarity-free
choice when a polarity is given.

A scalarisation can carry parameters (the weights of [`WeightedSum`](@ref), the metric of
[`SquaredNorm`](@ref)); this is the factor's default, and [`scalar_free_energy`](@ref) and
[`scalar_energy`](@ref) accept another as the keyword `σ`.
"""
scalarisation(::AbstractLenticulumFactor) = IdentityScalarisation()
scalarisation(f::AbstractLenticulumFactor, polarity) = scalarisation(f)

"""
    pointwise_energyspace(factor[, polarity]) -> AbstractEnergySpace

The space the **pointwise** energy ``\\mathbf{l}^c`` of [`energy`](@ref) lands in. Defaults to
[`energyspace`](@ref)`(factor)`: one space for both, as in AutoBayes.

A factor overrides it when its pointwise energy is finer than its free energy, e.g. a
residual ``r \\in \\mathbb{R}^k`` (`EuclideanEnergySpace(k)`): signed coordinate by
coordinate, with the minimum at ``r = 0`` rather than at the bottom of a cone. The Jacobian
of that vector is what Gauss–Newton steps, the Fisher metric and the implicit function theorem
need (`Scalar and Multivariate Energy.md` §6).
"""
pointwise_energyspace(f::AbstractLenticulumFactor) = energyspace(f)
pointwise_energyspace(f::AbstractLenticulumFactor, polarity) = pointwise_energyspace(f)

"""
    pointwise_scalarisation(factor[, polarity]) -> AbstractScalarisation

The default scalarisation of the pointwise energy, used by [`scalar_energy`](@ref). Defaults to
[`scalarisation`](@ref)`(factor)`; a residual factor returns a norm such as
`SquaredNorm(Q⁻¹)`.
"""
pointwise_scalarisation(f::AbstractLenticulumFactor) = scalarisation(f)
pointwise_scalarisation(f::AbstractLenticulumFactor, polarity) = pointwise_scalarisation(f)

"""
    energy(factor, x, a, y, ps, st) -> (𝐥, st)

The **vector** energy ``\\mathbf{l}^c(x, a, y)``, an element of
[`pointwise_energyspace`](@ref)`(factor)`.

Pointwise: it is evaluated at samples drawn from the inversion, so it needs no
normalisation. That is why energies compose by direct sum with no expectation involved.

It need not be non-negative: a residual is signed, and "small" means close to 0 under the
factor's scalarisation, not low in a cone (`Scalar and Multivariate Energy.md` §8).
"""
function energy end

"""
    entropy(factor, π, y, ps, st) -> (𝐇, st)

The **vector** entropy ``\\mathbf{H}^c(\\pi, y)``, an element of the same space.

Note the argument type: unlike [`energy`](@ref), this eats a *distribution* `π`, not a
sample. That type difference is exactly why entropies chain (averaged under the downstream
inversion, evaluated at the pushforward prior) while energies merely add.

Usually the entropy is scalar but must be told which coordinate of ``E_c`` it regularises;
returning `h * u` for a fixed direction `u` is the idiomatic way to say so.
For the graded `(fit, complexity, negentropy)` convention, entropy has coordinates
`(0, 0, H)`: `free_energy` subtracts this vector, producing `negentropy = -H`.
"""
function entropy end

"""
    scalar_energy(factor, x, a, y, ps, st; polarity = nothing, σ) -> (Real, st)

AutoBayes' ``l^c``: the scalarisation of [`energy`](@ref). Provided so that a factor written
against the paper's interface needs no vector machinery.

`σ` defaults to [`pointwise_scalarisation`](@ref)`(factor)`, or to
`pointwise_scalarisation(factor, polarity)` when a polarity is given; pass another to scalarise
the same energy differently (a weighted norm, other weights), as `norm(x, p)` takes `p`.
Calls [`localise`](@ref) with the incoming `st` before evaluating and scalarising the energy.
"""
function scalar_energy(f::AbstractLenticulumFactor, x, a, y, ps, st; polarity = nothing,
                       σ = _pointwise_scalarisation(f, polarity))
    fixed = localise(σ, f, x, a, y, ps, st)
    l, st = energy(f, x, a, y, ps, st)
    return scalarise(fixed, l), st
end

_pointwise_scalarisation(f, ::Nothing) = pointwise_scalarisation(f)
_pointwise_scalarisation(f, polarity) = pointwise_scalarisation(f, polarity)
_scalarisation(f, ::Nothing) = scalarisation(f)
_scalarisation(f, polarity) = scalarisation(f, polarity)

"""
    energy_jacobian(factor, x, a, y, ps, st; wrt = :x, ε = 1e-6, ad = nothing) -> Matrix

The Jacobian ``J = \\partial \\mathbf{l}^c / \\partial u`` of the pointwise vector energy
[`energy`](@ref) with respect to `wrt` (`:x`, `:y`, or `:ps`). Parameter columns follow depth-first
tuple field order and column-major array order. This is the reason the pointwise
energy is kept vector-valued (`L-ADR-0002`, `LP-002`): from `J` and a [`SquaredNorm`](@ref)
metric ``M``, [`gauss_newton_metric`](@ref) forms ``J^\\top M J`` without second derivatives.

This is a **reference implementation by central finite differences** (step `ε (1 + |u_j|)`), so
that `LenticulumCore` needs no AD dependency: accurate to about `√eps`, one energy evaluation per
coordinate and side, and not differentiable itself. Pass `ad=AutoForwardDiff()` or another ADTypes backend to use the optional
DifferentiationInterface extension instead. Load that package and the backend first.
"""
function energy_jacobian(
    f::AbstractLenticulumFactor, x, a, y, ps, st; wrt::Symbol = :x, ε::Real = 1e-6, ad = nothing
)
    _check_factor_ad(ad, f, st)
    u = wrt === :ps ? _flatten(ps) : wrt === :x ? x : wrt === :y ? y :
        throw(ArgumentError("wrt must be :x, :y or :ps; got :$wrt"))
    offs = wrt === :ps ? offset_tree(ps) : nothing
    return _jacobian(ad, _energy_at, u, f, x, a, y, ps, offs, st, Val(wrt); ε)
end

# Optional factor/backend compatibility checks run before entering AD.
_check_factor_ad(ad, f) = nothing
_check_factor_ad(ad, f, st) = _check_factor_ad(ad, f)

_energy_vector(r::Number) = [r]
_energy_vector(r::AbstractArray) = vec(r)
_energy_vector(r::GradedEnergy) = reduce(vcat, map(_energy_vector, values(r.parts)); init = Float64[])
# Dispatch on coordinates before entering AD, so Enzyme's static activity analysis
# never has to merge an active vector with a constant input/parameter tree.
_energy_at(u, f, x, a, y, ps, offs, st, ::Val{:ps}) =
    _energy_vector(first(energy(f, x, a, y, _unflatten(u, offs, ps), st)))
_energy_at(u, f, x, a, y, ps, offs, st, ::Val{:x}) =
    _energy_vector(first(energy(f, u, a, y, ps, st)))
_energy_at(u, f, x, a, y, ps, offs, st, ::Val{:y}) =
    _energy_vector(first(energy(f, x, a, u, ps, st)))

# Also used by implicit layers and graph training. Context arguments are DI Constants.
function _jacobian(::Nothing, fun, u, args...; ε = 1e-6)
    r0 = fun(u, args...)
    T = float(promote_type(eltype(r0), eltype(u)))
    J = zeros(T, length(r0), length(u))
    up, um = collect(T, u), collect(T, u)
    for j in eachindex(u)
        h = T(ε) * (one(T) + abs(u[j]))
        up[j], um[j] = u[j] + h, u[j] - h
        J[:, j] .= (fun(up, args...) .- fun(um, args...)) ./ (2h)
        up[j] = um[j] = u[j]
    end
    return J
end
_ad_error(ad) = throw(ArgumentError(
    "AD backend $(typeof(ad)) requires `using DifferentiationInterface` and its backend package " *
    "(e.g. `using ForwardDiff` or `using Zygote`); pass an ADTypes backend object."))
_jacobian(ad, fun, u, args...; ε = 1e-6) = _ad_error(ad)
_gradient(ad, fun, u, args...; ε = 1e-6) = _ad_error(ad)
_scalar_vector(u, fun, args...) = [fun(u, args...)]
_gradient(::Nothing, fun, u, args...; ε = 1e-6) =
    vec(_jacobian(nothing, _scalar_vector, u, fun, args...; ε))

"""
    parameter_gradient(loss, ps; ad = nothing, ε = 1e-6)

Differentiate scalar `loss(ps)` and return the numeric parameter tree's structure.
Arrays use column-major order; tuples/NamedTuples use depth-first field order. Empty
subtrees are preserved. State and data captured by `loss` are held fixed. No state is
committed by this operation. `ad=nothing` uses central differences; an ADTypes object
selects the optional DifferentiationInterface extension and a user-loaded backend.
Only numeric scalar/array leaves and tuple/NamedTuple containers are supported.
"""
parameter_gradient(loss, ps; ad = nothing, ε = 1e-6) =
    _parameter_gradient(ad, _call_loss, ps, loss; ε)
_call_loss(ps, loss) = loss(ps)
_parameter_loss(θ, fun, like, offs, args...) = fun(_unflatten(θ, offs, like), args...)
function _parameter_gradient(ad, fun, ps, args...; ε = 1e-6)
    θ, offs = _flatten(ps), offset_tree(ps)
    isempty(θ) && return _unflatten(θ, offs, ps)
    g = _gradient(ad, _parameter_loss, θ, fun, ps, offs, args...; ε)
    return _unflatten(g === nothing ? zero(θ) : g, offs, ps)
end

"""
    energy_gradient(f, x, a, y, ps, st; ad = nothing, ε = 1e-6, kwargs...)

Parameter gradient of `first(scalar_energy(...; kwargs...))`, shaped like `ps`.
Scalarisation and polarity keywords have exactly `scalar_energy`'s meaning, including
any parameter dependence of a point scalarisation. The incoming state is held fixed.
"""
function energy_gradient(f, x, a, y, ps, st; ad = nothing, ε = 1e-6, kwargs...)
    _check_factor_ad(ad, f, st)
    return _parameter_gradient(ad, _scalar_energy_params, ps, f, x, a, y, st, (; kwargs...); ε)
end
_scalar_energy_params(ps, f, x, a, y, st, kw) = first(scalar_energy(f, x, a, y, ps, st; kw...))

"""
    free_energy_gradient(f, π, y, ps, st; ad = nothing, ε = 1e-6, kwargs...)

Parameter gradient of `first(scalar_free_energy(...; kwargs...))`, shaped like `ps`.
Beliefs and incoming state are held fixed. This differentiates the family-defined free
energy, including expectations if that implementation differentiates them; it does not
supply an estimator for a stochastic inversion.
"""
function free_energy_gradient(f, π, y, ps, st; ad = nothing, ε = 1e-6, kwargs...)
    _check_factor_ad(ad, f, st)
    return _parameter_gradient(ad, _scalar_free_energy_params, ps, f, π, y, st, (; kwargs...); ε)
end
_scalar_free_energy_params(ps, f, π, y, st, kw) = first(scalar_free_energy(f, π, y, ps, st; kw...))

"""
    gauss_newton_metric(σ::SquaredNorm, J::AbstractMatrix) -> Symmetric
    gauss_newton_metric(factor, x, a, y, ps, st; wrt = :x, polarity = nothing, σ, ε = 1e-6) -> Symmetric

The Gauss–Newton / Fisher pull-back metric ``J^\\top M J`` at `(x, a, y)`, where ``J`` is
[`energy_jacobian`](@ref) and ``M`` is the metric of `factor`'s [`SquaredNorm`](@ref)
pointwise scalarisation (`I` when `M === nothing`). Always symmetric positive semidefinite
whenever ``M \\succeq 0``. The factor form first localises `σ`, so it accepts
[`SampsonNorm`](@ref). This freezes the metric at the point: it ignores derivatives
of the metric and is not the full Hessian of the point-dependent scalar energy.
The pull-back `wrt` may differ from `SampsonNorm.wrt`, which defines the noise space.
"""
function gauss_newton_metric(σ::SquaredNorm, J::AbstractMatrix)
    G = σ.M === nothing ? J' * J : J' * σ.M * J
    return Symmetric((G .+ G') ./ 2)
end

function gauss_newton_metric(
    f::AbstractLenticulumFactor, x, a, y, ps, st;
    wrt::Symbol = :x, polarity = nothing,
    σ = _pointwise_scalarisation(f, polarity), ε::Real = 1e-6, ad = nothing,
)
    σ = localise(σ, f, x, a, y, ps, st)
    σ isa SquaredNorm || throw(ArgumentError(
        "gauss_newton_metric requires a SquaredNorm pointwise scalarisation; got $(typeof(σ))"))
    # Preserve existing analytic overrides that predate the optional ad keyword.
    J = ad === nothing ? energy_jacobian(f, x, a, y, ps, st; wrt, ε) :
                         energy_jacobian(f, x, a, y, ps, st; wrt, ε, ad)
    return gauss_newton_metric(σ, J)
end
