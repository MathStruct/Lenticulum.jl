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
oplus(a::GradedEnergySpace, b::GradedEnergySpace) = GradedEnergySpace(merge(a.parts, b.parts))
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

oplus(a::GradedEnergy, b::GradedEnergy) = GradedEnergy(merge(a.parts, b.parts))
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
    SquaredNorm(M = nothing)

``\\sigma(e) = \\tfrac12 \\|e\\|_M^2``. **Convex, not linear.**

The natural scalarisation for a residual factor: ``r_\\theta(x) \\approx 0`` iff
``\\sigma(r_\\theta(x)) \\approx 0``. Its differential is ``\\mathrm{d}\\sigma(e) = Me``, so the
scalar gradient is ``J^\\top M r`` and the Gauss–Newton metric is ``J^\\top M J`` — both of
which need the Jacobian of the *vector* energy, which is exactly what is lost by
scalarising early.
"""
struct SquaredNorm{M} <: AbstractScalarisation
    M::M
end
SquaredNorm() = SquaredNorm(nothing)
islinear(::SquaredNorm) = false

"""
    GradedScalarisation(parts::NamedTuple)

``\\sigma_{dc}(e_c, e_d) = \\sigma_c(e_c) + \\sigma_d(e_d)``, the scalarisation induced on a
direct sum. Additivity across summands is automatic; linearity is inherited from the parts.
"""
struct GradedScalarisation{names,T<:Tuple} <: AbstractScalarisation
    parts::NamedTuple{names,T}
end
islinear(σ::GradedScalarisation) = all(islinear, values(σ.parts))

oplus(a::GradedScalarisation, b::GradedScalarisation) = GradedScalarisation(merge(a.parts, b.parts))

"""
    scalarise(σ::AbstractScalarisation, e) -> Real

Apply ``\\sigma`` to a (possibly graded) energy.
"""
scalarise(::IdentityScalarisation, e::Real) = e
scalarise(σ::WeightedSum, e) = sum(σ.λ .* e)
scalarise(σ::SquaredNorm{Nothing}, e) = sum(abs2, e) / 2
scalarise(σ::SquaredNorm, e) = dot_quadratic(σ.M, e) / 2
function scalarise(σ::GradedScalarisation{names}, e::GradedEnergy{names}) where {names}
    return sum(map(scalarise, values(σ.parts), values(e.parts)); init = 0.0)
end

dot_quadratic(M, e) = sum(e .* (M * e))

"""
    jensen_gap(σ, samples) -> Real

``\\mathbb{E}[\\sigma(e)] - \\sigma(\\mathbb{E}[e])`` over a collection of energy samples: the
exact discrepancy between the scalar chain rule (AutoBayes Theorem 23) and the multivariate
one.

Zero for a linear `σ`; for `SquaredNorm` it equals ``\\tfrac12 \\operatorname{tr}
\\operatorname{Cov}(e)``, the posterior variance of the upstream factor's loss. Report it —
it is the same kind of object as the mutual information that measures the laxness of the
tensor in AutoBayes' Remark 26.
"""
function jensen_gap(σ::AbstractScalarisation, samples)
    islinear(σ) && return 0.0
    isempty(samples) && return 0.0
    mean_e = reduce(+, samples) * (1 / length(samples))
    mean_σ = sum(Base.Fix1(scalarise, σ), samples) / length(samples)
    return mean_σ - scalarise(σ, mean_e)
end

# --- Factor interface ------------------------------------------------------

"""
    energyspace(factor) -> AbstractEnergySpace

The space ``E_c`` of `factor`'s vector energy. Defaults to `ScalarEnergySpace()`.
"""
energyspace(::AbstractLenticulumFactor) = ScalarEnergySpace()

"""
    scalarisation(factor) -> AbstractScalarisation

The ``\\sigma_c`` used to turn `factor`'s vector loss into a number. Defaults to the
identity, which makes the factor behave exactly as in AutoBayes.
"""
scalarisation(::AbstractLenticulumFactor) = IdentityScalarisation()

"""
    energy(factor, x, a, y, ps, st) -> (𝐥, st)

The **vector** energy ``\\mathbf{l}^c(x, a, y)``, an element of the cone of
[`energyspace`](@ref)`(factor)`.

Pointwise: it is evaluated at samples drawn from the inversion, so it needs no
normalisation. That is why energies compose by direct sum with no expectation involved.
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
"""
function entropy end

"""
    scalar_energy(factor, x, a, y, ps, st) -> (Real, st)

AutoBayes' ``l^c``: the scalarisation of [`energy`](@ref). Provided so that a factor written
against the paper's interface needs no vector machinery.
"""
function scalar_energy(f::AbstractLenticulumFactor, x, a, y, ps, st)
    l, st = energy(f, x, a, y, ps, st)
    return scalarise(scalarisation(f), l), st
end
