# ---------------------------------------------------------------------------
# Gaussian beliefs in CANONICAL (information / natural-parameter) form.
#
#     N⁻¹(η, Λ)  ∝  exp(-½ xᵀΛx + ηᵀx),      Λ = Σ⁻¹,   η = Λμ
#
# The reason for the canonical form is one line:
#
#     combine(N⁻¹(η₁,Λ₁), N⁻¹(η₂,Λ₂))  =  N⁻¹(η₁+η₂, Λ₁+Λ₂)
#
# Pooling beliefs is ADDITION: exact, associative, commutative, and total. `TrivialBelief` is
# (0,0), the additive identity, and a `DiracBelief` is the Λ → ∞ limit — the ρ_in = ∞ hard clamp
# of `Channels and Polarity.md`.
#
# The type lives here, beside the other belief types, so that every package (the diffusion
# factors and implicit layers included) can emit Gaussians; its message rules (`combine`,
# densities, entropy, damping) are in Mycelium's `gaussian.jl`, its projections
# (`moment_match`, `reduce_mixture`) in Lenticulum's `beliefs.jl`.
#
# See `Gaussian Belief.md` and `Beliefs.md` in vault/Factor Graphs/.
# ---------------------------------------------------------------------------

"""
    GaussianBelief(η, Λ)

A Gaussian in canonical form: information vector `η = Λμ` and precision `Λ = Σ⁻¹`.

`Λ` may be **singular or zero**, and that is not an error: a factor → variable message is a
*likelihood*, not a distribution, and a likelihood that constrains only some directions has a
rank-deficient precision. The moment form cannot represent this at all, which is the second
reason for the canonical parametrisation.

Use [`Gaussian`](@ref) to build one from a mean and covariance.
"""
struct GaussianBelief{V<:AbstractVector,M<:AbstractMatrix} <: LenticulumCore.AbstractBelief
    η::V
    Λ::M
    function GaussianBelief(η::V, Λ::M) where {V<:AbstractVector,M<:AbstractMatrix}
        size(Λ, 1) == size(Λ, 2) == length(η) ||
            throw(DimensionMismatch("η has length $(length(η)) but Λ is $(size(Λ))"))
        return new{V,M}(η, Λ)
    end
end

"""
    Gaussian(μ, Σ)
    Gaussian(μ::Real, σ²::Real)

A Gaussian from its **moments**. Converts to canonical form immediately; `Σ` must be positive
definite.
"""
function Gaussian(μ::AbstractVector, Σ::AbstractMatrix)
    Λ = inv(_chol(Σ, "Σ"))
    return GaussianBelief(Λ * μ, Λ)
end
Gaussian(μ::Real, σ²::Real) = Gaussian([float(μ)], fill(float(σ²), 1, 1))

"""
    uninformative(n)

The improper uniform belief on ``\\mathbb{R}^n``: `η = 0`, `Λ = 0`. The identity for
`Mycelium.combine`, and the canonical-form spelling of
`LenticulumCore.TrivialBelief`.
"""
uninformative(n::Int) = GaussianBelief(zeros(n), zeros(n, n))

dimension(b::GaussianBelief) = length(b.η)

_chol(M, name) = try
    cholesky(Symmetric(Matrix(M)))
catch
    throw(ArgumentError("$name is not positive definite: $(M)"))
end

"""
    isproper(b::GaussianBelief) -> Bool

Whether `Λ` is positive definite, i.e. whether the belief is a normalisable distribution
rather than a likelihood. `belief_mean`, `belief_cov`, `variable_entropy` and
`belief_logdensity` all require this.
"""
isproper(b::GaussianBelief) = isposdef(Symmetric(Matrix(b.Λ)))

"""
    belief_mean(b) -> Vector

``\\mu = \\Lambda^{-1}\\eta``. Errors on an improper belief, rather than returning a
least-squares pseudo-mean that would be silently wrong.

Named `belief_mean` rather than extending `Statistics.mean` to avoid a fifth name collision in
this project — see `Mycelium.md` §2.
"""
function belief_mean(b::GaussianBelief)
    C = _improper_check(b, "belief_mean")
    return C \ b.η
end

"""
    belief_cov(b) -> Matrix

``\\Sigma = \\Lambda^{-1}``.
"""
belief_cov(b::GaussianBelief) = inv(_improper_check(b, "belief_cov"))

function _improper_check(b::GaussianBelief, who)
    try
        return cholesky(Symmetric(Matrix(b.Λ)))
    catch
        throw(ArgumentError(
            "$who needs a proper belief, but Λ is singular. This belief is a likelihood \
             message, not a distribution — combine it with a prior first."))
    end
end

"""
    logpartition(b) -> Real

``\\log \\int \\exp(-\\tfrac12 x^\\top\\Lambda x + \\eta^\\top x)\\,dx
 = \\tfrac12\\eta^\\top\\Lambda^{-1}\\eta - \\tfrac12\\log\\det\\Lambda + \\tfrac n2\\log 2\\pi``.
"""
function logpartition(b::GaussianBelief)
    C = _improper_check(b, "logpartition")
    n = dimension(b)
    return (b.η' * (C \ b.η)) / 2 - logdet(C) / 2 + n * log(2π) / 2
end

function Base.show(io::IO, b::GaussianBelief)
    if isproper(b)
        print(io, "Gaussian(μ=", round.(belief_mean(b); digits = 4),
            ", Σ=", round.(belief_cov(b); digits = 4), ")")
    else
        print(io, "GaussianBelief(η=", round.(b.η; digits = 4),
            ", Λ=", round.(b.Λ; digits = 4), "; improper)")
    end
end
