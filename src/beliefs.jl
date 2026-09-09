# ---------------------------------------------------------------------------
# Gaussian beliefs in CANONICAL (information / natural-parameter) form.
#
#     N⁻¹(η, Λ)  ∝  exp(-½ xᵀΛx + ηᵀx),      Λ = Σ⁻¹,   η = Λμ
#
# The reason for the canonical form is one line:
#
#     combine(N⁻¹(η₁,Λ₁), N⁻¹(η₂,Λ₂))  =  N⁻¹(η₁+η₂, Λ₁+Λ₂)
#
# Pooling beliefs is ADDITION: exact, associative, commutative, and total. That single fact
# unblocks `Mycelium.combine`, which `messages.md` records as the package's main gap. It also
# makes the two degenerate beliefs land in the right places: `TrivialBelief` is (0,0), the
# additive identity, and a `DiracBelief` is the Λ → ∞ limit — which is exactly the
# ρ_in = ∞ hard clamp of `Channels and Polarity.md`.
#
# See `beliefs.md` and `Gaussian Beliefs.md`.
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
[`Mycelium.combine`](@ref), and the canonical-form spelling of
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

# --- The Mycelium belief interface, now actually implementable ------------

"""
    Mycelium.combine(a::GaussianBelief, b::GaussianBelief)

**Addition of canonical parameters.** Exact, associative, commutative, total — no
approximation, no density evaluation, no failure case.

This is the operation `messages.md` §1 records as the package's blocking gap. For Gaussians it
is free; for anything else it is importance reweighting. That asymmetry is why Gaussian belief
propagation is the workhorse it is.
"""
function Mycelium.combine(a::GaussianBelief, b::GaussianBelief)
    dimension(a) == dimension(b) || throw(DimensionMismatch(
        "cannot combine beliefs of dimension $(dimension(a)) and $(dimension(b))"))
    return GaussianBelief(a.η + b.η, a.Λ + b.Λ)
end

# A hard clamp still dominates a Gaussian: it is the Λ → ∞ limit.
Mycelium.combine(a::LenticulumCore.DiracBelief, ::GaussianBelief) = a
Mycelium.combine(::GaussianBelief, b::LenticulumCore.DiracBelief) = b

"""
    Mycelium.belief_logdensity(b::GaussianBelief, x)

``\\log p_b(x) = -\\tfrac n2\\log 2\\pi + \\tfrac12\\log\\det\\Lambda
 - \\tfrac12 (x-\\mu)^\\top\\Lambda(x-\\mu)``. Requires a proper belief.
"""
function Mycelium.belief_logdensity(b::GaussianBelief, x::AbstractVector)
    C = _improper_check(b, "belief_logdensity")
    d = x .- (C \ b.η)
    return -dimension(b) * log(2π) / 2 + logdet(C) / 2 - (d' * (b.Λ * d)) / 2
end

"""
    Mycelium.variable_entropy(b::GaussianBelief)

``H = \\tfrac n2(1 + \\log 2\\pi) - \\tfrac12\\log\\det\\Lambda``.

Note this can be **negative** — differential entropy of a concentrated Gaussian is negative,
and the Bethe counting correction of `free_energy.md` depends on the sign being carried
correctly. This is the first belief type for which the correction is not identically zero, and
implementing it is what exposed the sign bug recorded in `free_energy.md` §6.
"""
function Mycelium.variable_entropy(b::GaussianBelief)
    C = _improper_check(b, "variable_entropy")
    n = dimension(b)
    return n * (1 + log(2π)) / 2 - logdet(C) / 2
end

"""
    Mycelium.belief_distance(a::GaussianBelief, b::GaussianBelief)

``\\max(\\|\\Delta\\eta\\|_\\infty, \\|\\Delta\\Lambda\\|_\\infty)`` on the canonical
parameters.

Chosen over a KL divergence because it is defined for **improper** beliefs too, and messages
are routinely improper. A convergence criterion that throws on half the messages is not a
convergence criterion. [`kl_divergence`](@ref) is available separately when both beliefs are
proper.
"""
function Mycelium.belief_distance(a::GaussianBelief, b::GaussianBelief)
    dimension(a) == dimension(b) && return max(
        maximum(abs, a.η .- b.η; init = 0.0), maximum(abs, a.Λ .- b.Λ; init = 0.0))
    return Inf
end

"""
    kl_divergence(q::GaussianBelief, p::GaussianBelief) -> Real

``D_{KL}(q \\,\\|\\, p)``. Both must be proper. Used to measure the laxness quantities the
framework keeps insisting should be reported rather than hidden — see
`Composition of Bayesian Lenses.md` Remark 16.
"""
function kl_divergence(q::GaussianBelief, p::GaussianBelief)
    Cq, Cp = _improper_check(q, "kl_divergence"), _improper_check(p, "kl_divergence")
    n = dimension(q)
    Σq = inv(Cq)
    d = (Cq \ q.η) .- (Cp \ p.η)
    return (tr(p.Λ * Σq) + d' * (p.Λ * d) - n + logdet(Cq) - logdet(Cp)) / 2
end

# Damping is a convex combination of canonical parameters. Legitimate: the PSD cone is
# convex, so a damped message is still a valid (possibly improper) Gaussian.
Mycelium.can_damp(::GaussianBelief, ::GaussianBelief) = true
Mycelium._damp(a::GaussianBelief, b::GaussianBelief, α::Real) =
    GaussianBelief(α .* a.η .+ (1 - α) .* b.η, α .* a.Λ .+ (1 - α) .* b.Λ)
