# ---------------------------------------------------------------------------
# The Mycelium belief interface for `LenticulumCore.GaussianBelief`: pooling is addition of
# canonical parameters; densities, entropy, distance and damping in closed form; and the
# product rule that lets Gaussians be mixture components. See `gaussian.md`.
# ---------------------------------------------------------------------------

"""
    combine(a::GaussianBelief, b::GaussianBelief)

**Addition of canonical parameters.** Exact, associative, commutative, total — no
approximation, no density evaluation, no failure case.

This is the operation `messages.md` §1 records as the package's blocking gap. For Gaussians it
is free; for anything else it is importance reweighting. That asymmetry is why Gaussian belief
propagation is the workhorse it is.
"""
function combine(a::GaussianBelief, b::GaussianBelief)
    dimension(a) == dimension(b) || throw(DimensionMismatch(
        "cannot combine beliefs of dimension $(dimension(a)) and $(dimension(b))"))
    return GaussianBelief(a.η + b.η, a.Λ + b.Λ)
end

# A hard clamp still dominates a Gaussian: it is the Λ → ∞ limit.
combine(a::LenticulumCore.DiracBelief, ::GaussianBelief) = a
combine(::GaussianBelief, b::LenticulumCore.DiracBelief) = b

"""
    belief_logdensity(b::GaussianBelief, x)

``\\log p_b(x) = -\\tfrac n2\\log 2\\pi + \\tfrac12\\log\\det\\Lambda
 - \\tfrac12 (x-\\mu)^\\top\\Lambda(x-\\mu)``. Requires a proper belief.
"""
function belief_logdensity(b::GaussianBelief, x::AbstractVector)
    C = LenticulumCore._improper_check(b, "belief_logdensity")
    d = x .- LenticulumCore._precision_solve(C, b.η)
    return -dimension(b) * log(2π) / 2 + LenticulumCore._precision_logdet(C) / 2 - (d' * (b.Λ * d)) / 2
end

"""
    variable_entropy(b::GaussianBelief)

``H = \\tfrac n2(1 + \\log 2\\pi) - \\tfrac12\\log\\det\\Lambda``.

Note this can be **negative** — differential entropy of a concentrated Gaussian is negative,
and the Bethe counting correction of `free_energy.md` depends on the sign being carried
correctly. This is the first belief type for which the correction is not identically zero, and
implementing it is what exposed the sign bug recorded in `free_energy.md` §6.
"""
function variable_entropy(b::GaussianBelief)
    C = LenticulumCore._improper_check(b, "variable_entropy")
    n = dimension(b)
    return n * (1 + log(2π)) / 2 - LenticulumCore._precision_logdet(C) / 2
end

"""
    belief_distance(a::GaussianBelief, b::GaussianBelief)

``\\max(\\|\\Delta\\eta\\|_\\infty, \\|\\Delta\\Lambda\\|_\\infty)`` on the canonical
parameters.

Chosen over a KL divergence because it is defined for **improper** beliefs too, and messages
are routinely improper. A convergence criterion that throws on half the messages is not a
convergence criterion. `Lenticulum.kl_divergence` is available separately when both beliefs are
proper.
"""
function belief_distance(a::GaussianBelief, b::GaussianBelief)
    dimension(a) == dimension(b) && return max(
        maximum(abs, a.η .- b.η; init = 0.0), maximum(abs, a.Λ .- b.Λ; init = 0.0))
    return Inf
end

# Damping is a convex combination of canonical parameters. Legitimate: the PSD cone is
# convex, so a damped message is still a valid (possibly improper) Gaussian.
can_damp(::GaussianBelief, ::GaussianBelief) = true
_damp(a::GaussianBelief, b::GaussianBelief, α::Real) =
    GaussianBelief(α .* a.η .+ (1 - α) .* b.η, α .* a.Λ .+ (1 - α) .* b.Λ)

# Product rule for mixture components: the pooled Gaussian and the log of the overlap
# ∫ p_c p_d. An improper d (a likelihood message exp(ηᵀx − xᵀΛx/2)) has no normaliser; the
# overlap is then ∫ p_c(x) exp(ηᵀx − xᵀΛx/2) dx, which differs from the proper one by a
# constant depending on d alone and cancels when the mixture weights are normalised.
#
# Neither case subtracts log-partitions: A(c⊙d) − A(c) − A(d) cancels terms of size μᵀΛμ/2,
# and at |μ| ~ 1e8 nothing is left (LP-019 item 6). Proper pairs use the convolution
# log N(μ_c − μ_d; 0, Σ_c + Σ_d), with Σ_c + Σ_d scaled by its largest diagonal entry so that
# finite covariances near floatmax do not overflow. An improper d is expanded about μ_c. The
# helpers `_precision_factor`/`_solve`/`_logdet` keep the diagonal device and AD paths.
function combine_with_overlap(c::GaussianBelief, d::GaussianBelief)
    return _overlap(c, d, nothing)
end

_sym(Λ::LinearAlgebra.Diagonal) = Λ
_sym(Λ) = LenticulumCore._symmetric_part(Λ)

# D⁻¹(A + B)D⁻¹ for D = Diagonal(s). Diagonal inputs stay diagonal and use broadcasts only,
# which keeps device arrays and Zygote off scalar indexing.
_scaled_sum(A::LinearAlgebra.Diagonal, B::LinearAlgebra.Diagonal, s) =
    LinearAlgebra.Diagonal(LinearAlgebra.diag(A) ./ s ./ s .+ LinearAlgebra.diag(B) ./ s ./ s)
_scaled_sum(A, B, s) = (A ./ s) ./ s' .+ (B ./ s) ./ s'

# `ref`: re-centre an improper site at this point, i.e. drop the constant ηᵀref − refᵀΛref/2,
# which depends on the site alone. Mixture products pass one shared point per site so that
# the log masses stay O(1) however far the components are from the origin (LP-019 review).
function _overlap(c::GaussianBelief, d::GaussianBelief, ref)
    isproper(c) || throw(ArgumentError("combine_with_overlap requires a proper first Gaussian belief"))
    cd = combine(c, d)
    n = dimension(c)
    μc = LenticulumCore.belief_mean(c)
    if isproper(d)
        Σc, Σd = LenticulumCore.belief_cov(c), LenticulumCore.belief_cov(d)
        # Congruence scaling D⁻¹(Σc + Σd)D⁻¹ with D = diag(√max(Σc_ii, Σd_ii)): every
        # coordinate is O(1), so neither huge nor tiny variances overflow or underflow.
        s = sqrt.(max.(LinearAlgebra.diag(Σc), LinearAlgebra.diag(Σd)))
        # Subtract, then scale, then halve before squaring. Scaling first would round each mean
        # separately (Float32 cancellation, LP-019 fourth review); it is kept only as the
        # fallback when μc − μd itself overflows near ±floatmax.
        μd = LenticulumCore.belief_mean(d)
        δ = μc - μd
        δh = all(isfinite, δ) ? (δ ./ s) ./ 2 : (μc ./ s .- μd ./ s) ./ 2
        C = LenticulumCore._precision_factor(_scaled_sum(Σc, Σd, s))
        logZ = -2 * LinearAlgebra.dot(δh, LenticulumCore._precision_solve(C, δh)) -
               (LenticulumCore._precision_logdet(C) + 2 * sum(log, s) + n * log(2π)) / 2
    else
        # x = μ_c + u: ∫ N(u; 0, Σ_c) exp(rᵀu − uᵀΛu/2) du with r = η − Λμ_c, times the site
        # at μ_c relative to `ref`.
        # The site's quadratic form is that of its symmetric part, which is also what the
        # factorisations use; the raw Λ would add its skew part to r (LP-019 fourth review).
        Λd = _sym(d.Λ)
        Cc = LenticulumCore._improper_check(c, "combine_with_overlap")
        Ccd = LenticulumCore._improper_check(cd, "combine_with_overlap")
        r = d.η - Λd * μc
        u = ref === nothing ? μc : μc - ref
        η0 = ref === nothing ? d.η : d.η - Λd * ref
        logZ = LinearAlgebra.dot(η0, u) - LinearAlgebra.dot(u, Λd * u) / 2 +
               LinearAlgebra.dot(r, LenticulumCore._precision_solve(Ccd, r)) / 2 +
               (LenticulumCore._precision_logdet(Cc) - LenticulumCore._precision_logdet(Ccd)) / 2
    end
    return cd, logZ
end
