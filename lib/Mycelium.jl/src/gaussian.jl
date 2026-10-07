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
    d = x .- (C \ b.η)
    return -dimension(b) * log(2π) / 2 + logdet(C) / 2 - (d' * (b.Λ * d)) / 2
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
    return n * (1 + log(2π)) / 2 - logdet(C) / 2
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
# ∫ p_c p_d = exp(A(c⊙d) − A(c) − A(d)), with A the log-partition. An improper d (a likelihood
# message) has no A; the overlap is then taken up to the constant A(d), which is the same for
# every component and cancels when the mixture weights are normalised.
function _product(c::GaussianBelief, d::GaussianBelief)
    isproper(c) || throw(ArgumentError("a mixture component must be a proper Gaussian"))
    cd = combine(c, d)
    logZ = logpartition(cd) - logpartition(c) - (isproper(d) ? logpartition(d) : 0.0)
    return cd, logZ
end
