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
    # A declared-improper site stays declared only beside zero precision: two improper sites can
    # sum to a proper belief, which `isproper` then reads off Λ.
    improper = (a.improper && _zero_precision(b)) || (b.improper && _zero_precision(a))
    return GaussianBelief(a.η + b.η, a.Λ + b.Λ, improper)
end
_zero_precision(b::GaussianBelief) = all(iszero, b.Λ)

# A hard clamp still dominates a Gaussian: it is the Λ → ∞ limit.
combine(a::LenticulumCore.DiracBelief, ::GaussianBelief) = a
combine(::GaussianBelief, b::LenticulumCore.DiracBelief) = b

# Extended Gaussians (`LenticulumCore/src/extended_gaussian.jl`): the product is exact for flat,
# hard and Gaussian directions alike. A Gaussian joins as its extended form. A clamp still
# dominates, but is checked against the hard constraints of the other side first: a clamp that
# violates one is a contradiction, as two unequal clamps are.
const _EG = LenticulumCore.ExtendedGaussian
combine(a::_EG, b::_EG) = LenticulumCore._conjoin(a, b)
combine(a::_EG, b::GaussianBelief) = _extendable(b) ?
    LenticulumCore._conjoin(a, LenticulumCore.extended(b)) : combine(_canonical(a, "combine"), b)
combine(a::GaussianBelief, b::_EG) = _extendable(a) ?
    LenticulumCore._conjoin(LenticulumCore.extended(a), b) : combine(a, _canonical(b, "combine"))

# A `GaussianBelief` of dual numbers (ForwardDiff) has no extended form: the extended form needs an
# eigendecomposition. A pair of it and an extended Gaussian is then taken in canonical form, which
# exists when the extended side has no hard directions; with hard ones there is no common form
# (P1; LP-028, finding 1: a variable that hears a dual flagged message from the factor being
# differentiated and a plain extended one from another factor threw in `combine`).
_extendable(b::GaussianBelief) = eltype(b.η) <: AbstractFloat && eltype(b.Λ) <: AbstractFloat
_soft(a::_EG) = size(LenticulumCore.hard_directions(a), 2) == 0
_canonical(a::_EG, what) = _soft(a) ? GaussianBelief(a) : throw(ArgumentError(
    "$what: an extended Gaussian with hard directions and a GaussianBelief of non-float numbers \
     (ForwardDiff) have no common form"))
combine(a::_EG, b::LenticulumCore.DiracBelief) = (LenticulumCore._conjoin(a, LenticulumCore.extended(b)); b)
combine(a::LenticulumCore.DiracBelief, b::_EG) = (LenticulumCore._conjoin(LenticulumCore.extended(a), b); a)

"""
    variable_entropy(b::ExtendedGaussian)

[`LenticulumCore.support_entropy`](@ref): the entropy relative to Lebesgue measure on the
belief's support. A variable that a hard constraint fixes in some directions has a finite term
in the Bethe counting correction, consistent with the hard factor's own term; a point gets `0`,
as a `DiracBelief` does. Flat directions (an improper marginal) throw. See `gaussian.md`,
"Extended Gaussians".
"""
variable_entropy(b::_EG) = LenticulumCore.support_entropy(b)

"""
    belief_distance(a::ExtendedGaussian, b::ExtendedGaussian)

The largest of ``\\|\\Delta\\mu\\|_\\infty``, ``\\|\\Delta\\Sigma\\|_\\infty`` and ``\\|\\Delta P_D\\|_\\infty``, with
``P_D`` the orthogonal projector onto the flat directions: in moments, which an extended Gaussian
has also where it is hard, and independent of the basis chosen for its flat directions. A
`GaussianBelief`, `DiracBelief` or `TrivialBelief` on the other side is compared as its extended
form, so a message that changes kind still converges.
"""
function belief_distance(a::_EG, b::_EG)
    LenticulumCore.dimension(a) == LenticulumCore.dimension(b) || return Inf
    return max(maximum(abs, a.μ .- b.μ; init = 0.0), maximum(abs, a.Σ .- b.Σ; init = 0.0),
               maximum(abs, a.D * a.D' .- b.D * b.D'; init = 0.0))
end
belief_distance(a::_EG, b::LenticulumCore.DiracBelief) = belief_distance(a, LenticulumCore.extended(b))
belief_distance(a::LenticulumCore.DiracBelief, b::_EG) = belief_distance(LenticulumCore.extended(a), b)
# against a `GaussianBelief` of dual numbers: the canonical distance (see `_canonical`)
belief_distance(a::_EG, b::GaussianBelief) = _extendable(b) ?
    belief_distance(a, LenticulumCore.extended(b)) : belief_distance(_canonical(a, "belief_distance"), b)
belief_distance(a::GaussianBelief, b::_EG) = belief_distance(b, a)
belief_distance(a::_EG, ::LenticulumCore.TrivialBelief) = belief_distance(a, LenticulumCore.extended(LenticulumCore.TrivialBelief(), LenticulumCore.dimension(a)))
belief_distance(::LenticulumCore.TrivialBelief, b::_EG) = belief_distance(b, LenticulumCore.TrivialBelief())

# Damping is the geometric mixture new^α · old^(1-α), as for canonical Gaussians: precisions mix,
# a direction flat on one side becomes informative gradually, and hard directions take the
# weighted average of the two values (`LenticulumCore._damp_extended`).
can_damp(::_EG, ::_EG) = true
_damp(a::_EG, b::_EG, α::Real) = LenticulumCore._damp_extended(a, b, α)
# A message that changes form between sweeps (a likelihood that becomes flat or proper as its
# inputs change) is damped as extended Gaussians. Without these methods `damp` fell back to the
# new message undamped, without a sign (P1, `test/forms_crossing.jl`). Against a `GaussianBelief`
# of dual numbers the pair is damped in canonical form, which is the same rule (`_canonical`);
# with hard directions on the extended side it is not damped (`can_damp` says so).
can_damp(a::_EG, b::GaussianBelief) = _extendable(b) || _soft(a)
can_damp(a::GaussianBelief, b::_EG) = _extendable(a) || _soft(b)
_damp(a::_EG, b::GaussianBelief, α::Real) = _extendable(b) ?
    LenticulumCore._damp_extended(a, LenticulumCore.extended(b), α) : _damp(GaussianBelief(a), b, α)
_damp(a::GaussianBelief, b::_EG, α::Real) = _extendable(a) ?
    LenticulumCore._damp_extended(LenticulumCore.extended(a), b, α) : _damp(a, GaussianBelief(b), α)

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
    GaussianBelief(α .* a.η .+ (1 - α) .* b.η, α .* a.Λ .+ (1 - α) .* b.Λ, a.improper && b.improper)

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

# Two improper components (likelihoods, unnormalised exponentials, as a mixture message to a
# channel with a flat direction carries them): their product is exp((η_c + η_d)ᵀx − xᵀ(Λ_c + Λ_d)x/2)
# exactly, with no overlap to integrate. When that is proper it is normalised and its mass moves
# into logZ; otherwise it stays an unnormalised exponential with logZ = 0. With `ref`, the same
# constant of d is dropped as in the proper case, d at `ref`.
function _improper_overlap(c::GaussianBelief, d::GaussianBelief, ref)
    dimension(c) == dimension(d) || throw(DimensionMismatch(
        "cannot combine beliefs of dimension $(dimension(c)) and $(dimension(d))"))
    η, Λ = c.η + d.η, c.Λ + d.Λ
    shift = ref === nothing ? 0.0 : LinearAlgebra.dot(d.η, ref) - LinearAlgebra.dot(ref, _sym(d.Λ) * ref) / 2
    LenticulumCore._precision_isproper(Λ) || return GaussianBelief(η, Λ, true), -shift
    cd = GaussianBelief(η, Λ)
    C = LenticulumCore._precision_factor(_sym(Λ))
    logZ = LinearAlgebra.dot(η, LenticulumCore._precision_solve(C, η)) / 2 +
           (dimension(cd) * log(2π) - LenticulumCore._precision_logdet(C)) / 2
    return cd, logZ - shift
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
    isproper(c) || isproper(d) || return _improper_overlap(c, d, ref)
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

# --- Expectation propagation (`passing.md`, "Expectation propagation") -----------------

# The projection of a Gaussian mixture: its total mean and covariance. Only proper components
# have moments; a tilted belief with an improper one is not normalisable, and the update is
# skipped (`nothing`).
function ep_project(m::LenticulumCore.MixtureBelief{<:GaussianBelief})
    all(LenticulumCore.isproper, m.components) || return nothing
    w = LenticulumCore.mixture_weights(m)
    w = w ./ sum(w)
    μs = [LenticulumCore.belief_mean(c) for c in m.components]
    μ = sum(w[k] .* μs[k] for k in eachindex(w))
    Σ = sum(w[k] .* (LenticulumCore.belief_cov(m.components[k]) .+ (μs[k] .- μ) * (μs[k] .- μ)') for k in eachindex(w))
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ); check = false)
    LinearAlgebra.issuccess(C) || return nothing
    Λ = Matrix(LinearAlgebra.Symmetric(inv(C)))          # from the factor the test just computed
    return GaussianBelief(C \ μ, Λ)
end

# The site p / π: canonical parameters subtract. A site whose precision is not positive definite
# (rank-deficient, or with a negative direction) is declared improper.
function ep_divide(p::GaussianBelief, c::GaussianBelief)
    dimension(p) == dimension(c) || throw(DimensionMismatch(
        "cannot divide beliefs of dimension $(dimension(p)) and $(dimension(c))"))
    Λ = p.Λ - c.Λ
    return GaussianBelief(p.η - c.η, Λ, !LinearAlgebra.isposdef(LinearAlgebra.Symmetric(Matrix(Λ))))
end
ep_divide(p::GaussianBelief, ::LenticulumCore.TrivialBelief) = p

# A Gaussian input with a negative direction (beyond rounding). A sign test has no derivative, so it
# runs on values (`stop_gradient` strips ForwardDiff duals, for which `eigvals` has no method).
function _indefinite(b::GaussianBelief)
    Λ = LinearAlgebra.Symmetric(Matrix(LenticulumCore.stop_gradient(b.Λ)))
    E = LinearAlgebra.eigvals(Λ)
    return isempty(E) ? false : minimum(E) < -16 * length(E) * eps(Float64) * maximum(abs, E)
end
