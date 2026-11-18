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
combine(a::_EG, b::GaussianBelief) = _extendable(b) || !_soft(a) ?
    (_indefinite(b) ? _potential_conjoin(b, a) : LenticulumCore._conjoin(a, LenticulumCore.extended(b))) :
    combine(GaussianBelief(a), b)
combine(a::GaussianBelief, b::_EG) = _extendable(a) || !_soft(b) ?
    (_indefinite(a) ? _potential_conjoin(a, b) : LenticulumCore._conjoin(LenticulumCore.extended(a), b)) :
    combine(a, GaussianBelief(b))

# A Gaussian with a negative direction (an expectation-propagation site, P27) has no extended form,
# but its product with an extended Gaussian can still be one: the potential [Hᵀx = Hᵀμ_b] ·
# exp((η_a + η_b)ᵀx − xᵀ(Λ_a + Λ_b)x/2), normalisable when the summed precision is positive on what
# b's hard directions leave free (`_from_potential` refuses a negative direction there). Along a
# direction where it is zero the potential must be constant, not exp(linear): checked, since
# `_from_potential` would drop that component.
function _potential_conjoin(a::GaussianBelief, b::_EG)
    H = LenticulumCore.hard_directions(b)
    ηb, Λb = LenticulumCore._canonical_part(b)
    η, Λ = a.η .+ ηb, Matrix(a.Λ) .+ Λb
    # `_psd_split` would count a negative direction as a zero one (flat): refuse it first
    U = LenticulumCore._complement(H)
    LenticulumCore._check_psd(LenticulumCore._symmetric_part(U' * Λ * U),
        "the product of an indefinite Gaussian and an extended Gaussian is not normalisable: its \
         precision is negative on the support")
    r = LenticulumCore._from_potential(Matrix(H'), H' * b.μ, η, Λ;
                                       scale = LinearAlgebra.norm(LenticulumCore.stop_gradient(b.μ)))
    g = r.D' * (η - Λ * r.μ)
    LinearAlgebra.norm(LenticulumCore.stop_gradient(g)) <=
        sqrt(eps(Float64)) * max(LinearAlgebra.norm(LenticulumCore.stop_gradient(η)), 1.0) || throw(ArgumentError(
        "the product of an indefinite Gaussian and an extended Gaussian is not normalisable: it grows \
         exponentially along a direction where its precision is zero"))
    return r
end

# A `GaussianBelief` of dual numbers (ForwardDiff) meets an extended Gaussian in canonical form when
# the extended side has no hard directions: the canonical sum needs no decomposition, and the
# extended form of the dual side would refuse a precision whose rank changes at first order (P25's
# limit), which the canonical sum takes. With hard directions there is no canonical form, and the
# pair is taken as extended Gaussians: the extended form of first-order duals moves at constant rank
# (P4, `LenticulumCore/src/extended_gaussian.jl`, "Derivatives"); nested duals are refused there.
# Before P4 that threw (P1; LP-028, finding 1: a variable that hears a dual flagged message from the
# factor being differentiated and a plain extended one from another factor).
_extendable(b::GaussianBelief) = eltype(b.η) <: AbstractFloat && eltype(b.Λ) <: AbstractFloat
_soft(a::_EG) = size(LenticulumCore.hard_directions(a), 2) == 0
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
belief_distance(a::_EG, b::GaussianBelief) = _extendable(b) || !_soft(a) ?
    belief_distance(a, LenticulumCore.extended(b)) : belief_distance(GaussianBelief(a), b)
belief_distance(a::GaussianBelief, b::_EG) = belief_distance(b, a)
belief_distance(a::_EG, ::LenticulumCore.TrivialBelief) = belief_distance(a, LenticulumCore.extended(LenticulumCore.TrivialBelief(), LenticulumCore.dimension(a), float(eltype(a.μ))))
belief_distance(::LenticulumCore.TrivialBelief, b::_EG) = belief_distance(b, LenticulumCore.TrivialBelief())

# Relative residuals (P9, `relative_residual` in messages.jl): the same groups, each against its
# own scale; the flat-direction projector at scale 1. Mixed pairs are converted as above.
function relative_residual(a::_EG, b::_EG, atol, rtol)
    LenticulumCore.dimension(a) == LenticulumCore.dimension(b) || return Inf
    return max(_group_ratio(a.μ, b.μ, atol, rtol), _group_ratio(a.Σ, b.Σ, atol, rtol),
               _ratio(maximum(abs, a.D * a.D' .- b.D * b.D'; init = 0.0), 1, atol, rtol))
end
relative_residual(a::_EG, b::LenticulumCore.DiracBelief, atol, rtol) = relative_residual(a, LenticulumCore.extended(b), atol, rtol)
relative_residual(a::LenticulumCore.DiracBelief, b::_EG, atol, rtol) = relative_residual(LenticulumCore.extended(a), b, atol, rtol)
relative_residual(a::_EG, b::GaussianBelief, atol, rtol) = _extendable(b) || !_soft(a) ?
    relative_residual(a, LenticulumCore.extended(b), atol, rtol) : relative_residual(GaussianBelief(a), b, atol, rtol)
relative_residual(a::GaussianBelief, b::_EG, atol, rtol) = relative_residual(b, a, atol, rtol)
relative_residual(a::_EG, ::LenticulumCore.TrivialBelief, atol, rtol) =
    relative_residual(a, LenticulumCore.extended(LenticulumCore.TrivialBelief(), LenticulumCore.dimension(a), float(eltype(a.μ))), atol, rtol)
relative_residual(a::LenticulumCore.TrivialBelief, b::_EG, atol, rtol) = relative_residual(b, a, atol, rtol)

# Damping is the geometric mixture new^α · old^(1-α), as for canonical Gaussians: precisions mix,
# a direction flat on one side becomes informative gradually, and hard directions take the
# weighted average of the two values (`LenticulumCore._damp_extended`).
can_damp(::_EG, ::_EG) = true
_damp(a::_EG, b::_EG, α::Real) = LenticulumCore._damp_extended(a, b, α)
# A message that changes form between sweeps (a likelihood that becomes flat or proper as its
# inputs change) is damped as extended Gaussians. Without these methods `damp` fell back to the
# new message undamped, without a sign (P1, `test/forms_crossing.jl`). Against a `GaussianBelief`
# of dual numbers the pair is damped in canonical form, which is the same rule (`_canonical`);
# with hard directions on the extended side the dual side is extended (P4; before, it was not damped).
can_damp(::_EG, ::GaussianBelief) = true
can_damp(::GaussianBelief, ::_EG) = true
_damp(a::_EG, b::GaussianBelief, α::Real) = _extendable(b) || !_soft(a) ?
    LenticulumCore._damp_extended(a, LenticulumCore.extended(b), α) : _damp(GaussianBelief(a), b, α)
_damp(a::GaussianBelief, b::_EG, α::Real) = _extendable(a) || !_soft(b) ?
    LenticulumCore._damp_extended(LenticulumCore.extended(a), b, α) : _damp(a, GaussianBelief(b), α)

"""
    belief_logdensity(b::GaussianBelief, x)

``\\log p_b(x) = -\\tfrac n2\\log 2\\pi + \\tfrac12\\log\\det\\Lambda
 - \\tfrac12 (x-\\mu)^\\top\\Lambda(x-\\mu)``. Requires a proper belief.
"""
function belief_logdensity(b::GaussianBelief, x::AbstractVector)
    C = LenticulumCore._improper_check(b, "belief_logdensity")
    d = x .- LenticulumCore._precision_solve(C, b.η)
    return -dimension(b) * LenticulumCore._log2π(eltype(d)) / 2 + LenticulumCore._precision_logdet(C) / 2 - (d' * (b.Λ * d)) / 2
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
    ld = LenticulumCore._precision_logdet(C)
    return n * (1 + LenticulumCore._log2π(typeof(ld))) / 2 - ld / 2
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
relative_residual(a::GaussianBelief, b::GaussianBelief, atol, rtol) = dimension(a) == dimension(b) ?
    max(_group_ratio(a.η, b.η, atol, rtol), _group_ratio(a.Λ, b.Λ, atol, rtol)) : Inf

# Damping is a convex combination of canonical parameters. Legitimate: the PSD cone is
# convex, so a damped message is still a valid (possibly improper) Gaussian.
can_damp(::GaussianBelief, ::GaussianBelief) = true
function _damp(a::GaussianBelief, b::GaussianBelief, α::Real)
    Λ = α .* a.Λ .+ (1 - α) .* b.Λ
    return GaussianBelief(α .* a.η .+ (1 - α) .* b.η, Λ, _damped_improper(a, b, Λ))
end

# Whether a damped pair of sites keeps the improper declaration. A flag says its precision is
# mathematically singular even when rounding makes it look positive definite, so two flagged
# sites damp to a proper belief only when their numerical nullspaces are genuinely complementary:
# each side singular (the flag's claim, visible in the spectrum) and the sum full rank. Two
# coincident nullspaces, a flag contradicted by a full-rank side, or a singular sum stay improper
# (GH-18).
function _damped_improper(a::GaussianBelief, b::GaussianBelief, Λ)
    (a.improper && b.improper) || return false
    n = size(Λ, 1)
    return !(_rank_deficient(a.Λ, n) && _rank_deficient(b.Λ, n) && !_rank_deficient(Λ, n))
end
_rank_deficient(M, n) = _numerical_rank(M) < n
# The rank of a precision, on its values (`stop_gradient`: a rank has no derivative), with the
# tolerance of the decomposition's element type — a Float32 precision carries Float32 rounding
# (GH-17); duals and anything else keep the Float64 default.
function _numerical_rank(M::AbstractMatrix)
    s = LinearAlgebra.svdvals(LenticulumCore.stop_gradient(M))
    isempty(s) && return 0
    T = eltype(s)
    tol = maximum(size(M)) * (T <: AbstractFloat ? eps(T) : eps(Float64)) * first(s)
    return count(>(tol), s)
end
function _numerical_rank(D::LinearAlgebra.Diagonal)
    d = LenticulumCore.stop_gradient(D.diag)
    isempty(d) && return 0
    T = eltype(d)
    tol = length(d) * (T <: AbstractFloat ? eps(T) : eps(Float64)) * maximum(abs, d)
    return count(x -> abs(x) > tol, d)
end

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
    T = promote_type(eltype(η), eltype(Λ))
    shift = ref === nothing ? zero(T) : LinearAlgebra.dot(d.η, ref) - LinearAlgebra.dot(ref, _sym(d.Λ) * ref) / 2
    LenticulumCore._precision_isproper(Λ) || return GaussianBelief(η, Λ, true), -shift
    cd = GaussianBelief(η, Λ)
    C = LenticulumCore._precision_factor(_sym(Λ))
    logZ = LinearAlgebra.dot(η, LenticulumCore._precision_solve(C, η)) / 2 +
           (dimension(cd) * LenticulumCore._log2π(T) - LenticulumCore._precision_logdet(C)) / 2
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
               (LenticulumCore._precision_logdet(C) + 2 * sum(log, s) + n * LenticulumCore._log2π(eltype(δh))) / 2
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

# --- A Gaussian component times an extended Gaussian (P27) ---------------------------------
#
# A mixture message (a switching factor's) meeting the extended message of a hard relation. The
# extended side d is written in coordinates along its support S = {x : Hᵀx = h}: x = x₀ + U s,
# with U an orthonormal basis of H⊥ (`_hard_reduction`), where its Gaussian part is the canonical
# pair d_s = (η_s, Λ_s), flat where Λ_s is zero. The overlap is that of c with d's law on S:
#
#     log Z = log ∫_S p_c(x) p_d(x) dσ(x),      σ Lebesgue on S in the coordinates s,
#
# the limit of ∫ p_c N(x; μ_d, Σ_d + εI) dx as ε → 0, finite for a proper c (no divergent factor:
# Z is the expectation of p_c under a probability measure on S). A proper c factors on S as the
# density of its marginal Hᵀx at h times its conditional given Hᵀx = h, whose canonical form in s
# is (Uᵀ(η_c − Λ_c x₀), UᵀΛ_cU), so
#
#     log Z = log N(h; Hᵀμ_c, HᵀΣ_cH) + log ∫ c_s(s) d_s(s) ds,
#
# the second term the Gaussian overlap of `_overlap`, exact for d proper on S and up to a constant
# of d alone when d is flat somewhere (as for two Gaussians: the weights of one mixture against
# the one site are right). Neither term subtracts large log-partitions. An improper c (an
# unnormalised exponential) has no marginal: its value at x₀ times the overlap of its restriction,
# exact but evaluated in absolute coordinates (no re-centring: far from the origin it loses digits,
# LP-019). Without hard directions the rule is that of the Gaussian pair. The product itself is
# `combine(c, d)`. Two extended Gaussians that both have hard directions are refused: their
# supports meet in a set of measure zero for both, and the overlap of two singular laws has no
# finite value to compare across pairs.
combine_with_overlap(c::GaussianBelief, d::_EG) = _extended_overlap(c, d)
combine_with_overlap(d::_EG, c::GaussianBelief) = _extended_overlap(c, d)
function combine_with_overlap(a::_EG, b::_EG)
    _soft(b) && return _extended_overlap(GaussianBelief(b), a)
    _soft(a) && return _extended_overlap(GaussianBelief(a), b)
    throw(ArgumentError(
        "no product rule for mixture components that are both extended Gaussians with hard \
         directions: the overlap of two laws on lower-dimensional supports is not finite"))
end

function _extended_overlap(c::GaussianBelief, d::_EG)
    n = dimension(c)
    n == LenticulumCore.dimension(d) || throw(DimensionMismatch(
        "cannot combine beliefs of dimension $n and $(LenticulumCore.dimension(d))"))
    cd = combine(c, d)
    if _soft(d)
        g = GaussianBelief(d)
        return cd, (_improper_first(c, g) ? _overlap(g, c, nothing) : _overlap(c, g, nothing))[2]
    end
    x0, U, ηs, Λs = LenticulumCore._hard_reduction(d)
    H = LenticulumCore.hard_directions(d)
    Λc = _sym(c.Λ)
    flat = size(d.D, 2) > 0
    if isproper(c)
        μc = LenticulumCore.belief_mean(c)
        Σh = LenticulumCore._symmetric_part(H' * LenticulumCore.belief_cov(c) * H)
        δ = H' * (x0 - μc)
        C = LenticulumCore._precision_factor(Σh)
        logZ = -LinearAlgebra.dot(δ, LenticulumCore._precision_solve(C, δ)) / 2 -
               (LenticulumCore._precision_logdet(C) + length(δ) * LenticulumCore._log2π(eltype(δ))) / 2
        size(U, 2) == 0 && return cd, logZ
        cs = GaussianBelief(U' * (c.η - Λc * x0), LenticulumCore._symmetric_part(U' * Λc * U))
        return cd, logZ + _overlap(cs, GaussianBelief(ηs, Λs, flat), nothing)[2]
    end
    logZ = LinearAlgebra.dot(c.η, x0) - LinearAlgebra.dot(x0, Λc * x0) / 2
    size(U, 2) == 0 && return cd, logZ
    cs = GaussianBelief(U' * (c.η - Λc * x0), LenticulumCore._symmetric_part(U' * Λc * U), true)
    ds = GaussianBelief(ηs, Λs, flat)
    return cd, logZ + (isproper(ds) ? _overlap(ds, cs, nothing) : _overlap(cs, ds, nothing))[2]
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

# Expectation propagation with an extended cavity (P27): a switching factor's mixture message on a
# variable a hard relation reaches. The tilted belief is a mixture of extended Gaussians that share
# the cavity's hard directions; its projection is the extended Gaussian of the total moments (zero
# variance along those directions, so they stay hard). A component flat anywhere has no moments:
# `nothing`, the update is skipped. The site p / cavity is defined on the cavity's support S only;
# there both are Gaussian, with log densities ηᵀx − xᵀΛx/2 + const for (η, Λ) their canonical parts
# (`_canonical_part`, zero precision along the hard and flat directions), so the site is the
# Gaussian (η_p − η_c, Λ_p − Λ_c), flat along the hard directions (the cavity imposes them) and
# possibly indefinite, as Gaussian sites are; its product with the cavity is p again
# (`_potential_conjoin`). A projection whose hard directions or hard values differ from the
# cavity's (a component fixed a further direction) has no such site: `nothing`.
function ep_project(m::LenticulumCore.MixtureBelief{<:_EG})
    all(c -> size(c.D, 2) == 0, m.components) || return nothing
    w = LenticulumCore.mixture_weights(m)
    w = w ./ sum(w)
    μ = sum(w[k] .* m.components[k].μ for k in eachindex(w))
    Σ = sum(w[k] .* (m.components[k].Σ .+ (m.components[k].μ .- μ) * (m.components[k].μ .- μ)') for k in eachindex(w))
    return _EG(μ, Matrix(LinearAlgebra.Symmetric(Σ)))
end

function ep_divide(p::_EG, c::_EG)
    n = LenticulumCore.dimension(p)
    n == LenticulumCore.dimension(c) || throw(DimensionMismatch(
        "cannot divide beliefs of dimension $n and $(LenticulumCore.dimension(c))"))
    Hp, Hc = LenticulumCore.hard_directions(p), LenticulumCore.hard_directions(c)
    (size(Hp, 2) == size(Hc, 2) && LinearAlgebra.norm(Hp * Hp' - Hc * Hc') <= sqrt(eps(Float64))) || return nothing
    # and the same values there: a projection fixing them elsewhere has no site either (LP-046, F1)
    δ = LenticulumCore.stop_gradient(Hc' * (p.μ - c.μ))
    LinearAlgebra.norm(δ) <= sqrt(eps(Float64)) * max(LinearAlgebra.norm(LenticulumCore.stop_gradient(c.μ)), 1.0) ||
        return nothing
    ηp, Λp = LenticulumCore._canonical_part(p)
    ηc, Λc = LenticulumCore._canonical_part(c)
    Λ = LenticulumCore._symmetric_part(Λp - Λc)
    return GaussianBelief(ηp - ηc, Λ, size(Hp, 2) > 0 || !LinearAlgebra.isposdef(LinearAlgebra.Symmetric(Matrix(Λ))))
end
ep_divide(p::_EG, ::LenticulumCore.TrivialBelief) = p

# A Gaussian input with a negative direction (beyond rounding). A sign test has no derivative, so it
# runs on values (`stop_gradient` strips ForwardDiff duals, for which `eigvals` has no method).
function _indefinite(b::GaussianBelief)
    Λ = LinearAlgebra.Symmetric(Matrix(LenticulumCore.stop_gradient(b.Λ)))
    E = LinearAlgebra.eigvals(Λ)
    return isempty(E) ? false : minimum(E) < -16 * length(E) * eps(Float64) * maximum(abs, E)
end

# --- Mixture reduction (P17) ---------------------------------------------------------------
# Runnalls' merge and the message of a `MixtureReduction` node for mixtures of Gaussians. They
# were defined in Lenticulum (`beliefs.jl`), which was type piracy: Mycelium alone refused to
# reduce a Gaussian mixture. Lenticulum re-exports `reduce_mixture`.

"""
    reduce_mixture(m::MixtureBelief{<:GaussianBelief}; max_components, prune = 1e-8)
        -> MixtureBelief

Bound the size of a Gaussian mixture, which products make grow multiplicatively. Components
with weight below `prune` (relative to the largest) are dropped; then, while there are more
than `max_components`, the pair whose merge costs least is replaced by the single Gaussian
with their combined weight, mean and covariance. The cost is Runnalls' upper bound on the
Kullback–Leibler divergence the merge introduces,
``\\tfrac12\\bigl[(w_i + w_j)\\log\\det\\Sigma_{ij} - w_i\\log\\det\\Sigma_i - w_j\\log\\det\\Sigma_j\\bigr]``.
Merging preserves the mixture's overall mean and covariance exactly. Pruning and
renormalising changes total variation by at most the removed mass; moment changes need
bounds on component locations and covariances and can be arbitrarily large.
"""
function reduce_mixture(m::LenticulumCore.MixtureBelief{<:GaussianBelief}; max_components::Integer, prune::Real = 1e-8)
    max_components ≥ 1 || throw(ArgumentError("max_components must be at least 1"))
    w = LenticulumCore.mixture_weights(m)
    keep = w .≥ prune * maximum(w)
    ws = w[keep]
    μs = [LenticulumCore.belief_mean(c) for c in m.components[keep]]
    Σs = [LenticulumCore.belief_cov(c) for c in m.components[keep]]
    function merged(i, j)
        wij = ws[i] + ws[j]
        μ = (ws[i] .* μs[i] .+ ws[j] .* μs[j]) ./ wij
        Σ = (ws[i] .* (Σs[i] .+ (μs[i] .- μ) * (μs[i] .- μ)') .+ ws[j] .* (Σs[j] .+ (μs[j] .- μ) * (μs[j] .- μ)')) ./ wij
        return wij, μ, Matrix(LinearAlgebra.Symmetric(Σ))
    end
    while length(ws) > max_components
        best, bi, bj = Inf, 0, 0
        for i in eachindex(ws), j in (i + 1):length(ws)
            wij, _, Σ = merged(i, j)
            cost = (wij * logdet(Σ) - ws[i] * logdet(Σs[i]) - ws[j] * logdet(Σs[j])) / 2
            cost < best && ((best, bi, bj) = (cost, i, j))
        end
        wij, μ, Σ = merged(bi, bj)
        ws[bi], μs[bi], Σs[bi] = wij, μ, Σ
        deleteat!(ws, bj); deleteat!(μs, bj); deleteat!(Σs, bj)
    end
    return LenticulumCore.MixtureBelief([LenticulumCore.Gaussian(μs[k], Σs[k]) for k in eachindex(ws)], ws)
end

"""
    reduce_message(m::MixtureBelief{<:GaussianBelief}, maxcomponents)

The message of a `MixtureReduction(maxcomponents)` node: [`reduce_mixture`](@ref) when `m` has more than
`maxcomponents` components, and the single component when one is left. Runnalls' merge needs
the components' moments, so a mixture with an improper (likelihood) component over the budget
is refused.
"""
function reduce_message(m::LenticulumCore.MixtureBelief{<:GaussianBelief}, maxcomponents::Integer)
    length(m.components) <= maxcomponents && return m
    all(isproper, m.components) || throw(ArgumentError(
        "cannot reduce a mixture with improper (likelihood) components: Runnalls' merge needs their \
         moments. Raise the MixtureReduction() budget to $(length(m.components)), or reduce on a side where the \
         components are proper"))
    r = reduce_mixture(m; max_components = maxcomponents)
    return length(r.components) == 1 ? only(r.components) : r
end
