# ---------------------------------------------------------------------------
# A nonlinear Gaussian factor from a hand-written function:  p(y | x) = N(y; f(x), Q)
#
# The measurement models and dynamics of filtering and SLAM: f is known (physics, geometry),
# not learned. Its messages are not Gaussian, so they are approximated by a Gaussian through a
# local LINEAR model y ≈ A x + b + noise(Ω), after which everything is GaussianFactor's algebra:
#
#   Linearization()   A = ∂f(x̄), b = f(x̄) - A x̄, Ω = 0                     (EKF)
#   Unscented()       A, b, Ω by statistical linear regression on sigma points  (UKF)
#
# The expansion point x̄ is the incoming belief on x going forwards, and the cavity (the
# prior argument, the other messages into x) going backwards, so loopy or repeated sweeps
# relinearise as the beliefs sharpen. `invert` iterates that relinearisation at the posterior
# (`iterations`), which is Gauss–Newton on the MAP: the iterated EKF / posterior linearisation.
#
# See `nonlinear.md` and `Open Problems in Implicit Diffusion Learning.md` I11.
# ---------------------------------------------------------------------------

"""
    Linearization(; iterations = 1)

Approximate a nonlinear factor's messages by its first-order Taylor expansion at the expansion
point: the extended Kalman filter's rule. `iterations > 1` relinearises `invert`'s backward
direction at the posterior mean, which is Gauss–Newton on the MAP (the iterated EKF).
`likelihood_message` returns the final site used by this update.
"""
Base.@kwdef struct Linearization
    iterations::Int = 1
end

"""
    Unscented(; α = 1.0, β = 2.0, κ = 0.0, iterations = 1)

Approximate a nonlinear factor's messages by the unscented transform: ``2n+1`` sigma points of
the expansion belief, pushed through ``f``, fitted by a linear model with a residual covariance
(statistical linear regression). Exact for linear ``f``, and second-order accurate in the mean
for smooth ``f``; it needs a covariance at the expansion point (from the incoming belief), and
falls back to linearisation at a point. `α, β, κ` are the usual scaling parameters (Wan & van
der Merwe 2000); `iterations` as for [`Linearization`](@ref) (posterior linearisation).

The sigma points lie at ``m \\pm \\alpha\\sqrt{n + \\kappa}`` standard deviations. The default
`α = 1` spreads them over the belief, so the regression sees the nonlinearity at the scale of the
uncertainty and the centre weight is not negative. Wan & van der Merwe's `α = 1e-3` puts them a
thousandth of a standard deviation out, which makes the transform a local, second-order
linearisation with weights of order ``10^6`` (review of 2026-11-10).
"""
Base.@kwdef struct Unscented
    α::Float64 = 1.0
    β::Float64 = 2.0
    κ::Float64 = 0.0
    iterations::Int = 1
end

"""
    NonlinearFactor(f, nin => nout; noise = 0.1, channels = (:x, :y), method = Linearization(),
                    jacobian = nothing, x0 = nothing)

The nonlinear Gaussian conditional ``p(y \\mid x) = \\mathcal N(y;\\ f(x),\\ Q)`` for a **given**
function `f` (a measurement model, known dynamics): the factor that turns Lenticulum's graphs
into extended or unscented Kalman filters and smoothers, and that sits beside a learned factor
in one graph.

- `noise` is `Q` (a matrix, a vector of variances, or a scalar variance), as in
  [`GaussianFactor`](@ref).
- `method` is [`Linearization`](@ref) (EKF) or [`Unscented`](@ref) (UKF).
- `jacobian(x)` is ``\\partial f/\\partial x`` (an `nout × nin` matrix); without it, central
  differences. The package depends on no AD backend; pass e.g.
  `x -> ForwardDiff.jacobian(f, x)` for an exact one.
- `x0` is the expansion point used when nothing is known about `x`: for a backward message with
  no cavity belief, and for a forward message with no input on `x` (the linear model at `x0` with
  `x` flat, which fixes `y` off the tangent plane of `f`'s image). Without it such messages are
  uninformative.

Both polarities are supported, like `GaussianFactor`; with `x0` also `x` `Latent()`, `y` the target. It has no parameters (`f` is fixed).
Messages are approximate (`SolverInversion` marks that), exact when `f` is affine.
"""
struct NonlinearFactor{F,J,Qm<:AbstractMatrix,M,X} <: LenticulumCore.AbstractLenticulumFactor
    f::F
    jacobian::J
    nin::Int
    nout::Int
    in_channel::Symbol
    out_channel::Symbol
    Q::Qm
    method::M
    x0::X
end

function NonlinearFactor(f, dims::Pair{Int,Int}; noise = 0.1, channels::Tuple{Symbol,Symbol} = (:x, :y),
                         method = Linearization(), jacobian = nothing, x0 = nothing)
    nin, nout = dims
    Q = _as_cov(noise, nout)
    size(Q) == (nout, nout) || throw(ArgumentError("noise covariance must have size ($nout, $nout), matching the residual dimension"))
    # Exact `issymmetric` rejects J S Jᵀ, which rounds asymmetrically (LP-019 follow-up).
    LenticulumCore._nearly_symmetric(Q) || throw(ArgumentError("noise covariance must be symmetric"))
    Q = _matrix(LenticulumCore._symmetric_part(Q))   # one consistent matrix from here on
    (_device_leaf(Q) === nothing ? isposdef(Symmetric(Q)) : LenticulumCore._precision_isproper(Q)) || throw(ArgumentError("noise covariance Q must be positive definite"))
    method isa Union{Linearization,Unscented} ||
        throw(ArgumentError("method must be Linearization() or Unscented(), got $(typeof(method))"))
    x0 === nothing || length(x0) == nin || throw(DimensionMismatch("x0 must have length $nin"))
    return NonlinearFactor(f, jacobian, nin, nout, channels[1], channels[2], Q, method,
                           x0 === nothing ? nothing : _vec(x0))
end

LuxCore.initialparameters(::AbstractRNG, ::NonlinearFactor) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::NonlinearFactor) = NamedTuple()
LuxCore.parameterlength(::NonlinearFactor) = 0

LenticulumCore.channels(f::NonlinearFactor) = (
    LenticulumCore.Channel(f.in_channel, f.nin),
    LenticulumCore.Channel(f.out_channel, f.nout),
)
# With an expansion point `x0`, also x `Latent()` and y the target: the forward message with no
# input on x, that of the linearisation at x0 (P28; nonlinear.md, "No input on x"). Without x0 that
# message is `TrivialBelief()`, which Mycelium sends for an unsupported polarity anyway.
function LenticulumCore.supported_polarities(f::NonlinearFactor)
    O, U, L = LenticulumCore.Observed(), LenticulumCore.Unobserved(), LenticulumCore.Latent()
    ks = (f.in_channel, f.out_channel)
    two = (LenticulumCore.Polarity(NamedTuple{ks}((O, U))),
           LenticulumCore.Polarity(NamedTuple{ks}((U, O))))
    return f.x0 === nothing ? two : (two..., LenticulumCore.Polarity(NamedTuple{ks}((L, U))))
end
function LenticulumCore.supports_polarity(f::NonlinearFactor, p::LenticulumCore.Polarity)
    keys(p) == (f.in_channel, f.out_channel) || return false
    pi_, po = p[f.in_channel], p[f.out_channel]
    return (pi_ isa LenticulumCore.Observed && po isa LenticulumCore.Unobserved) ||
           (pi_ isa LenticulumCore.Unobserved && po isa LenticulumCore.Observed) ||
           (f.x0 !== nothing && pi_ isa LenticulumCore.Latent && po isa LenticulumCore.Unobserved)
end
LenticulumCore.islearnable(::NonlinearFactor) = false
LenticulumCore.energyspace(::NonlinearFactor) = LenticulumCore.energyspace(GaussianFactor(1 => 1))
LenticulumCore.scalarisation(::NonlinearFactor) = LenticulumCore.scalarisation(GaussianFactor(1 => 1))

"""
    residual(f::NonlinearFactor, x, y) -> Vector

``r = y - f(x)``, the factor's vector energy.
"""
residual(f::NonlinearFactor, x, y, ps = NamedTuple()) = _vec(y) .- _vec(f.f(_vec(x)))
LenticulumCore.energy(f::NonlinearFactor, x, a, y, ps, st) = (residual(f, x, y), st)
# pointwise: the residual with SquaredNorm(Q⁻¹), as for GaussianFactor; the graded spaces above
# are those of the free energy
LenticulumCore.pointwise_energyspace(f::NonlinearFactor) = LenticulumCore.EuclideanEnergySpace(f.nout)
LenticulumCore.pointwise_scalarisation(f::NonlinearFactor) =
    LenticulumCore.SquaredNorm(_symmetric_inv(f.Q); check = false)

# --- The local linear model --------------------------------------------------------

# Generic in the element type: at a dual expansion point (ForwardDiff) the Jacobian is a matrix of
# duals, whose partials are the central differences of the derivative (P21). The step takes the
# values only, so it does not carry a derivative of its own.
function _jacobian(f::NonlinearFactor, x)
    if f.jacobian !== nothing
        J = f.jacobian(x)
        return _at_eltype(promote_type(eltype(x), eltype(J)), J)
    end
    J, R = _difference_jacobian(f, x, 1)
    return _rank_clean(f, x, J, R)
end

# Central differences with the step `scale · 1e-6 (1 + |xᵢ|)`, and a bound on the rounding error of
# each entry: the rounding of f's values over the step, eps (|f⁺ⱼ| + |f⁻ⱼ|) / 2hᵢ, and of the shifted
# point, eps |xᵢ| / hᵢ · |Jⱼᵢ|.
function _difference_jacobian(f::NonlinearFactor, x, scale)
    T = eltype(x)
    V = eltype(LenticulumCore.stop_gradient(x))
    J = _zeros_like(x, T, f.nout, f.nin)
    R = _zeros_like(x, V, f.nout, f.nin)
    # On a device the steps are chosen on the host (one copy of x) and each step vector is moved
    # to the device; the CPU path is unchanged (LP-092).
    dev = _device_leaf(x)
    xh = dev === nothing ? nothing : Array(LenticulumCore.stop_gradient(x))
    for i in 1:f.nin
        xi = dev === nothing ? abs(LenticulumCore.stop_gradient(x[i])) : abs(xh[i])
        h = scale * (V === Float64 ? V(1e-6) : cbrt(eps(V))) * (1 + xi)
        e = dev === nothing ? (e = _zeros_like(x, V, f.nin); e[i] = h; e) :
            _on_device(dev, setindex!(zeros(V, f.nin), h, i))
        fp, fm = _vec(f.f(x .+ e)), _vec(f.f(x .- e))
        if i == 1
            # Learn the function's output type from the first difference pair: no extra f call.
            JT = promote_type(T, eltype(fp), eltype(fm))
            RT = promote_type(V, eltype(LenticulumCore.stop_gradient(fp)), eltype(LenticulumCore.stop_gradient(fm)))
            JT === T || (J = _zeros_like(x, JT, f.nout, f.nin))
            RT === V || (R = _zeros_like(x, RT, f.nout, f.nin))
        end
        J[:, i] = (fp .- fm) ./ (2h)
        R[:, i] = eps(V) .* ((abs.(LenticulumCore.stop_gradient(fp)) .+ abs.(LenticulumCore.stop_gradient(fm))) ./ (2h) .+
                                   xi / h .* abs.(LenticulumCore.stop_gradient(J[:, i])))
    end
    return J, R
end

# P29 (nonlinear.md, "The rank of a difference Jacobian"): central differences are accurate to
# about 1e-10 relative, so a Jacobian of rank r below min(nout, nin) comes out with full numerical
# rank, and the rank tests of the messages (eps-relative) lose what f does not depend on. So the
# singular values below the difference error are cut. The error is bounded per entry, B = R + |J(h)
# − J(2h)|/3 (rounding, and truncation by Richardson), and the rank is decided in the coordinates
# where every entry's bound is at most 1: J̃ = D_r⁻¹ J D_c⁻¹, D_r and D_c diagonal powers of two with
# B ≤ r cᵀ, so ‖error(J̃)‖₂ ≤ √(mn), and the singular values of J̃ below κ√(mn) are cut. Rows of very
# different size, or steps of very different size, then do not hide a small singular value that
# its own entries resolve (review of P29). The cut is the oblique projection P = D_c⁻¹ Ṽ Ṽᵀ D_c on the
# right: D_r⁻¹ (J P) D_c⁻¹ is J̃'s truncated SVD, and J P has exactly rank r for the tests
# downstream. With dual numbers P is a constant and (J − value(J))(I − P), whose value is exactly
# zero, gives back the derivative of J in the cut directions, which matters where the rank changes
# at the point (f = x.^2 at 0). J(2h) is computed only in a gray zone (a singular value of J below
# 1e-3 σ₁, or of the rounding-scaled J̃ below 10³√(mn)); a well-conditioned J costs no extra
# evaluation of f. A non-finite J is returned as it is. κ = 10, measured in
# bench/p29_difference_rank.jl.
const _FD_RANK_KAPPA = 10.0

# powers of two r, c with B ≤ r cᵀ (each column, then each row, at its largest bound); a zero bound
# is floored at eps times the largest
function _error_scales(B)
    pow2(v) = exp2(ceil(log2(v)))
    m = maximum(B)
    c = [pow2(max(maximum(view(B, :, i)), eps(eltype(B)) * m)) for i in axes(B, 2)]
    Bc = B ./ c'
    r = [pow2(max(maximum(view(Bc, j, :)), eps(eltype(B)))) for j in axes(B, 1)]
    return r, c
end

function _rank_clean(f::NonlinearFactor, x, J, R)
    isempty(J) && return J
    # The rank decision is made on the host (values only; R carries none); the projector goes back
    # to J's array type. Broadcasting the host Jv against a device R or J fails on CUDA (Q8).
    Jv = Matrix(LenticulumCore.stop_gradient(J))
    R = Matrix(R)
    all(isfinite, Jv) && all(isfinite, R) || return J
    maximum(R) == 0 && return J                        # f's values exactly zero: nothing rounded
    s = LinearAlgebra.svdvals(Jv)
    lim = sqrt(eltype(Jv)(length(J)))
    r0, c0 = _error_scales(R)
    if !(s[end] <= oftype(s[1], 1e-3) * s[1])
        last(LinearAlgebra.svdvals(Jv ./ (r0 * c0'))) > oftype(lim, 1e3) * lim && return J
    end
    J2, _ = _difference_jacobian(f, x, 2)
    B = R .+ abs.(Jv .- Matrix(LenticulumCore.stop_gradient(J2))) ./ 3
    all(isfinite, B) || return J
    rs, cs = _error_scales(B)
    F = _host_spectral(A -> (G = LinearAlgebra.svd(A); (U = G.U, S = G.S, V = G.V)), Jv ./ (rs * cs'))
    k = count(>(eltype(Jv)(_FD_RANK_KAPPA) * lim), F.S)
    k == length(F.S) && return J
    V = F.V[:, 1:k]
    device = _device_leaf(J)
    P = _on_device(device, Diagonal(1 ./ cs) * (V * V') * Diagonal(cs))
    eltype(J) === eltype(Jv) && return J * P
    return J * P .+ (J .- _on_device(device, Jv)) * (I - P)
end

"""
    linearize(f::NonlinearFactor, m, S) -> (A, b, Ω)

The local linear model ``y \\approx Ax + b + \\nu``, ``\\nu\\sim\\mathcal N(0, \\Omega + Q)``, of the factor
around the belief ``\\mathcal N(m, S)`` on ``x``: the Jacobian at ``m`` (``\\Omega = 0``) for
[`Linearization`](@ref); the statistical linear regression of ``f`` on the sigma points of
``\\mathcal N(m, S)`` for [`Unscented`](@ref), whose ``\\Omega`` is the part of ``f``'s spread the line does
not explain. Every message of the factor is `GaussianFactor`'s message for ``(A, b, Q + \\Omega)``.
"""
linearize(f::NonlinearFactor, m, S) = _linearize(f.method, f, _vec(m), S)

function _linearize(::Linearization, f, m, S)
    A = _jacobian(f, m)
    return (A, _vec(f.f(m)) .- A * m, _zeros_like(A, eltype(A), f.nout, f.nout))
end

function _linearize(u::Unscented, f, m, S)
    all(iszero, S) && return _linearize(Linearization(), f, m, S)
    n = length(m)
    T = promote_type(eltype(m), eltype(S))
    α, β, κ = T(u.α), T(u.β), T(u.κ)
    λ = α^2 * (n + κ) - n
    E = _scalar_or_eigen(Symmetric(_matrix(S)))
    vals = max.(E.values, false)
    L = E.vectors * Diagonal(sqrt.(vals))                          # S = L Lᵀ, also when singular
    c = sqrt(n + λ)
    X = hcat(m, m .+ c .* L, m .- c .* L)                          # 2n + 1 sigma points
    if _device_leaf(m) === nothing
        wm = vcat(λ / (n + λ), fill!(similar(m, T, 2n), one(T) / (2(n + λ))))
        wc = copy(wm); wc[1] += 1 - α^2 + β
    else
        tail = fill!(similar(m, T, 2n), one(T) / (2(n + λ)))
        wm = vcat(fill!(similar(m, T, 1), λ / (n + λ)), tail)
        wc = vcat(fill!(similar(m, T, 1), λ / (n + λ) + (1 - α^2 + β)), tail)
    end
    Y = reduce(hcat, (_vec(f.f(x)) for x in eachcol(X)))
    ȳ = Y * wm
    Dx, Dy = X .- m, Y .- ȳ
    Cxy = (Dx .* wc') * Dy'                                        # n × nout
    Syy = (Dy .* wc') * Dy'
    tol = n * eps(eltype(LenticulumCore.stop_gradient(E.values))) * maximum(vals; init = zero(eltype(vals)))
    inv_vals = _device_leaf(vals) === nothing ? [v > tol ? one(v) / v : false for v in vals] :
               map(v -> v > tol ? inv(v) : zero(v), vals)
    Sinv_Cxy = E.vectors * (inv_vals .* (E.vectors' * Cxy))        # reuse E instead of rank + pinv
    A = _matrix(Sinv_Cxy')
    Ω = Syy .- A * S * A'
    EΩ = _scalar_or_eigen(Symmetric((Ω .+ Ω') ./ 2))
    if minimum(EΩ.values; init = zero(eltype(EΩ.values))) < -eltype(EΩ.values)(1e-10) * (1 + maximum(abs, EΩ.values; init = zero(eltype(EΩ.values))))
        @warn "Unscented linearize: residual covariance Ω = Syy - ASAᵀ has negative eigenvalue $(minimum(EΩ.values)) (due to negative centre weight wc[1] at α = $(u.α)); clamping to positive semidefinite. Consider α = 1.0 (cubature weights) if exact PSD residuals are required." maxlog = 1
    end
    Ω_psd = EΩ.vectors * Diagonal(max.(EΩ.values, false)) * EΩ.vectors'
    return (A, ȳ .- A * m, _matrix(Symmetric(Ω_psd)))
end

# the expansion belief on x: an incoming belief's moments, or the factor's x0, or nothing
function _expansion(f::NonlinearFactor, b)
    m, S = _moments(b, f.nin)
    m === nothing || return (m, S)
    f.x0 === nothing && return (nothing, nothing)
    return (f.x0, _zeros_like(f.x0, eltype(f.x0), f.nin, f.nin))
end

# --- Messages -----------------------------------------------------------------------

"""
    Mycelium.factor_message(f::NonlinearFactor, target, polarity, inputs, prior, ps, st)

The factor → variable message, as for [`GaussianFactor`](@ref) but through the local linear model
of [`linearize`](@ref):

- forwards (target ``y``), around the incoming belief ``\\mathcal N(m, S)`` on ``x``:
  ``\\mathcal N(Am + b,\\ ASA^\\top + \\Omega + Q)``; for a point input ``x``, exactly ``\\mathcal N(f(x), Q)``;
- backwards (target ``x``), around the **cavity** `prior` (the other messages into ``x``), or
  `x0`: the likelihood ``\\mathcal N^{-1}(A^\\top R^{-1}(m_y - b),\\ A^\\top R^{-1}A)``,
  ``R = Q + \\Omega + S_y``. With no expansion point it is uninformative.

Backwards, `method.iterations > 1` repeats linearisation at the current posterior,
using the same final site as `likelihood_message` and `invert`.

The prior is used only as the point to linearise at, never multiplied into the returned site,
so nothing is counted twice.
"""
Mycelium.factor_message(f::NonlinearFactor, target::Symbol, polarity, inputs, prior, ps, st) =
    (_nl_iterated_message(f, target, inputs, prior), st)

function _nl_message(f::NonlinearFactor, target::Symbol, inputs, prior)
    if target === f.out_channel
        src = _get(inputs, f.in_channel)
        m, S = _moments(src, f.nin)
        m === nothing && return _nl_no_input_forward(f, src)
        A, b, Ω = linearize(f, m, S)
        Λ = _spd_inv(A * S * A' + Ω + _covariance_at(f.Q, A, b, m, S), "ASAᵀ + Ω + Q")
        return GaussianBelief(Λ * (A * m .+ b), Λ)
    elseif target === f.in_channel
        src = _get(inputs, f.out_channel)
        x̄, P = _expansion(f, prior)
        x̄ === nothing && return LenticulumCore.TrivialBelief()
        if _partly_flat(src)                      # the linear model's relation, flat directions integrated out
            A, b, Ω = linearize(f, x̄, P)
            blocks = NamedTuple{(f.in_channel, f.out_channel)}((-A, _eye(A, eltype(A), f.nout)))
            return first(_linear_message(blocks, b, _covariance_at(f.Q, A, b, src) .+ Ω, target, [(f.out_channel, _split_moments(src, f.nout))], nothing))
        end
        my, Sy = _moments(src, f.nout)
        my === nothing && return LenticulumCore.TrivialBelief()
        return _likelihood(f, my, Sy, x̄, P)
    end
    throw(ArgumentError("channel :$target is not a channel of this NonlinearFactor"))
end

# The forward message when x carries no information (P28; nonlinear.md, "No input on x"): a flat x
# gives no point to linearise at, so the factor's `x0` is used, and the message is `GaussianFactor`'s
# exact one for the linear model there (P3): y flat along range(A), fixed off it up to the noise.
# The unscented transform of a point is the Jacobian (`_linearize` with S = 0), so both methods
# agree. Without x0 nothing is known about where f is evaluated: `TrivialBelief()`, as before.
# A square A of full rank is answered without the flat path; the rank test is the unmemoised one,
# because A is a fresh array on every call and would only fill the memo of `_rank_deficient`.
function _nl_no_input_forward(f::NonlinearFactor, src)
    f.x0 === nothing && return LenticulumCore.TrivialBelief()
    A, b, Ω = linearize(f, f.x0, _zeros_like(f.x0, eltype(f.x0), f.nin, f.nin))
    size(A, 1) == size(A, 2) && !_rank_deficient_svd(A) && return LenticulumCore.TrivialBelief()
    g = GaussianFactor(f.nin, f.nout, f.in_channel, f.out_channel, _covariance_at(f.Q, A, b, src) .+ Ω, nothing, nothing)
    return first(_flat_message(g, f.out_channel, f.in_channel, f.nin, src, (A = A, b = b), nothing))
end

function _likelihood(f, my, Sy, x̄, P)
    A, b, Ω = linearize(f, x̄, P)
    Ri = _spd_inv(_covariance_at(f.Q, A, b, my, Sy) + Ω + Sy, "Q + Ω + Sᵧ")
    return _likelihood_message(A' * (Ri * (my .- b)), A' * Ri * A, _rank_deficient(A))
end

# --- The lens: the posterior, relinearised `iterations` times -------------------------

"""
    NonlinearGaussianModel(factor, polarity)

The open model of a [`NonlinearFactor`](@ref) in the direction the polarity selects. Pure.
"""
struct NonlinearGaussianModel{F<:NonlinearFactor,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::NonlinearGaussianModel) = true
LenticulumCore.latentspace(::NonlinearGaussianModel) = nothing

LenticulumCore.assemble(f::NonlinearFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(NonlinearGaussianModel(f, p), LenticulumCore.SolverInversion(f.method)), st)

"""
    LenticulumCore.likelihood_message(lens, π, inputs, ps, st) -> (site, st)

The final approximate likelihood site; `invert` combines it with `π` to obtain the approximate
posterior ``\\pi \\cdot \\mu_{c\\to\\text{target}}``. Backwards, the likelihood is relinearised at the
current posterior mean `method.iterations` times (with the posterior's
covariance for the unscented transform): Gauss–Newton on the MAP, the iterated EKF / iterated
posterior linearisation. One iteration is the plain EKF / UKF update.
"""
function LenticulumCore.likelihood_message(
    lens::LenticulumCore.BayesianLens{<:NonlinearGaussianModel,<:LenticulumCore.SolverInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    target = only(LenticulumCore.unobserved_channels(lens.model.polarity))
    return (_nl_iterated_message(f, target, inputs, π), st)
end

function _nl_iterated_message(f, target, inputs, π)
    msg = _nl_message(f, target, inputs, π)
    target === f.in_channel || return msg
    _partly_flat(_get(inputs, f.out_channel)) && return msg      # the flat path is not iterated
    my, Sy = _moments(_get(inputs, f.out_channel), f.nout)
    my === nothing && return msg
    for _ in 2:f.method.iterations
        post = Mycelium.combine(π, msg)
        x̄, P = _moments(post, f.nin)
        x̄ === nothing && break
        msg = _likelihood(f, my, Sy, x̄, P)
    end
    return msg
end

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:NonlinearGaussianModel,<:LenticulumCore.SolverInversion},
    π, inputs, ps, st,
)
    msg, st = LenticulumCore.likelihood_message(lens, π, inputs, ps, st)
    return (Mycelium.combine(π, msg), st)
end

# --- The Bethe contribution: the linear model's, around the factor's incoming beliefs ----

Mycelium.uses_cavity(::NonlinearFactor) = true
# its backward message linearises at the target's cavity, so one tree sweep need not be a fixed
# point: such trees are iterated and their residual checked (GH-15)
Mycelium.message_uses_cavity(::NonlinearFactor) = true

"""
    Mycelium.local_free_energy(f::NonlinearFactor, msgs, ps, st)

The factor's Bethe contribution, graded `(fit, complexity, negentropy)`, computed for the local
linear model around the incoming belief on ``x`` (its point, if it is a `DiracBelief`): exact
for affine ``f``, the usual EKF / UKF approximation of the evidence otherwise.
"""
function Mycelium.local_free_energy(f::NonlinearFactor, msgs, ps, st)
    mx = _get(msgs, f.in_channel)
    x̄, P = _expansion(f, mx)
    x̄ === nothing && throw(ArgumentError("local_free_energy: no belief on :$(f.in_channel) to linearise at"))
    A, b, Ω = linearize(f, x̄, P)
    g = GaussianFactor(f.nin, f.nout, f.in_channel, f.out_channel, _covariance_at(f.Q, A, b, msgs) .+ Ω, nothing, nothing)
    return Mycelium.local_free_energy(g, msgs, (A = A, b = b), st)
end

function LenticulumCore.free_energy(f::NonlinearFactor, π, y, ps, st)
    return Mycelium.local_free_energy(f, _game_messages(f, π, y), ps, st)
end

function LenticulumCore.entropy(f::NonlinearFactor, π, y, ps, st)
    F, st = LenticulumCore.free_energy(f, π, y, ps, st)
    return LenticulumCore.GradedEnergy((fit = false, complexity = false, negentropy = _negentropy(F[:negentropy]))), st
end

Adapt.@adapt_structure NonlinearFactor
Adapt.@adapt_structure NonlinearGaussianModel
