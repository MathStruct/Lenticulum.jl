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
    Unscented(; α = 1e-3, β = 2.0, κ = 0.0, iterations = 1)

Approximate a nonlinear factor's messages by the unscented transform: ``2n+1`` sigma points of
the expansion belief, pushed through ``f``, fitted by a linear model with a residual covariance
(statistical linear regression). Exact for linear ``f``, and second-order accurate in the mean
for smooth ``f``; it needs a covariance at the expansion point (from the incoming belief), and
falls back to linearisation at a point. `α, β, κ` are the usual scaling parameters (Wan & van
der Merwe 2000); `iterations` as for [`Linearization`](@ref) (posterior linearisation).
"""
Base.@kwdef struct Unscented
    α::Float64 = 1e-3
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
- `x0` is the expansion point used when a backward message is asked for before anything is
  known about `x` (no cavity belief). Without it such a message is uninformative.

Both polarities are supported, like `GaussianFactor`. It has no parameters (`f` is fixed).
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
    isposdef(Symmetric(Q)) || throw(ArgumentError("noise covariance Q must be positive definite"))
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
function LenticulumCore.supported_polarities(f::NonlinearFactor)
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    ks = (f.in_channel, f.out_channel)
    return (LenticulumCore.Polarity(NamedTuple{ks}((O, U))),
            LenticulumCore.Polarity(NamedTuple{ks}((U, O))))
end
function LenticulumCore.supports_polarity(f::NonlinearFactor, p::LenticulumCore.Polarity)
    keys(p) == (f.in_channel, f.out_channel) || return false
    pi_, po = p[f.in_channel], p[f.out_channel]
    return (pi_ isa LenticulumCore.Observed && po isa LenticulumCore.Unobserved) ||
           (pi_ isa LenticulumCore.Unobserved && po isa LenticulumCore.Observed)
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
    LenticulumCore.SquaredNorm(inv(Symmetric(f.Q)); check = false)

# --- The local linear model --------------------------------------------------------

function _jacobian(f::NonlinearFactor, x)
    f.jacobian === nothing || return Matrix{Float64}(f.jacobian(x))
    J = zeros(f.nout, f.nin)
    for i in 1:f.nin
        h = 1e-6 * (1 + abs(x[i]))
        e = zeros(f.nin); e[i] = h
        J[:, i] = (_vec(f.f(x .+ e)) .- _vec(f.f(x .- e))) ./ (2h)
    end
    return J
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
    return (A, _vec(f.f(m)) .- A * m, zeros(f.nout, f.nout))
end

function _linearize(u::Unscented, f, m, S)
    all(iszero, S) && return _linearize(Linearization(), f, m, S)
    n = length(m)
    λ = u.α^2 * (n + u.κ) - n
    E = LinearAlgebra.eigen(Symmetric(Matrix{Float64}(S)))
    vals = max.(E.values, 0.0)
    L = E.vectors * Diagonal(sqrt.(vals))                          # S = L Lᵀ, also when singular
    c = sqrt(n + λ)
    X = hcat(m, m .+ c .* L, m .- c .* L)                          # 2n + 1 sigma points
    wm = vcat(λ / (n + λ), fill(1 / (2(n + λ)), 2n))
    wc = copy(wm); wc[1] += 1 - u.α^2 + u.β
    Y = reduce(hcat, (_vec(f.f(x)) for x in eachcol(X)))
    ȳ = Y * wm
    Dx, Dy = X .- m, Y .- ȳ
    Cxy = (Dx .* wc') * Dy'                                        # n × nout
    Syy = (Dy .* wc') * Dy'
    tol = n * eps(Float64) * maximum(vals; init = 0.0)
    inv_vals = [v > tol ? 1.0 / v : 0.0 for v in vals]
    Sinv_Cxy = E.vectors * (inv_vals .* (E.vectors' * Cxy))        # reuse E instead of rank + pinv
    A = Matrix(Sinv_Cxy')
    Ω = Syy .- A * S * A'
    EΩ = LinearAlgebra.eigen(Symmetric((Ω .+ Ω') ./ 2))
    if minimum(EΩ.values; init = 0.0) < -1e-10 * (1 + maximum(abs, EΩ.values; init = 0.0))
        @warn "Unscented linearize: residual covariance Ω = Syy - ASAᵀ has negative eigenvalue $(minimum(EΩ.values)) (due to negative centre weight wc[1] at α = $(u.α)); clamping to positive semidefinite. Consider α = 1.0 (cubature weights) if exact PSD residuals are required." maxlog = 1
    end
    Ω_psd = EΩ.vectors * Diagonal(max.(EΩ.values, 0.0)) * EΩ.vectors'
    return (A, ȳ .- A * m, Matrix(Symmetric(Ω_psd)))
end

# the expansion belief on x: an incoming belief's moments, or the factor's x0, or nothing
function _expansion(f::NonlinearFactor, b)
    m, S = _moments(b, f.nin)
    m === nothing || return (m, S)
    f.x0 === nothing && return (nothing, nothing)
    return (f.x0, zeros(f.nin, f.nin))
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
        m, S = _moments(_get(inputs, f.in_channel), f.nin)
        m === nothing && return LenticulumCore.TrivialBelief()
        A, b, Ω = linearize(f, m, S)
        Λ = inv(_chol(A * S * A' + Ω + f.Q, "ASAᵀ + Ω + Q"))
        return GaussianBelief(Λ * (A * m .+ b), Λ)
    elseif target === f.in_channel
        my, Sy = _moments(_get(inputs, f.out_channel), f.nout)
        my === nothing && return LenticulumCore.TrivialBelief()
        x̄, P = _expansion(f, prior)
        x̄ === nothing && return LenticulumCore.TrivialBelief()
        return _likelihood(f, my, Sy, x̄, P)
    end
    throw(ArgumentError("channel :$target is not a channel of this NonlinearFactor"))
end

function _likelihood(f, my, Sy, x̄, P)
    A, b, Ω = linearize(f, x̄, P)
    Ri = inv(_chol(f.Q + Ω + Sy, "Q + Ω + Sᵧ"))
    return GaussianBelief(A' * (Ri * (my .- b)), A' * Ri * A)
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
    g = GaussianFactor(f.nin, f.nout, f.in_channel, f.out_channel, f.Q .+ Ω, nothing, nothing)
    return Mycelium.local_free_energy(g, msgs, (A = A, b = b), st)
end

function LenticulumCore.free_energy(f::NonlinearFactor, π, y, ps, st)
    return Mycelium.local_free_energy(f, _game_messages(f, π, y), ps, st)
end

function LenticulumCore.entropy(f::NonlinearFactor, π, y, ps, st)
    F, st = LenticulumCore.free_energy(f, π, y, ps, st)
    return LenticulumCore.GradedEnergy((fit = 0.0, complexity = 0.0, negentropy = -F[:negentropy])), st
end
