# ---------------------------------------------------------------------------
# The VP-SDE of Song et al. 2021 (arXiv:2011.13456), §3.4 / Eq. 32-33.
#
#     forward SDE:          dx = -½β(t)x dt + √β(t) dw,        t ∈ [0,1]
#     β(t)                = β_min + t(β_max - β_min)
#     perturbation kernel:  p_{0t}(x_t | x_0) = N(x_t; α_t x_0, σ_t² I)
#     α_t                 = exp(-½B(t)),   B(t) = ∫₀ᵗ β(s)ds
#     σ_t²                = 1 - exp(-B(t)) = 1 - α_t²
#
# "Variance preserving" is that last identity: α_t² + σ_t² = 1, so a unit-variance data
# distribution keeps unit variance for every t. It is checked in the test suite rather than
# asserted here, because it is the one property the whole file exists to have.
#
# See `schedule.md`.
# ---------------------------------------------------------------------------

"""
    abstract type AbstractNoiseSchedule

A forward corruption process, presented through its **perturbation kernel** rather than its
SDE: everything downstream needs only `alpha(s, t)` and `sigma(s, t)`.

That is a deliberate narrowing. The SDE is the definition; the Gaussian marginal is what
RED-Diff and score matching actually touch, and a schedule whose marginal is not Gaussian
would not fit this interface. [`drift`](@ref) and [`diffusion`](@ref) are provided for
completeness and are not used by the inversion.
"""
abstract type AbstractNoiseSchedule end

"""
    VPSDE(; βmin = 0.1, βmax = 20.0, tmin = 1e-3)

The variance-preserving SDE. Defaults are Song et al.'s, which are in turn DDPM's
discretisation in the continuous limit.

`tmin` is a **sampling floor**, not part of the mathematics: at `t = 0` we have `σ_t = 0`,
the score `-ε_θ/σ_t` is a division by zero, and the RED-Diff weight `λ_t = λσ_t/α_t`
degenerates. Every published implementation carries such a floor and most do not say so.
Times are drawn from `[tmin, 1]`; see [`sample_time`](@ref).
"""
struct VPSDE{T<:Real} <: AbstractNoiseSchedule
    βmin::T
    βmax::T
    tmin::T
end

function VPSDE(; βmin = 0.1, βmax = 20.0, tmin = 1e-3)
    βmin, βmax, tmin = promote(float(βmin), float(βmax), float(tmin))
    0 < βmin <= βmax || throw(ArgumentError("need 0 < βmin ≤ βmax; got $βmin, $βmax"))
    0 <= tmin < 1 || throw(ArgumentError("need 0 ≤ tmin < 1; got $tmin"))
    return VPSDE(βmin, βmax, tmin)
end

"""
    beta(s::VPSDE, t) -> Real

``\\beta(t) = \\beta_{\\min} + t(\\beta_{\\max} - \\beta_{\\min})``, the SDE's noise rate.
"""
beta(s::VPSDE, t) = s.βmin + t * (s.βmax - s.βmin)

"""
    integrated_beta(s::VPSDE, t) -> Real

``B(t) = \\int_0^t \\beta(u)\\,du = \\beta_{\\min}t + \\tfrac12(\\beta_{\\max}-\\beta_{\\min})t^2``.

In closed form, which is the only reason the VP-SDE is convenient: a schedule needing
numerical quadrature here would put a solve inside every message.
"""
integrated_beta(s::VPSDE, t) = s.βmin * t + (s.βmax - s.βmin) * t^2 / 2

"""
    alpha(s, t) -> Real

The signal coefficient ``\\alpha_t = \\exp(-\\tfrac12 B(t))`` of the perturbation kernel.
"""
alpha(s::VPSDE, t) = exp(-integrated_beta(s, t) / 2)

"""
    sigma(s, t) -> Real

The noise coefficient ``\\sigma_t = \\sqrt{1 - \\exp(-B(t))} = \\sqrt{1 - \\alpha_t^2}``.

Computed from `expm1` rather than as `sqrt(1 - alpha^2)`: near `t = 0` the latter is
`sqrt(1 - (1-ε)²)` and loses half its significant digits to cancellation.
"""
sigma(s::VPSDE, t) = sqrt(-expm1(-integrated_beta(s, t)))

"""
    snr(s, t) -> Real

``\\mathrm{SNR}_t = \\alpha_t/\\sigma_t``. RED-Diff's weighting is
``\\lambda_t = \\lambda/\\mathrm{SNR}_t``; see [`REDDiff`](@ref).
"""
snr(s::AbstractNoiseSchedule, t) = alpha(s, t) / sigma(s, t)

"""
    drift(s::VPSDE, x, t)
    diffusion(s::VPSDE, t)

The SDE coefficients ``f(x,t) = -\\tfrac12\\beta(t)x`` and ``g(t) = \\sqrt{\\beta(t)}``.

Provided for completeness and to document what the schedule *is*. Nothing in this package
integrates the SDE — RED-Diff replaces sampling with optimisation, which is the whole point
(`RED-Diff as a Statistical Game.md` §2).
"""
drift(s::VPSDE, x, t) = (-beta(s, t) / 2) .* x
diffusion(s::VPSDE, t) = sqrt(beta(s, t))

"""
    perturb(s, x₀, t, ε) -> x_t

``x_t = \\alpha_t x_0 + \\sigma_t \\varepsilon``: one draw from the perturbation kernel,
written as a **reparametrisation** so it is differentiable in `x₀`.

That differentiability is exactly what RED-Diff keeps and what its stop-gradient does *not*
discard — see `reddiff.md` §2.
"""
perturb(s::AbstractNoiseSchedule, x₀, t, ε) = alpha(s, t) .* x₀ .+ sigma(s, t) .* ε

"""
    sample_time(rng, s) -> Real

A time drawn uniformly from `[tmin, 1]`, the outer expectation of the score-matching
regulariser.
"""
sample_time(rng::AbstractRNG, s::VPSDE) = s.tmin + (1 - s.tmin) * rand(rng)

"""
    marginal_variance(s, t, v₀) -> Real

The variance of ``x_t`` when ``x_0`` has variance `v₀`: ``\\alpha_t^2 v_0 + \\sigma_t^2``.

For `v₀ = 1` this is identically `1` — the variance-preserving property. It is used by the
analytic Gaussian oracle in the test suite, which is the only closed-form ``\\varepsilon_\\theta``
available and therefore the only thing the inversion can be checked against.
"""
marginal_variance(s::AbstractNoiseSchedule, t, v₀) = alpha(s, t)^2 * v₀ + sigma(s, t)^2
