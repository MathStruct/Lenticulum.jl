# ---------------------------------------------------------------------------
# RED-Diff (Mardani et al. 2023, arXiv:2305.04391) as a proximal operator.
#
# The variational posterior is q(x₀|y) = N(μ, σ²I) with σ → 0, i.e. a POINT MASS. So the
# inversion returns a `DiracBelief`, and that is not a shortcut — it is the paper's own
# variational family.
#
# The objective (their Eq. 8, specialised to the linear clamp of `ImplicitREDDiff.md`):
#
#     E(x₀, x) = E_{t,ε}[ ω(t)‖ε_θ(α_t x + σ_t ε, t) - ε‖² ] + ½‖P(x₀ - x)‖²
#                └───────────────  the score-matching regulariser  ──────┘   └─ the clamp ─┘
#
# and their Proposition 2, which is the whole reason this is implementable:
#
#     ∇_x reg(x) = E_{t,ε}[ λ_t (ε_θ(x_t, t) - ε) ],        λ_t = λ/SNR_t = λσ_t/α_t
#
# with a STOP-GRADIENT on ε_θ: the denoiser's Jacobian is never formed. Only forward passes
# through the network occur, so this file needs no automatic differentiation — see
# `reddiff.md` §2.
#
# See `reddiff.md` and `RED-Diff as a Statistical Game.md`.
# ---------------------------------------------------------------------------

"""
    REDDiff(; λ = 0.25, steps = 200, lr = 0.1, samples = 1, adam = true,
              β₁ = 0.9, β₂ = 0.999, ϵ = 1e-8, rng = Random.default_rng())

The RED-Diff proximal operator: configuration for the inner optimisation that *is* the
Bayesian inversion.

| field | meaning |
|---|---|
| `λ` | regulariser strength; enters as ``\\lambda_t = \\lambda\\sigma_t/\\alpha_t`` |
| `steps` | inner iterations `L` of Algorithm 1 |
| `lr` | step size |
| `samples` | Monte-Carlo draws of ``(t,\\varepsilon)`` per step (the paper uses 1) |
| `adam` | Adam vs plain gradient descent; the paper uses Adam |

`λ = 0.25` is the value Mardani et al. tuned across their experiments. **It is not a
universal constant**, and for a Gaussian data distribution there is exactly one λ that makes
the implied prior correct — see [`calibrate_lambda`](@ref) and `reddiff.md` §4, which is the
sharpest thing this package has to say about RED-Diff.
"""
struct REDDiff{T<:Real,R<:AbstractRNG}
    λ::T
    steps::Int
    lr::T
    samples::Int
    adam::Bool
    β₁::T
    β₂::T
    ϵ::T
    rng::R
end

function REDDiff(;
    λ = 0.25, steps = 200, lr = 0.1, samples = 1, adam = true,
    β₁ = 0.9, β₂ = 0.999, ϵ = 1e-8, rng = Random.default_rng(),
)
    λ, lr, β₁, β₂, ϵ = promote(float(λ), float(lr), float(β₁), float(β₂), float(ϵ))
    steps > 0 || throw(ArgumentError("steps must be positive; got $steps"))
    samples > 0 || throw(ArgumentError("samples must be positive; got $samples"))
    return REDDiff(λ, steps, lr, samples, adam, β₁, β₂, ϵ, rng)
end

"""
    reddiff_weight(cfg, s, t) -> Real

``\\lambda_t = \\lambda/\\mathrm{SNR}_t = \\lambda\\,\\sigma_t/\\alpha_t``.

The paper motivates this as converting the noise-space objective to signal space. Note it
**grows without bound as ``t \\to 1``**: late, heavily-noised times get the most weight, which
is the opposite of the ELBO weighting and is why RED-Diff is mode-seeking rather than
distribution-matching.
"""
reddiff_weight(cfg::REDDiff, s::AbstractNoiseSchedule, t) = cfg.λ * sigma(s, t) / alpha(s, t)

"""
    regulariser_gradient(pred, x, cfg, ps, st) -> (g, st)

``\\nabla_x \\mathrm{reg}(x) = \\mathbb{E}_{t,\\varepsilon}[\\lambda_t(\\varepsilon_\\theta(x_t,t) - \\varepsilon)]``
— Proposition 2, by Monte Carlo with `cfg.samples` draws.

This is the entire prior contribution: one forward pass of the network per draw, no
Jacobian, no backward pass. The estimator is unbiased in ``(t,\\varepsilon)`` **but the
underlying gradient is not the true gradient of the regulariser** — the denoiser Jacobian has
been dropped. `reddiff.md` §2 is about what that costs.
"""
function regulariser_gradient(pred::NoisePredictor, x, cfg::REDDiff, ps, st)
    s = pred.schedule
    g = zero(x)
    for _ in 1:(cfg.samples)
        t = sample_time(cfg.rng, s)
        ε = randn(cfg.rng, eltype(x), size(x))
        x_t = perturb(s, x, t, ε)
        ε̂, st = epsilon(pred, x_t, t, ps, st)
        g = g .+ reddiff_weight(cfg, s, t) .* (ε̂ .- ε)
    end
    return (g ./ cfg.samples, st)
end

"""
    reddiff_solve(pred, cfg, x_init, datagrad, hard, ps, st) -> (x, st)

Algorithm 1: `steps` iterations of

```math
x \\leftarrow \\mathrm{Optim}\\bigl(x,\\ \\underbrace{\\nabla_x \\tfrac12\\|P(x_0-x)\\|^2}_{\\texttt{datagrad}}
 + \\underbrace{\\mathbb{E}[\\lambda_t(\\varepsilon_\\theta - \\varepsilon)]}_{\\text{Prop. 2}}\\bigr)
```

`datagrad(x) -> g` supplies the clamp's gradient; keeping it a closure is what makes this
operator reusable for a general forward model ``f`` rather than only the diagonal ``P``.

`hard` is a `(mask, values)` pair applied by **projection** after every step: coordinates
with ``\\rho_{in} = \\infty`` are overwritten rather than penalised, because an infinite
precision is not a number you can put in a gradient. That projection is the categorical
*cup* of `Copiers Cups and Caps.md`, and it is the one place the ``\\rho \\to \\infty`` limit
is taken literally rather than numerically.
"""
function reddiff_solve(pred::NoisePredictor, cfg::REDDiff, x_init, datagrad, hard, ps, st)
    x = copy(x_init)
    _project!(x, hard)
    m = zero(x)
    v = zero(x)
    for k in 1:(cfg.steps)
        gr, st = regulariser_gradient(pred, x, cfg, ps, st)
        g = datagrad(x) .+ gr
        if cfg.adam
            m = cfg.β₁ .* m .+ (1 - cfg.β₁) .* g
            v = cfg.β₂ .* v .+ (1 - cfg.β₂) .* abs2.(g)
            m̂ = m ./ (1 - cfg.β₁^k)
            v̂ = v ./ (1 - cfg.β₂^k)
            x = x .- cfg.lr .* m̂ ./ (sqrt.(v̂) .+ cfg.ϵ)
        else
            x = x .- cfg.lr .* g
        end
        _project!(x, hard)
    end
    return (x, st)
end

_project!(x, ::Nothing) = x
function _project!(x, hard::Tuple)
    mask, vals = hard
    @inbounds for i in eachindex(x)
        mask[i] && (x[i] = vals[i])
    end
    return x
end

# --- The Gaussian calibration ---------------------------------------------

"""
    calibrate_lambda(s::AbstractNoiseSchedule, v₀ = 1.0; nodes = 2000) -> Real

The unique `λ` for which RED-Diff's regulariser is the **exact** negative log-prior gradient
of a Gaussian data distribution ``\\mathcal{N}(0, v_0 I)``.

For that data distribution ``\\varepsilon_\\theta`` is available in closed form, and the
regulariser gradient collapses to a linear shrinkage:

```math
\\nabla_x\\mathrm{reg}(x) \\;=\\; \\kappa\\,x,
\\qquad
\\kappa \\;=\\; \\lambda\\int_{t_{\\min}}^{1}\\frac{\\sigma_t^2}{\\alpha_t^2 v_0 + \\sigma_t^2}\\,dt
```

whereas the true prior gradient is ``x/v_0``. Setting ``\\kappa = 1/v_0`` gives the returned
value. Computed by the trapezoidal rule; the integrand is smooth on ``[t_{\\min},1]``.

> [!important] Why this matters beyond the test suite
> `λ` is presented in the paper as a tuned hyperparameter, and here it is *derived*. That
> tells you what tuning `λ` is really doing: **choosing how strong the learned prior is**,
> with a wrong value biasing every posterior by a known factor. See `reddiff.md` §4.
"""
function calibrate_lambda(s::AbstractNoiseSchedule, v₀ = 1.0; nodes::Int = 2000)
    a, b = _tmin(s), one(float(v₀))
    h = (b - a) / nodes
    f(t) = sigma(s, t)^2 / marginal_variance(s, t, v₀)
    acc = (f(a) + f(b)) / 2
    for i in 1:(nodes - 1)
        acc += f(a + i * h)
    end
    return 1 / (v₀ * acc * h)
end

_tmin(s::VPSDE) = s.tmin
_tmin(::AbstractNoiseSchedule) = 0.0
