# ---------------------------------------------------------------------------
# Conditional samples from a learned relation: the reverse diffusion with the clamped
# coordinates replaced at every step.
#
# Root finding returns modes of the smoothed density; a query "draw a 3" wants samples of
# p(z_F | z_C). This file runs the predictor's own reverse process (DDPM's ancestral step, exact
# for every Gaussian step of the VP-SDE) on all coordinates, and after every step overwrites the
# clamped ones by a fresh forward sample of the clamp at the new time (the replacement method).
# Replacement ignores how the clamp should bend the free coordinates' history; RePaint's
# resampling (`resample`) re-noises and repeats each step to let them catch up.
#
# Uses the predictor only, not the field's nodes. See `sampler.md`.
# ---------------------------------------------------------------------------

"""
    implicit_sample(m::ImplicitDiffusion, z₀, ρ, ps, st; n = 1, steps = 100, resample = 0,
                    T = 1.0, rng = Random.default_rng()) -> (Z, st)

`n` samples of the free coordinates given the hard-clamped ones, as the columns of `Z`
(``\\dim z\\times n``): the conditional ``p(z_F \\mid z_C = z_{0,C})`` of the model's data
distribution, where the root finders return its modes. `ρ` marks the polarity as everywhere
else: `Inf` is clamped to `z₀`, `0` is free. Soft evidence (finite, positive `ρ`) is not
supported: it needs likelihood guidance, which is approximate for a diffusion model.

The sampler is the reverse VP-SDE discretised by DDPM's ancestral step on a grid from `T` to
the schedule's `tmin`, uniform in log-SNR (`steps` steps), using the predictor of `m` only (not
its nodes). After every step the clamped coordinates are replaced by ``\\alpha_s z_{0,C} + \\sigma_s\\xi``, the
clamp's forward marginal at the new time, and the last step returns Tweedie's denoised estimate
with the clamps set exactly. With `resample = r` each step is repeated `r` more times after
re-noising back to the previous time (RePaint's harmonisation), which corrects the free
coordinates' lag behind the clamp at `r + 1` times the cost. On the circle a few resamplings
are needed: without them the conditional samples are biased towards the centre (`sampler.md` §3).

With `m.batched` all `n` chains go through the network in one call per step; otherwise one call
per chain.
"""
function implicit_sample(m::ImplicitDiffusion, z₀, ρ, ps, st; n::Integer = 1, steps::Integer = 100,
                         resample::Integer = 0, T = 1.0, rng::AbstractRNG = Random.default_rng())
    all(r -> r == 0 || isinf(r), ρ) || throw(ArgumentError(
        "implicit_sample takes hard clamps (ρ = Inf) and free coordinates (ρ = 0); soft evidence needs guidance"))
    s = m.predictor.schedule
    C = isinf.(ρ)
    d = length(z₀)
    ts = _logsnr_grid(s, float(T), s.tmin, steps)
    X = randn(rng, d, n)
    _replace!(X, C, z₀, s, ts[1], rng)
    for k in 1:steps
        t, u = ts[k], ts[k + 1]
        for r in 0:resample
            X, st = _ancestral(m, X, t, u, ps, st, rng)
            _replace!(X, C, z₀, s, u, rng)
            r < resample && (X = _renoise(s, X, u, t, rng))
        end
    end
    X̂, st = _denoise_columns(m, X, last(ts), ps, st)
    X̂[C, :] .= z₀[C]
    return (X̂, st)
end

# times from T down to tmin, uniform in log-SNR log(α/σ): the steps crowd at low noise, where the
# density has its fine structure (a uniform grid in t gives too narrow samples there)
function _logsnr_grid(s, T, tmin, steps)
    ℓ(t) = log(alpha(s, t) / sigma(s, t))
    function time_of(l)                               # ℓ decreases in t; bisection
        lo, hi = tmin, T
        for _ in 1:100
            mid = (lo + hi) / 2
            ℓ(mid) > l ? (lo = mid) : (hi = mid)
        end
        return (lo + hi) / 2
    end
    ls = range(ℓ(T), ℓ(tmin); length = steps + 1)
    return [T; [time_of(l) for l in ls[2:(end - 1)]]; tmin]
end

# the clamp's forward marginal at time t
function _replace!(X, C, z₀, s, t, rng)
    any(C) || return X
    X[C, :] .= alpha(s, t) .* z₀[C] .+ sigma(s, t) .* randn(rng, count(C), size(X, 2))
    return X
end

# ε̂ for every column: one call when batched, one per column otherwise
function _eps_columns(m::ImplicitDiffusion, X, t, ps, st)
    if m.batched
        E, st = epsilon(m.predictor, X, t, ps, st)
        return (reshape(E, size(X)), st)
    end
    E = similar(X)
    for b in axes(X, 2)
        e, st = epsilon(m.predictor, X[:, b], t, ps, st)
        E[:, b] .= vec(e)
    end
    return (E, st)
end

function _denoise_columns(m, X, t, ps, st)
    s = m.predictor.schedule
    E, st = _eps_columns(m, X, t, ps, st)
    return ((X .- sigma(s, t) .* E) ./ alpha(s, t), st)
end

# DDPM's ancestral step t → u < t: the Gaussian posterior q(x_u | x_t, x̂₀) with Tweedie's x̂₀
function _ancestral(m, X, t, u, ps, st, rng)
    s = m.predictor.schedule
    X̂, st = _denoise_columns(m, X, t, ps, st)
    a = alpha(s, t) / alpha(s, u)                    # x_t = a x_u + σ_{t|u} ξ
    σt², σu² = sigma(s, t)^2, sigma(s, u)^2
    σtu² = σt² - a^2 * σu²
    mean = (a * σu² / σt²) .* X .+ (alpha(s, u) * σtu² / σt²) .* X̂
    return (mean .+ sqrt(σtu² * σu² / σt²) .* randn(rng, size(X)...), st)
end

# forward from u back to t > u: x_t = a x_u + σ_{t|u} ξ
function _renoise(s, X, u, t, rng)
    a = alpha(s, t) / alpha(s, u)
    return a .* X .+ sqrt(sigma(s, t)^2 - a^2 * sigma(s, u)^2) .* randn(rng, size(X)...)
end
