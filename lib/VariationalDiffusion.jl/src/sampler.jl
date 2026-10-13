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
# Optional corrector steps (Langevin or MALA) at every noise level move the chains towards the
# level's own marginal; a product of relations needs them, because the sum of the factors'
# scores is not the score of the noised product (T9). Uses the predictors only, not the field's
# nodes. See `sampler.md` and `product.md` §5.
# ---------------------------------------------------------------------------

"""
    implicit_sample(m, z₀, ρ, ps, st; n = 1, steps = 100, resample = 0, corrector = :none,
                    corrector_steps = 1, corrector_step = 0.1, T = 1.0,
                    rng = Random.default_rng()) -> (Z, st)

`n` samples of the free coordinates given the hard-clamped ones, as the columns of `Z`
(``\\dim z\\times n``): the conditional ``p(z_F \\mid z_C = z_{0,C})`` of the model's data
distribution, where the root finders return its modes. `ρ` marks the polarity as everywhere
else: `Inf` is clamped to `z₀`, `0` is free. Soft evidence (finite, positive `ρ`) is not
supported: it needs likelihood guidance, which is approximate for a diffusion model. `m` is an
[`ImplicitDiffusion`](@ref) or a [`ProductRelation`](@ref) whose relations share one noise schedule.

The sampler is the reverse VP-SDE discretised by DDPM's ancestral step on a grid from `T` to
the schedule's `tmin`, uniform in log-SNR (`steps` steps), using the predictors of `m` only (not
their nodes). After every step the clamped coordinates are replaced by ``\\alpha_s z_{0,C} + \\sigma_s\\xi``, the
clamp's forward marginal at the new time, and the last step returns Tweedie's denoised estimate
with the clamps set exactly. With `resample = r` each step is repeated `r` more times after
re-noising back to the previous time (RePaint's harmonisation), which corrects the free
coordinates' lag behind the clamp at `r + 1` times the cost. On the circle a few resamplings
are needed: without them the conditional samples are biased towards the centre (`sampler.md` §3).

**Corrector steps.** With `corrector = :langevin` or `:mala`, every step is followed by
`corrector_steps` Markov chain steps on the free coordinates that target the model's density at
the new noise level (for a product, ``\\prod_i p_{i,u}^{\\beta_i}``): a Langevin step
``x \\leftarrow x + \\eta\\,s(x) + \\sqrt{2\\eta}\\,\\xi`` with ``\\eta = `` `corrector_step` ``\\cdot\\,\\sigma_u^2``, and
with `:mala` a Metropolis–Hastings test that makes the step exact for that density. `:mala`
needs log-densities: closed-form mixtures and energy networks, or products of them. For a
single model the corrector refines a sampler that is already consistent; for a product it is
what makes the samples right, because the summed scores of the factors are not the scores of
the noised product (Du et al. 2023; `product.md` §5).

**Reweighting.** Corrector steps fix the shape of each mode but cannot move mass between modes
that a low noise level separates. With `smc = true` the chains are also weighted, as a sequential
Monte Carlo sampler for the sequence of densities ``\\pi_u \\propto \\prod_i p_{i,u}^{\\beta_i}`` (for a single
model, ``p_u``): each ancestral step is a proposal, and the chain's weight is multiplied by
``\\pi_u(x_u)\\,q(x_t \\mid x_u) / (\\pi_t(x_t)\\,r(x_u \\mid x_t))``, with ``q`` the forward noising and ``r`` the ancestral
step; the chains are resampled (systematically) when the effective sample size falls below
`n/2`, and once more at the end, so `Z` holds unweighted draws. This is consistent as `n` grows,
whatever the mismatch between summed scores and the noised product (Skreta et al. 2025's
Feynman–Kac correctors are this construction in continuous time). It needs log-densities, like
`:mala`. With clamps, the clamped coordinates follow ``\\alpha_u z_{0,C}`` without noise, so that the
targets are the conditionals ``\\pi_u(x_F \\mid x_C = \\alpha_u z_{0,C})``. The moves between reweightings must
leave each level's density invariant, so `smc` takes `corrector = :none` or `:mala`, and no `resample`.

**Held clamps and guidance.** For a predictor with noise blocks, `clamp_level = tmin` (or another
level) holds every fully clamped block at that level instead of replacing it: the network sees
the clamp clean, at every step, as in training with independent block times (UniDiffuser), and
the reverse diffusion of the free coordinates follows the conditional score
``\\nabla\\log p_t(x_F \\mid z_C)``. Blocks that are only partly clamped are still replaced. With
`guidance = w > 0` the noise prediction is
``(1 + w)\\,\\hat\\varepsilon(x, z_C) - w\\,\\hat\\varepsilon(x, \\varnothing)``, the unconditional prediction taken
with the held blocks at level 1, where they are noise (classifier-free guidance, Ho & Salimans
2022): it sharpens the conditional towards the clamp. Not combined with correctors or `smc`.

With `m.batched` all `n` chains go through the network in one call per step; otherwise one call
per chain.
"""
function implicit_sample(m::AbstractImplicitRelation, z₀, ρ, ps, st; n::Integer = 1, steps::Integer = 100,
                         resample::Integer = 0, corrector::Symbol = :none, corrector_steps::Integer = 1,
                         corrector_step::Real = 0.1, smc::Bool = false, T = 1.0, clamp_level = nothing,
                         guidance::Real = 0.0, rng::AbstractRNG = Random.default_rng())
    all(r -> r == 0 || isinf(r), ρ) || throw(ArgumentError(
        "implicit_sample takes hard clamps (ρ = Inf) and free coordinates (ρ = 0); soft evidence needs guidance"))
    if clamp_level !== nothing
        (m isa ImplicitDiffusion && m.predictor.blocks !== nothing) || throw(ArgumentError(
            "clamp_level needs an ImplicitDiffusion whose predictor has noise blocks"))
        (corrector === :none && !smc) || throw(ArgumentError("clamp_level does not combine with correctors or smc"))
        return _sample_held(m, z₀, ρ, ps, st; n, steps, resample, T, level = float(clamp_level), w = float(guidance), rng)
    end
    guidance == 0 || throw(ArgumentError("guidance needs clamp_level (and noise blocks)"))
    corrector in (:none, :langevin, :mala) ||
        throw(ArgumentError("corrector must be :none, :langevin or :mala, got :$corrector"))
    (corrector === :mala || smc) && !_has_energy(m) && throw(ArgumentError(
        "corrector = :mala and smc = true need log-densities: a closed-form mixture, an energy network, or a product of them"))
    smc && resample > 0 && throw(ArgumentError("smc = true does not combine with resample"))
    smc && corrector === :langevin && throw(ArgumentError(
        "smc = true needs moves that leave each level's density invariant: corrector = :mala or :none"))
    s = _schedule(m)
    C = isinf.(ρ)
    F = .!C
    d = length(z₀)
    ts = _logsnr_grid(s, float(T), s.tmin, steps)
    X = randn(rng, d, n)
    if smc
        X[C, :] .= alpha(s, ts[1]) .* z₀[C]
        L, st = _logdensity_columns(m, X, ts[1], ps, st)
        logw = L .+ vec(sum(abs2, X[F, :]; dims = 1)) ./ 2      # π_T against the start N(0, I)
    else
        _replace!(X, C, z₀, s, ts[1], rng)
    end
    for k in 1:steps
        t, u = ts[k], ts[k + 1]
        if smc
            M, sd, st = _ancestral_parts(m, X, t, u, ps, st)
            Y = copy(X)
            Y[F, :] .= M[F, :] .+ sd .* randn(rng, count(F), n)
            Y[C, :] .= alpha(s, u) .* z₀[C]
            Lu, st = _logdensity_columns(m, Y, u, ps, st)
            a = alpha(s, t) / alpha(s, u)
            v = sigma(s, t)^2 - a^2 * sigma(s, u)^2                 # q(x_t | x_u) = N(a x_u, v I)
            logw .+= Lu .- L .- vec(sum(abs2, X[F, :] .- a .* Y[F, :]; dims = 1)) ./ (2v) .+
                     vec(sum(abs2, Y[F, :] .- M[F, :]; dims = 1)) ./ (2sd^2)
            X, L = Y, Lu
            if _ess(logw) < n / 2
                idx = _systematic(rng, logw)
                X, L = X[:, idx], L[idx]
                logw .= 0
            end
        else
            for r in 0:resample
                X, st = _ancestral(m, X, t, u, ps, st, rng)
                _replace!(X, C, z₀, s, u, rng)
                r < resample && (X = _renoise(s, X, u, t, rng))
            end
        end
        if corrector !== :none
            X, _, L, st = _correct(m, X, F, u, ps, st, rng; mala = corrector === :mala, steps = corrector_steps,
                                   η = corrector_step * sigma(s, u)^2, L = smc ? L : nothing)
        end
    end
    smc && (X = X[:, _systematic(rng, logw)])
    X̂, st = _denoise_columns(m, X, last(ts), ps, st)
    X̂[C, :] .= z₀[C]
    return (X̂, st)
end

_schedule(m::ImplicitDiffusion) = m.predictor.schedule

# The reverse diffusion with the fully clamped blocks held at `level` (and, for guidance, at 1).
function _sample_held(m::ImplicitDiffusion, z₀, ρ, ps, st; n, steps, resample, T, level, w, rng)
    s = _schedule(m)
    blocks = m.predictor.blocks
    C = isinf.(ρ)
    held = [all(C[b]) for b in blocks.blocks]
    any(held) || throw(ArgumentError("clamp_level: no block is fully clamped"))
    H = held[blocks.owner]                            # coordinates of held blocks
    R = C .& .!H                                      # clamped coordinates of annealed blocks: replaced
    d = length(z₀)
    τ(t, l) = [h ? l : t for h in held]
    function hold!(X, l)
        X[H, :] .= alpha(s, l) .* z₀[H] .+ sigma(s, l) .* randn(rng, count(H), size(X, 2))
        return X
    end
    function guided(X, t)
        E, st = _eps_columns(m, hold!(X, level), τ(t, level), ps, st)
        w == 0 && return E
        Eu, st = _eps_columns(m, hold!(copy(X), 1.0), τ(t, 1.0), ps, st)
        return (1 + w) .* E .- w .* Eu
    end
    ts = _logsnr_grid(s, float(T), s.tmin, steps)
    X = randn(rng, d, n)
    _replace!(X, R, z₀, s, ts[1], rng)
    for k in 1:steps
        t, u = ts[k], ts[k + 1]
        for r in 0:resample
            E = guided(X, t)
            X̂ = (X .- sigma(s, t) .* E) ./ alpha(s, t)
            a = alpha(s, t) / alpha(s, u)
            σt², σu² = sigma(s, t)^2, sigma(s, u)^2
            σtu² = σt² - a^2 * σu²
            X = (a * σu² / σt²) .* X .+ (alpha(s, u) * σtu² / σt²) .* X̂ .+ sqrt(σtu² * σu² / σt²) .* randn(rng, d, n)
            _replace!(X, R, z₀, s, u, rng)
            r < resample && (X = _renoise(s, X, u, t, rng))
        end
    end
    tl = last(ts)
    X̂ = (X .- sigma(s, tl) .* guided(X, tl)) ./ alpha(s, tl)
    X̂[C, :] .= z₀[C]
    return (X̂, st)
end

# the effective sample size of log-weights, and systematic resampling by them
function _ess(logw)
    w = exp.(logw .- maximum(logw))
    return sum(w)^2 / sum(abs2, w)
end
function _systematic(rng, logw)
    w = exp.(logw .- maximum(logw)); c = cumsum(w ./ sum(w))
    n = length(w); u0 = rand(rng) / n
    return [min(searchsortedfirst(c, u0 + (k - 1) / n), n) for k in 1:n]
end

# `steps` Langevin steps on the free rows F of X at level u, with a Metropolis–Hastings test when
# `mala`; returns the chains, the acceptance rate, their log-densities (when `L`, the incoming
# ones, is given or `mala`) and the state
function _correct(m, X, F, u, ps, st, rng; mala::Bool, steps::Integer, η::Real, L = nothing)
    σ = sigma(_schedule(m), u)
    E, st = _eps_columns(m, X, u, ps, st)
    S = -E ./ σ
    track = L !== nothing
    mala && L === nothing && ((L, st) = _logdensity_columns(m, X, u, ps, st))
    accepted = 0
    for _ in 1:steps
        M = X[F, :] .+ η .* S[F, :]
        Y = copy(X)
        Y[F, :] .= M .+ sqrt(2η) .* randn(rng, count(F), size(X, 2))
        Ey, st = _eps_columns(m, Y, u, ps, st)
        Sy = -Ey ./ σ
        if !mala
            X, S = Y, Sy
            accepted += size(X, 2)
            continue
        end
        Ly, st = _logdensity_columns(m, Y, u, ps, st)
        fwd = vec(sum(abs2, Y[F, :] .- M; dims = 1))                       # log q(y | x) = -fwd/(4η)
        bwd = vec(sum(abs2, X[F, :] .- Y[F, :] .- η .* Sy[F, :]; dims = 1))   # log q(x | y) = -bwd/(4η)
        acc = log.(rand(rng, size(X, 2))) .< Ly .- L .+ (fwd .- bwd) ./ (4η)
        X[:, acc] .= Y[:, acc]; S[:, acc] .= Sy[:, acc]; L[acc] .= Ly[acc]
        accepted += count(acc)
    end
    track && !mala && ((L, st) = _logdensity_columns(m, X, u, ps, st))
    return (X, accepted / (steps * size(X, 2)), L, st)
end

# log p_u of every column, up to a constant per level: closed form for a mixture, -E for an
# energy network (its energy is -log p_u up to a constant in x)
_logdensity_columns(m::ImplicitDiffusion{<:NoisePredictor{<:GaussianMixtureEps}}, X, t, ps, st) =
    ([mixture_logdensity(m.predictor.model, x, t, ps) for x in eachcol(X)], st)
function _logdensity_columns(m::ImplicitDiffusion{<:NoisePredictor{<:EnergyNetwork}}, X, t, ps, st)
    L = similar(float.(X), size(X, 2))
    for b in axes(X, 2)
        E, st = energy(m.predictor, X[:, b], t, ps, st)
        L[b] = -E
    end
    return (L, st)
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
    s = _schedule(m)
    E, st = _eps_columns(m, X, t, ps, st)
    return ((X .- sigma(s, t) .* E) ./ alpha(s, t), st)
end

# DDPM's ancestral step t → u < t: the Gaussian posterior q(x_u | x_t, x̂₀) with Tweedie's x̂₀,
# as its mean and standard deviation, and as a draw
function _ancestral_parts(m, X, t, u, ps, st)
    s = _schedule(m)
    X̂, st = _denoise_columns(m, X, t, ps, st)
    a = alpha(s, t) / alpha(s, u)                    # x_t = a x_u + σ_{t|u} ξ
    σt², σu² = sigma(s, t)^2, sigma(s, u)^2
    σtu² = σt² - a^2 * σu²
    mean = (a * σu² / σt²) .* X .+ (alpha(s, u) * σtu² / σt²) .* X̂
    return (mean, sqrt(σtu² * σu² / σt²), st)
end

function _ancestral(m, X, t, u, ps, st, rng)
    M, sd, st = _ancestral_parts(m, X, t, u, ps, st)
    return (M .+ sd .* randn(rng, size(X)...), st)
end

# forward from u back to t > u: x_t = a x_u + σ_{t|u} ξ
function _renoise(s, X, u, t, rng)
    a = alpha(s, t) / alpha(s, u)
    return a .* X .+ sqrt(sigma(s, t)^2 - a^2 * sigma(s, u)^2) .* randn(rng, size(X)...)
end
