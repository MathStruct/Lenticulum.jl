# ---------------------------------------------------------------------------
# Closed-form noise predictors: the oracles.
#
# For a Gaussian-mixture data distribution  p₀ = (1/J) Σ_j N(μ_j, s² I)  the optimal noise
# predictor is available exactly:
#
#     p_t(x)    = (1/J) Σ_j N(x; α_t μ_j, v_t I),       v_t = α_t² s² + σ_t²
#     ε*(x, t)  = -σ_t ∇log p_t(x) = σ_t Σ_j γ_j(x) u_j,  u_j = (x - α_t μ_j)/v_t
#
# with responsibilities γ_j(x) ∝ exp(-‖x - α_t μ_j‖²/(2v_t)). A mixture of narrow components
# placed along a curve is a *relation* with an exact score — a ring of components is the unit
# circle — so inference, its Jacobian and the backward pass can all be checked against
# arithmetic. The means μ are the parameters, so the backward pass has something to train.
#
# See `analytic.md` and `Implicit Diffusion Learners.md`.
# ---------------------------------------------------------------------------

"""
    GaussianMixtureEps(schedule, μ₀::AbstractMatrix; s = 0.05)

The exact noise predictor ``\\varepsilon^\\ast`` of the data distribution
``\\frac1J\\sum_j\\mathcal N(\\mu_j, s^2 I)`` under `schedule`. Columns of `μ₀` (an ``n\\times J``
matrix) are the initial component means; they are the layer's **parameters**
(`ps.μ`), so a mixture can be moved by gradient descent. The component width `s` is fixed.

Wrap it in a [`NoisePredictor`](@ref) with the same schedule. Its input Jacobian
([`epsilon_jacobian`](@ref)) and parameter VJP ([`epsilon_vjp_params`](@ref)) are exact.
"""
struct GaussianMixtureEps{S<:AbstractNoiseSchedule,M<:AbstractMatrix,T<:Real} <: LuxCore.AbstractLuxLayer
    schedule::S
    μ₀::M
    s::T
end
GaussianMixtureEps(schedule::AbstractNoiseSchedule, μ₀::AbstractMatrix; s = 0.05) =
    GaussianMixtureEps(schedule, float.(μ₀), float(s))

LuxCore.initialparameters(::AbstractRNG, l::GaussianMixtureEps) = (μ = copy(l.μ₀),)
LuxCore.initialstates(::AbstractRNG, ::GaussianMixtureEps) = NamedTuple()

# u_j, γ_j and v_t at (x, t): the three quantities every formula below is built from
function _mixture_parts(l::GaussianMixtureEps, x, t, μ)
    a, sg = alpha(l.schedule, t), sigma(l.schedule, t)
    v = a^2 * l.s^2 + sg^2
    D = x .- a .* μ                                   # n × J
    ℓ = vec(-sum(abs2, D; dims = 1)) ./ (2v)
    γ = exp.(ℓ .- maximum(ℓ))
    γ ./= sum(γ)
    return (U = D ./ v, γ = γ, v = v, a = a, sg = sg, ℓ = ℓ)
end

function (l::GaussianMixtureEps)(inp, ps, st)
    x, t = inp
    p = _mixture_parts(l, x, t, ps.μ)
    return (p.sg .* (p.U * p.γ), st)
end

"""
    mixture_logdensity(l::GaussianMixtureEps, x, t, ps) -> Real

``\\log p_t(x)`` of the mixture. Used to check that the deterministic RED-Diff field is the
gradient of a smoothed log-density (`Implicit Diffusion Learners.md` §3).
"""
function mixture_logdensity(l::GaussianMixtureEps, x, t, ps)
    p = _mixture_parts(l, x, t, ps.μ)
    m = maximum(p.ℓ)
    return m + log(sum(exp.(p.ℓ .- m)) / length(p.ℓ)) - length(x) / 2 * log(2π * p.v)
end

# --- the two derivatives the backward pass needs, in closed form -----------------

"""
    epsilon_jacobian(pred::NoisePredictor, x, t, ps, st) -> Matrix

``\\partial\\varepsilon_\\theta/\\partial x`` at ``(x, t)``. The generic method uses central finite
differences (``O(n)`` forward passes; fine for small ``n``), or the predictor's AD backend when it
has one (`ad = …`); closed-form predictors override it.
For an exact score the matrix is **symmetric** — it is ``-\\sigma_t`` times a Hessian of
``\\log p_t`` — and the backward pass does not assume that, because a learned network's is not.
"""
function epsilon_jacobian(pred::NoisePredictor, x, t, ps, st)
    pred.ad === nothing || return _ad_jacobian(pred.ad, pred, x, t, ps, st)
    n = length(x)
    J = zeros(eltype(float(x)), n, n)
    for i in 1:n
        h = 1e-6 * (1 + abs(x[i]))
        e = zeros(eltype(J), n); e[i] = h
        fp, _ = epsilon(pred, x .+ e, t, ps, st)
        fm, _ = epsilon(pred, x .- e, t, ps, st)
        J[:, i] = (fp .- fm) ./ (2h)
    end
    return J
end

# exact:  ∂ε/∂x = σ_t [ I/v_t - Σ_j γ_j u_j (u_j - ū)ᵀ ],  ū = Σ_j γ_j u_j   (symmetric)
function epsilon_jacobian(pred::NoisePredictor{<:GaussianMixtureEps}, x, t, ps, st)
    p = _mixture_parts(pred.model, x, t, ps.μ)
    ū = p.U * p.γ
    n = length(x)
    return p.sg .* (Matrix{eltype(ū)}(I, n, n) ./ p.v .- (p.U .* p.γ') * (p.U .- ū)')
end

"""
    epsilon_vjp_params(pred::NoisePredictor, x, t, ps, st, w) -> ps̄

The vector–Jacobian product ``w^\\top\\,\\partial\\varepsilon_\\theta(x,t)/\\partial\\theta``, shaped
like `ps`. This is the one place the backward pass needs a derivative *with respect to the
network's parameters*.

- Closed-form predictors define it directly (e.g. [`GaussianMixtureEps`](@ref)).
- Any other model gets it from an **AD backend chosen by the user**: build the predictor with
  `NoisePredictor(model, schedule; ad = AutoZygote())` (or `AutoEnzyme()`, `AutoMooncake()`,
  `AutoForwardDiff()`, … — any `ADTypes` backend) and load `DifferentiationInterface` together
  with the backend package. This package itself depends on none of them.

Without either, it throws.
"""
epsilon_vjp_params(pred::NoisePredictor, x, t, ps, st, w) = _ad_vjp_params(pred.ad, pred, x, t, ps, st, w)

_ad_vjp_params(::Nothing, pred, x, t, ps, st, w) = throw(ArgumentError(
    "epsilon_vjp_params needs w'∂ε/∂θ for $(typeof(pred.model)). Construct the predictor with an AD " *
    "backend, e.g. NoisePredictor(model, schedule; ad = AutoZygote()), and load DifferentiationInterface " *
    "and the backend package."))
_ad_vjp_params(ad, pred, x, t, ps, st, w) = throw(ArgumentError(
    "ad = $(ad) given, but no extension implements it: load DifferentiationInterface (and the backend package)."))

# input Jacobian through the chosen AD backend; the extension adds the method
_ad_jacobian(ad, pred, x, t, ps, st) = throw(ArgumentError(
    "ad = $(ad) given, but no extension implements it: load DifferentiationInterface (and the backend package)."))

# exact:  (∂ε/∂μ_j)ᵀ w = σ_t α_t γ_j [ -w/v_t + u_j (u_j - ū)ᵀ w ]
function epsilon_vjp_params(pred::NoisePredictor{<:GaussianMixtureEps}, x, t, ps, st, w)
    p = _mixture_parts(pred.model, x, t, ps.μ)
    ū = p.U * p.γ
    c = (p.U .- ū)' * w                               # (u_j - ū)ᵀ w, one per component
    G = (p.sg * p.a) .* ((.-w ./ p.v) .* p.γ' .+ p.U .* (p.γ .* c)')
    return (μ = G,)
end

# --- the kernel baseline: a Gaussian KDE is this mixture with one centre per sample ----------

"""
    kde_predictor(schedule, data::AbstractMatrix; bandwidth) -> NoisePredictor

A Gaussian **kernel density estimate** of the samples in the columns of `data`, as a noise
predictor: [`GaussianMixtureEps`](@ref) with one component per sample and width `bandwidth`.
Its ``\\varepsilon`` is the exact optimal noise predictor for that density, so every query function
([`implicit_infer`](@ref), [`implicit_roots`](@ref), [`implicit_laplace`](@ref)) works on it with
exact derivatives and no training. It is the classical baseline for a diffusion network: the
relation it defines is the ridge of the KDE. Query cost grows linearly with the number of
samples. Choose the bandwidth with [`kde_bandwidth`](@ref).
"""
kde_predictor(s::AbstractNoiseSchedule, data::AbstractMatrix; bandwidth::Real) =
    NoisePredictor(GaussianMixtureEps(s, float.(data); s = bandwidth), s)

"""
    kde_bandwidth(data; candidates = nothing, holdout = 0.2, rng = Xoshiro(0)) -> h

The bandwidth of an isotropic Gaussian KDE that maximises the **held-out log-likelihood**: a
random fraction `holdout` of the columns of `data` is set aside, a KDE on the rest is evaluated
on it, for each candidate. The default candidates span 0.5 % to 100 % of the data's scale
(the root mean coordinate variance) on a log grid. This chooses the best *density*, not the
best answers to queries, and uses no query information.
"""
function kde_bandwidth(data::AbstractMatrix; candidates = nothing, holdout = 0.2,
                       rng::AbstractRNG = Random.Xoshiro(0))
    n, d = size(data, 2), size(data, 1)
    perm = Random.randperm(rng, n)
    m = max(1, round(Int, holdout * n))
    test, train = data[:, perm[1:m]], data[:, perm[(m + 1):end]]
    μ = sum(data; dims = 2) ./ n
    scale = sqrt(sum(abs2, data .- μ) / (n * d))
    hs = candidates === nothing ? scale .* exp.(range(log(0.005), 0; length = 25)) : collect(candidates)
    D = [sum(abs2, view(test, :, j) .- view(train, :, i)) for i in axes(train, 2), j in axes(test, 2)]
    function heldout(h)
        ℓ = -D ./ (2h^2)
        mx = maximum(ℓ; dims = 1)
        return sum(mx .+ log.(sum(exp.(ℓ .- mx); dims = 1) ./ size(train, 2))) / m - d / 2 * log(2π * h^2)
    end
    return hs[argmax(heldout.(hs))]
end
