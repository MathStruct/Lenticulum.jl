# ---------------------------------------------------------------------------
# Proximal Diffusion Models (Fang, Díaz, Buchanan & Sulam, arXiv:2507.08956).
#
# Score-based diffusion discretises the reverse SDE *forwards* and needs ∇log p_t. ProxDM
# discretises it *backwards* and needs the proximal map of -log p_t instead:
#
#     prox_{-λ log p_t}(v) = argmin_u ½‖u - v‖² - λ log p_t(u)
#
# learned as f_θ(v; t, λ) = v - √λ ε_θ(v; t, λ) by "proximal matching". Three things here:
#
#   1. the interface `proximal(p, v, t, λ, ps, st)`, with a Lux wrapper (`ProxNetwork`) and an
#      exact oracle for Gaussian mixtures (`MixtureProx`);
#   2. the sampler, Algorithm 1 (PDA and PDA-hybrid);
#   3. deterministic proximal inference for the implicit learner: half-quadratic splitting
#      between the prior's prox and the clamp's prox (`prox_infer`), and the training loss.
#
# See `proxdm.md` and `ProxDM and Proximal Alternatives.md`.
# ---------------------------------------------------------------------------

"""
    AbstractProximalPredictor

Anything that can evaluate ``\\operatorname{prox}_{-\\lambda\\log p_t}(v)`` through
[`proximal`](@ref)`(p, v, t, λ, ps, st) -> (u, st)`.
"""
abstract type AbstractProximalPredictor end

"""
    proximal(p, v, t, λ, ps, st) -> (u, st)

``\\operatorname{prox}_{-\\lambda\\log p_t}(v) = \\arg\\min_u \\tfrac12\\lVert u-v\\rVert^2-\\lambda\\log p_t(u)``:
a MAP denoiser at level `t` with regularisation `λ`. Its optimality condition is
``u - v = \\lambda\\nabla\\log p_t(u)`` — a backward (implicit) gradient step.
"""
function proximal end

"""
    ProxNetwork(model, schedule; input = (v, t, λ) -> (v, t, λ))

A learned proximal operator in ProxDM's parametrisation
``f_\\theta(v; t, \\lambda) = v - \\sqrt\\lambda\\,\\varepsilon_\\theta(v; t, \\lambda)``: the network predicts the
normalised residual of the prox, conditioned on **two** scalars, ``t`` and ``\\lambda``. `model` is
any Lux layer; `input` adapts ``(v, t, \\lambda)`` to what it expects. Parameters are the model's.
Train it with [`proximal_matching_loss`](@ref).
"""
struct ProxNetwork{L,S<:AbstractNoiseSchedule,F} <: LuxCore.AbstractLuxWrapperLayer{:model}
    model::L
    schedule::S
    input::F
end
ProxNetwork(model, schedule::AbstractNoiseSchedule; input = (v, t, λ) -> (v, t, λ)) =
    ProxNetwork(model, schedule, input)

function proximal(p::ProxNetwork, v, t, λ, ps, st)
    ε̂, st = LuxCore.apply(p.model, p.input(v, t, λ), ps, st)
    return (v .- sqrt(λ) .* ε̂, st)
end

"""
    MixtureProx(mixture::GaussianMixtureEps; tol = 1e-10, maxiters = 100)

The **exact** proximal operator of ``-\\log p_t`` for the Gaussian-mixture data distribution of
[`GaussianMixtureEps`](@ref), by a Newton solve with backtracking on the prox objective. It plays
the role a perfectly trained `ProxNetwork` would, so samplers and proximal inference can be
checked against arithmetic. Uses ``\\nabla\\log p_t`` directly (not ``\\varepsilon/\\sigma_t``), so it is
valid at ``t = 0``. Where ``-\\log p_t`` is not convex the prox can be multivalued; the solve
returns the local minimiser reached from ``v``.
"""
struct MixtureProx{G<:GaussianMixtureEps,T<:Real} <: AbstractProximalPredictor
    mixture::G
    tol::T
    maxiters::Int
end
MixtureProx(m::GaussianMixtureEps; tol = 1e-10, maxiters::Integer = 100) = MixtureProx(m, float(tol), Int(maxiters))

# ∇log p_t and its Hessian, straight from the mixture: -ū and -(I/v - Σ γ_j u_j (u_j - ū)ᵀ)
function _mixture_grad_hess(l::GaussianMixtureEps, x, t, μ)
    p = _mixture_parts(l, x, t, μ)
    ū = p.U * p.γ
    n = length(x)
    H = .-(Matrix{eltype(ū)}(I, n, n) ./ p.v .- (p.U .* p.γ') * (p.U .- ū)')
    return (.-ū, H)
end

function proximal(p::MixtureProx, v, t, λ, ps, st)
    l = p.mixture
    obj(u) = sum(abs2, u .- v) / 2 - λ * mixture_logdensity(l, u, t, ps)
    u = float.(copy(v))
    for _ in 1:(p.maxiters)
        g, H = _mixture_grad_hess(l, u, t, ps.μ)
        r = u .- v .- λ .* g                                  # optimality residual
        norm(r) ≤ p.tol * (1 + norm(u)) && break
        K = Matrix{eltype(u)}(I, length(u), length(u)) .- λ .* H
        d = isposdef(Symmetric((K .+ K') ./ 2)) ? -(K \ r) : -r   # Newton where convex, else gradient
        η, f0 = 1.0, obj(u)
        while obj(u .+ η .* d) > f0 - 1e-4 * η * abs(sum(r .* d)) && η > 1e-10
            η /= 2
        end
        u = u .+ η .* d
    end
    return (u, st)
end

# --- 2. the sampler: Algorithm 1 -------------------------------------------------

"""
    proxdm_sample(p, schedule, x, ps, st; steps = 50, hybrid = false, rng = Random.default_rng(),
                  T = 1.0) -> (x₀, st)

ProxDM's sampler (Fang et al., Algorithm 1), started from `x` (pure noise for the full reverse
process). With ``\\gamma_k = \\int_{t_{k-1}}^{t_k}\\beta(s)\\,ds`` on a uniform grid ``0 = t_0 < \\dots < t_N = T``:

- **PDA** (backward Euler): ``X_{k-1} = \\operatorname{prox}_{-\\frac{2\\gamma_k}{2-\\gamma_k}\\log p_{t_{k-1}}}\\bigl(\\tfrac{2}{2-\\gamma_k}(X_k + \\sqrt{\\gamma_k}\\,z_k)\\bigr)``, which needs ``\\gamma_k < 2``;
- **PDA-hybrid** (`hybrid = true`): ``X_{k-1} = \\operatorname{prox}_{-\\gamma_k\\log p_{t_{k-1}}}\\bigl((1+\\tfrac12\\gamma_k)X_k + \\sqrt{\\gamma_k}\\,z_k\\bigr)``, no step-size limit.

Both follow from discretising the drift of the reverse VP-SDE at the *new* point (hybrid: the
score term only), which turns each step into a proximal step. Unlike score-based samplers the
last step denoises *after* adding noise, so no final denoising step is needed.
"""
function proxdm_sample(p, s::AbstractNoiseSchedule, x, ps, st; steps::Integer = 50, hybrid::Bool = false,
                       rng::AbstractRNG = Random.default_rng(), T = 1.0)
    ts = range(0.0, float(T); length = steps + 1)
    x = float.(copy(x))
    for k in steps:-1:1
        γ = integrated_beta(s, ts[k + 1]) - integrated_beta(s, ts[k])
        z = randn(rng, eltype(x), size(x))
        if hybrid
            x, st = proximal(p, (1 + γ / 2) .* x .+ sqrt(γ) .* z, ts[k], γ, ps, st)
        else
            γ < 2 || throw(ArgumentError("PDA needs γ_k < 2 (got $γ at step $k); use more steps or hybrid = true"))
            x, st = proximal(p, (2 / (2 - γ)) .* (x .+ sqrt(γ) .* z), ts[k], 2γ / (2 - γ), ps, st)
        end
    end
    return (x, st)
end

# --- 3. proximal inference for the implicit learner, and training -----------------

"""
    prox_infer(p, z₀, ρ, t, ps, st; λ = 0.01, z_init = z₀, maxiters = 500, tol = 1e-9)
        -> (ImplicitSolution, st)

Deterministic inference for the implicit learner with a **proximal** prior instead of a score:
half-quadratic splitting between the prior's prox at level `t` and the clamp's prox,

```math
u \\leftarrow \\operatorname{prox}_{-\\lambda\\log p_t}(w),\\qquad
w_i \\leftarrow \\frac{u_i/\\lambda + \\rho_i^2 z_{0,i}}{1/\\lambda + \\rho_i^2}\\quad(\\rho_i<\\infty),\\qquad
w_i \\leftarrow z_{0,i}\\quad(\\rho_i=\\infty).
```

A fixed point ``(u, w)`` is a stationary point of
``-\\log p_t(u) + \\tfrac1{2\\lambda}\\lVert u-w\\rVert^2 + \\tfrac12\\lVert P(w - z_0)\\rVert^2`` — the
relaxed problem whose ``\\lambda\\to0`` limit is the implicit learner's (`Deterministic
Relaxation.md`); this is the plug-and-play / DiffPIR splitting. Hard clamps are exact. Returns `w`
(which satisfies the clamps) as `z`, with `residual` ``= \\lVert u - w\\rVert``; `stable` is not
assessed by this solver and is reported `true` when converged.
"""
function prox_infer(p, z₀, ρ, t, ps, st; λ = 0.01, z_init = z₀, maxiters::Integer = 500, tol = 1e-9)
    hard = isinf.(ρ)
    w = float.(copy(z_init)); w[hard] .= z₀[hard]
    u = copy(w)
    for it in 1:maxiters
        u, st = proximal(p, w, t, λ, ps, st)
        wn = similar(w)
        wn[hard] .= z₀[hard]
        wn[.!hard] .= (u[.!hard] ./ λ .+ ρ[.!hard] .^ 2 .* z₀[.!hard]) ./ (1 / λ .+ ρ[.!hard] .^ 2)
        δ = norm(wn .- w)
        w = wn
        if δ ≤ tol * (1 + norm(w))
            return (ImplicitSolution(w, norm(u .- w), it, true, true), st)
        end
    end
    return (ImplicitSolution(w, norm(u .- w), maxiters, false, false), st)
end

"""
    proximal_matching_loss(p::ProxNetwork, x_t, t, λ, ε, ζ, ps, st) -> (ℓ, st)

One sample of ProxDM's training objective (their Eq. 9):
``\\ell_{PM}\\bigl(\\varepsilon_\\theta(x_t + \\sqrt\\lambda\\,\\varepsilon; t, \\lambda), \\varepsilon; \\zeta\\bigr)`` with
``\\ell_{PM}(a, b; \\zeta) = 1 - \\exp\\bigl(-\\lVert a-b\\rVert^2/(d\\zeta^2)\\bigr)``. As ``\\zeta\\to0`` its minimiser is
the prox (a MAP denoiser), where the squared loss would give the MMSE denoiser. Shrink ``\\zeta``
during training. `x_t` is a sample of ``p_t``, `ε` a fresh standard normal.
"""
function proximal_matching_loss(p::ProxNetwork, x_t, t, λ, ε, ζ, ps, st)
    ε̂, st = LuxCore.apply(p.model, p.input(x_t .+ sqrt(λ) .* ε, t, λ), ps, st)
    d = length(ε)
    return (1 - exp(-sum(abs2, ε̂ .- ε) / (d * ζ^2)), st)
end
