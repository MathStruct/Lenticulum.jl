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
    ProxNetwork(model, schedule; input = (v, t, λ) -> (v, t, λ), ad = nothing)

A learned proximal operator in ProxDM's parametrisation
``f_\\theta(v; t, \\lambda) = v - \\sqrt\\lambda\\,\\varepsilon_\\theta(v; t, \\lambda)``: the network predicts the
normalised residual of the prox, conditioned on **two** scalars, ``t`` and ``\\lambda``. `model` is
any Lux layer; `input` adapts ``(v, t, \\lambda)`` to what it expects. Parameters are the model's.
Train it with [`proximal_matching_loss`](@ref).

`ad` is an ADTypes backend (`AutoZygote()`, …; load DifferentiationInterface and the backend)
for the derivatives the adjoint of [`prox_infer`](@ref) needs: the parameter VJP, which has no
other route, and the input VJP, which otherwise falls back to finite differences.
"""
struct ProxNetwork{L,S<:AbstractNoiseSchedule,F,A} <: LuxCore.AbstractLuxWrapperLayer{:model}
    model::L
    schedule::S
    input::F
    ad::A
end
ProxNetwork(model, schedule::AbstractNoiseSchedule; input = (v, t, λ) -> (v, t, λ), ad = nothing) =
    ProxNetwork(model, schedule, input, ad)

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
    proxdm_sample(p, schedule, x, ps, st; steps = 50, scheme = :pda, rng = Random.default_rng(),
                  T = 1.0) -> (x₀, st)

ProxDM's sampler (Fang et al., Algorithm 1), started from `x` (pure noise for the full reverse
process). With ``\\gamma_k = \\int_{t_{k-1}}^{t_k}\\beta(s)\\,ds`` on a uniform grid ``0 = t_0 < \\dots < t_N = T``:

- **PDA** (backward Euler): ``X_{k-1} = \\operatorname{prox}_{-\\frac{2\\gamma_k}{2-\\gamma_k}\\log p_{t_{k-1}}}\\bigl(\\tfrac{2}{2-\\gamma_k}(X_k + \\sqrt{\\gamma_k}\\,z_k)\\bigr)``, which needs ``\\gamma_k < 2``;
- **PDA-hybrid** (`scheme = :hybrid`): ``X_{k-1} = \\operatorname{prox}_{-\\gamma_k\\log p_{t_{k-1}}}\\bigl((1+\\tfrac12\\gamma_k)X_k + \\sqrt{\\gamma_k}\\,z_k\\bigr)``, no step-size limit.
- **Trapezoidal** (`scheme = :trapezoidal`, second order): the drift averaged over both ends,
  ``X_{k-1} = X_k + \\tfrac{\\gamma_k}2\\bigl[(\\tfrac12X_k + s_k) + (\\tfrac12X_{k-1} + s_{k-1})\\bigr] + \\sqrt{\\gamma_k}\\,z_k``, i.e.
  ``X_{k-1} = \\operatorname{prox}_{-\\frac{2\\gamma_k}{4-\\gamma_k}\\log p_{t_{k-1}}}\\bigl(\\tfrac{(1+\\gamma_k/4)X_k + (\\gamma_k/2)s_k + \\sqrt{\\gamma_k}z_k}{1-\\gamma_k/4}\\bigr)``,
  which needs ``\\gamma_k < 4``. The explicit score ``s_k = \\nabla\\log p_{t_k}(X_k)`` costs nothing: the
  previous prox returned ``X_k = \\operatorname{prox}_\\lambda(v)``, and its optimality condition gives
  ``s_k = (X_k - v)/\\lambda``. The first step is a hybrid step, which supplies the first score.
  Still one prox call per step.

PDA and hybrid follow from discretising the drift of the reverse VP-SDE at the *new* point
(hybrid: the score term only), which turns each step into a proximal step; they are first order
in the step (the spread converges like ``1/N``), the trapezoidal scheme second order. Unlike
score-based samplers the last step denoises *after* adding noise, so no final denoising step is
needed. `hybrid = true` is the old spelling of `scheme = :hybrid`.
"""
function proxdm_sample(p, s::AbstractNoiseSchedule, x, ps, st; steps::Integer = 50, hybrid::Bool = false,
                       scheme::Symbol = hybrid ? :hybrid : :pda, rng::AbstractRNG = Random.default_rng(), T = 1.0)
    scheme in (:pda, :hybrid, :trapezoidal) || throw(ArgumentError("scheme must be :pda, :hybrid or :trapezoidal, got :$scheme"))
    ts = range(0.0, float(T); length = steps + 1)
    x = float.(copy(x))
    score = nothing                                   # ∇log p_t at x, once a prox has returned it
    for k in steps:-1:1
        γ = integrated_beta(s, ts[k + 1]) - integrated_beta(s, ts[k])
        z = randn(rng, eltype(x), size(x))
        if scheme === :trapezoidal && score !== nothing
            γ < 4 || throw(ArgumentError("the trapezoidal scheme needs γ_k < 4 (got $γ at step $k); use more steps"))
            λ = (γ / 2) / (1 - γ / 4)
            v = ((1 + γ / 4) .* x .+ (γ / 2) .* score .+ sqrt(γ) .* z) ./ (1 - γ / 4)
            x, st = proximal(p, v, ts[k], λ, ps, st)
            score = (x .- v) ./ λ                     # the prox's optimality condition
        elseif scheme !== :pda                        # hybrid, and the trapezoidal scheme's first step
            v = (1 + γ / 2) .* x .+ sqrt(γ) .* z
            x, st = proximal(p, v, ts[k], γ, ps, st)
            score = (x .- v) ./ γ
        else
            γ < 2 || throw(ArgumentError("PDA needs γ_k < 2 (got $γ at step $k); use more steps or hybrid = true"))
            x, st = proximal(p, (2 / (2 - γ)) .* (x .+ sqrt(γ) .* z), ts[k], 2γ / (2 - γ), ps, st)
        end
    end
    return (x, st)
end

# --- 3. proximal inference for the implicit learner, and training -----------------

"""
    prox_infer(p, z₀, ρ, t, ps, st; λ = 0.01, z_init = z₀, maxiters = 500, tol = 1e-9,
               linsolve = DenseSolve()) -> (ImplicitSolution, st)

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

Its derivative is [`prox_pullback`](@ref), which is also the `ChainRulesCore.rrule` of this
function; `linsolve` is the linear solver that adjoint uses (the forward pass does not use it).
"""
function prox_infer(p, z₀, ρ, t, ps, st; λ = 0.01, z_init = z₀, maxiters::Integer = 500, tol = 1e-9,
                    linsolve::AbstractLinearSolve = DenseSolve())
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

# --- 4. the adjoint of proximal inference -------------------------------------------

"""
    prox_vjp_input(p, v, t, λ, ps, st, y) -> v̄

``(\\partial_v \\operatorname{prox}_{-\\lambda\\log p_t}(v))^\\top y``. Exact for [`MixtureProx`](@ref) (the
implicit function theorem on the prox's optimality condition: ``(I - \\lambda\\nabla^2\\log p_t(u))^{-1}y``);
through `ad` for a [`ProxNetwork`](@ref); by central differences otherwise.
"""
prox_vjp_input(p, v, t, λ, ps, st, y) = prox_jacobian(p, v, t, λ, ps, st)' * y

"""
    prox_vjp_params(p, v, t, λ, ps, st, y) -> ps̄

``(\\partial_\\theta \\operatorname{prox}_{-\\lambda\\log p_t}(v))^\\top y``, shaped like `ps`. Exact for
[`MixtureProx`](@ref); through `ad` for a [`ProxNetwork`](@ref).
"""
prox_vjp_params(p, v, t, λ, ps, st, y) = throw(ArgumentError(
    "prox_vjp_params has no method for $(typeof(p)): implement it, or use a ProxNetwork with an AD backend"))

"""
    prox_jacobian(p, v, t, λ, ps, st) -> Matrix

The input Jacobian of the prox at `v`. Exact for [`MixtureProx`](@ref); through `ad` for a
[`ProxNetwork`](@ref); central differences (step ``10^{-6}(1 + \\lVert v\\rVert_\\infty)``) otherwise.
"""
function prox_jacobian(p, v, t, λ, ps, st)
    v = float.(collect(v))
    h = 1e-6 * (1 + maximum(abs, v))
    J = zeros(eltype(v), length(v), length(v))
    for j in eachindex(v)
        e = h .* (eachindex(v) .== j)
        J[:, j] .= (vec(first(proximal(p, v .+ e, t, λ, ps, st))) .- vec(first(proximal(p, v .- e, t, λ, ps, st)))) ./ (2h)
    end
    return J
end

# MixtureProx: u - v = λ g(u; μ) with g = ∇log p_t, so du = K⁻¹(dv + λ ∂g/∂μ dμ), K = I - λ∇²log p_t(u),
# which is symmetric. (∂g/∂μ_j)ᵀ q = -α γ_j [-q/v_t + u_j (u_j - ū)ᵀ q], the ε formula of
# `epsilon_vjp_params` without the factor σ_t, so it holds at t = 0 too.
function _prox_K(p::MixtureProx, v, t, λ, ps, st)
    u, _ = proximal(p, v, t, λ, ps, st)
    _, H = _mixture_grad_hess(p.mixture, u, t, ps.μ)
    return (u, Symmetric(Matrix{eltype(u)}(I, length(u), length(u)) .- λ .* (H .+ H') ./ 2))
end
prox_jacobian(p::MixtureProx, v, t, λ, ps, st) = inv(last(_prox_K(p, v, t, λ, ps, st)))
prox_vjp_input(p::MixtureProx, v, t, λ, ps, st, y) = last(_prox_K(p, v, t, λ, ps, st)) \ y
function prox_vjp_params(p::MixtureProx, v, t, λ, ps, st, y)
    u, K = _prox_K(p, v, t, λ, ps, st)
    q = K \ y
    m = _mixture_parts(p.mixture, u, t, ps.μ)
    ū = m.U * m.γ
    c = (m.U .- ū)' * q
    return (μ = (-λ * m.a) .* ((.-q ./ m.v) .* m.γ' .+ m.U .* (m.γ .* c)'),)
end

# ProxNetwork: prox(v) = v - √λ ε̂(v), so the VJPs are y - √λ (∂ε̂/∂v)ᵀy and -√λ (∂ε̂/∂θ)ᵀy
prox_vjp_input(p::ProxNetwork, v, t, λ, ps, st, y) =
    p.ad === nothing ? prox_jacobian(p, v, t, λ, ps, st)' * y : y .- sqrt(λ) .* _prox_ad_vjp_input(p.ad, p, v, t, λ, ps, st, y)
prox_jacobian(p::ProxNetwork, v, t, λ, ps, st) =
    p.ad === nothing ? invoke(prox_jacobian, Tuple{Any,Any,Any,Any,Any,Any}, p, v, t, λ, ps, st) :
    I - sqrt(λ) .* _prox_ad_jacobian(p.ad, p, v, t, λ, ps, st)
function prox_vjp_params(p::ProxNetwork, v, t, λ, ps, st, y)
    p.ad === nothing && throw(ArgumentError(
        "prox_vjp_params needs an AD backend: ProxNetwork(model, schedule; input, ad = AutoZygote()), " *
        "with DifferentiationInterface and the backend package loaded"))
    return _scale(-sqrt(λ), _prox_ad_vjp_params(p.ad, p, v, t, λ, ps, st, y))
end
_scale(a, x::NamedTuple) = map(xi -> _scale(a, xi), x)
_scale(a, x::AbstractArray) = a .* x
_scale(a, x::Real) = a * x

# the extension adds methods for ADTypes backends
_prox_ad_vjp_input(ad, p, v, t, λ, ps, st, y) = _no_prox_ad(ad)
_prox_ad_vjp_params(ad, p, v, t, λ, ps, st, y) = _no_prox_ad(ad)
_prox_ad_jacobian(ad, p, v, t, λ, ps, st) = _no_prox_ad(ad)
_no_prox_ad(ad) = throw(ArgumentError(
    "ad = $(ad) given, but no extension implements it: load DifferentiationInterface (and the backend package)."))

"""
    prox_pullback(p, sol, z₀, ρ, t, z̄, ps, st; λ = 0.01, linsolve = DenseSolve())
        -> (z₀ = z̄₀, ρ = ρ̄, ps = ps̄)

Reverse-mode derivative of [`prox_infer`](@ref)'s answer ``w^\\star(z_0, \\rho, \\theta)`` by the
adjoint method; `λ` and `t` must be the ones inference used. The fixed point of the splitting
satisfies, on the coordinates that are not hard-clamped (``F``),

```math
w_F = a\\odot\\operatorname{prox}(w)_F + (1-a)\\odot z_{0,F},\\qquad a_i = \\frac1{1+\\lambda\\rho_i^2},
```

and ``w_C = z_{0,C}`` on the clamped ones. With ``D`` the prox's input Jacobian at ``w`` and
``u = \\operatorname{prox}(w)``, solve once

```math
(I - \\operatorname{diag}(a)\\,D_{FF})^\\top\\mu = \\bar z_F,\\qquad y = (a\\odot\\mu \\text{ on } F,\\ 0 \\text{ on } C),
```

and read off every gradient:

| w.r.t. | cotangent |
|---|---|
| parameters ``\\theta`` | ``(\\partial_\\theta\\operatorname{prox})^\\top y`` (via [`prox_vjp_params`](@ref)) |
| hard-clamped inputs ``z_{0,C}`` | ``\\bar z_C + (D^\\top y)_C`` |
| soft and free targets ``z_{0,F}`` | ``(1-a)\\odot\\mu`` |
| precisions ``\\rho_F`` | ``-2\\lambda\\rho_F\\odot a^2\\odot(u_F - z_{0,F})\\odot\\mu`` |

Only vector-Jacobian products of the prox appear, so `linsolve = KrylovSolve()` runs GMRES on
them and never forms ``D``; `DenseSolve()` forms it ([`prox_jacobian`](@ref)). Throws if `sol` did
not converge. See `proxdm.md` §5.
"""
function prox_pullback(p, sol::ImplicitSolution, z₀, ρ, t, z̄, ps, st; λ = 0.01,
                       linsolve::AbstractLinearSolve = DenseSolve())
    sol.converged || throw(ArgumentError("prox_pullback: the solve did not converge, so there is no fixed point to differentiate"))
    w = sol.z
    n = length(w)
    T = float(eltype(w))
    F = findall(!isinf, ρ)
    C = findall(isinf, ρ)
    a = 1 ./ (1 .+ λ .* T.(ρ[F]) .^ 2)
    μ = _prox_adjoint(linsolve, p, w, t, λ, ps, st, F, a, T.(z̄[F]))
    y = zeros(T, n)
    y[F] .= a .* μ
    z̄₀ = zeros(T, n)
    z̄₀[F] .= (1 .- a) .* μ
    if !isempty(C)
        q = prox_vjp_input(p, w, t, λ, ps, st, y)
        z̄₀[C] .= z̄[C] .+ q[C]
    end
    ρ̄ = zeros(T, n)
    if any(>(0), ρ[F])
        u, _ = proximal(p, w, t, λ, ps, st)
        ρ̄[F] .= -2λ .* ρ[F] .* a .^ 2 .* (u[F] .- z₀[F]) .* μ
    end
    ps̄ = _hasparams(ps) ? prox_vjp_params(p, w, t, λ, ps, st, y) : ps
    return (z₀ = z̄₀, ρ = ρ̄, ps = ps̄)
end

function _prox_adjoint(::DenseSolve, p, w, t, λ, ps, st, F, a, b)
    D = prox_jacobian(p, w, t, λ, ps, st)
    return (I - Diagonal(a) * D[F, F])' \ b
end

function _prox_adjoint(ls::KrylovSolve, p, w, t, λ, ps, st, F, a, b)
    y = zeros(eltype(b), length(w))
    function At(μ)                                   # μ - (Dᵀ y)_F with y = a⊙μ on F
        fill!(y, 0)
        y[F] .= a .* μ
        return μ .- prox_vjp_input(p, w, t, λ, ps, st, y)[F]
    end
    μ, ok, k = _gmres(At, b; rtol = ls.rtol, maxiters = ls.maxiters, restart = ls.restart)
    ok || @warn "prox_pullback: GMRES stopped after $k products without reaching rtol = $(ls.rtol); the gradient is approximate"
    return μ
end

# --- 5. as a factor's inversion ------------------------------------------------------

"""
    ProxDM(prox; t = 0.005, λ = 0.01, tol = 1e-9, maxiters = 500, linsolve = DenseSolve(),
           nodes = nothing)

Proximal inference ([`prox_infer`](@ref)) as a [`DiffusionFactor`](@ref)'s inversion, in place
of [`REDDiff`](@ref) or [`ImplicitProx`](@ref):

```julia
f = DiffusionFactor((x = 1, y = 1), pred; prox = ProxDM(MixtureProx(mixture)))
ps, st = LuxCore.setup(rng, f)          # (predictor = …, prox = …), two parameter trees
```

`prox` is the proximal predictor ([`MixtureProx`](@ref), a [`ProxNetwork`](@ref), or any
[`AbstractProximalPredictor`](@ref)); `t` and `λ` are the prox's noise level and the
splitting's coupling. Inversion runs `prox_infer` on the factor's state space with the
polarity's precisions (`Inf` = hard clamp), warm-started at the incoming points, and returns a
`DiracBelief` on the target; [`implicit_solution`](@ref) returns the report and
[`implicit_factor_pullback`](@ref) the backward pass (through [`prox_pullback`](@ref), with
`linsolve`).

**Two parameter trees.** The factor keeps its `NoisePredictor`, which the free energy's score
summand needs; the prox is a second model with its own parameters. The factor's parameters and
states are therefore `(predictor = …, prox = …)`, and the inversion reads only `prox`. A
`ProxNetwork` must be a Lux layer; a `MixtureProx` gets `(μ = …,)`.

`nodes` (a [`FieldNodes`](@ref) of the factor's dimension) fix the score summand of the free
energy, as for `ImplicitProx` with ``\\lambda = 1``; by default 8 nodes from `Xoshiro(0)`.
"""
struct ProxDM{P,T<:Real,L<:AbstractLinearSolve,N}
    prox::P
    t::T
    λ::T
    tol::T
    maxiters::Int
    linsolve::L
    nodes::N
end
ProxDM(prox; t = 0.005, λ = 0.01, tol = 1e-9, maxiters::Integer = 500,
       linsolve::AbstractLinearSolve = DenseSolve(), nodes::Union{Nothing,FieldNodes} = nothing) =
    ProxDM(prox, float(t), float(λ), float(tol), Int(maxiters), linsolve, nodes)

# MixtureProx is not a Lux layer, but a factor has to set it up: its parameters are the means
LuxCore.initialparameters(::AbstractRNG, p::MixtureProx) = (μ = copy(p.mixture.μ₀),)
LuxCore.initialstates(::AbstractRNG, ::MixtureProx) = NamedTuple()
LuxCore.parameterlength(p::MixtureProx) = length(p.mixture.μ₀)
LuxCore.statelength(::MixtureProx) = 0
