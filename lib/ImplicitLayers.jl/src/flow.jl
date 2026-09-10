# ---------------------------------------------------------------------------
# Fixed-step explicit integrators, forwards and backwards.
#
# The one fact that makes a NeuralODE a good Lenticulum factor:
#
#     a flow is a DIFFEOMORPHISM, so integrating with dt < 0 inverts it.
#
# No root-finding, no learned inverse, no architectural constraint — the reverse direction is
# the same code with the endpoints swapped. That makes `NeuralODEFactor` the cheapest genuinely
# bidirectional factor in the project, and the contrast with `DEQFactor` (whose reverse
# direction is a hard root-find) is the most useful thing in this package.
#
# It is exact in exact arithmetic and NOT exact in floating point; §"round trip" in `flow.md`
# is about the difference, which is the well-known reverse-mode instability of NeuralODEs.
#
# See `flow.md`.
# ---------------------------------------------------------------------------

"""
    abstract type AbstractIntegrator

A fixed-step explicit scheme. Deliberately not adaptive: an adaptive solver chooses different
steps forwards and backwards, which destroys the round-trip property this package leans on
(`flow.md` §3).
"""
abstract type AbstractIntegrator end

"""
    EulerIntegrator(steps = 50)

Explicit Euler. First order, and present mainly as the thing `RK4Integrator` is checked
against — its error is visible enough to make the order of accuracy testable.
"""
struct EulerIntegrator <: AbstractIntegrator
    steps::Int
end
EulerIntegrator(; steps = 50) = EulerIntegrator(steps)

"""
    RK4Integrator(steps = 50)

Classical fourth-order Runge–Kutta, fixed step. The default: for the smooth non-stiff
dynamics a NeuralODE usually has, `steps = 50` of RK4 is far more accurate than anything
Euler reaches, and it is `Tsit5`'s spiritual ancestor without the adaptivity.
"""
struct RK4Integrator <: AbstractIntegrator
    steps::Int
end
RK4Integrator(; steps = 50) = RK4Integrator(steps)

nsteps(i::AbstractIntegrator) = i.steps

"""
    integrate(vf, u₀, t₀, t₁, integrator) -> u₁

Integrate ``du/dt = \\mathrm{vf}(u, t)`` from `t₀` to `t₁`.

**`t₁ < t₀` is allowed and is the whole point**: `dt` is simply negative and the same scheme
runs backwards. `vf` is called as `vf(u, t)`.
"""
function integrate(vf, u₀, t₀, t₁, itg::EulerIntegrator)
    n = nsteps(itg)
    dt = (t₁ - t₀) / n
    u = copy(u₀)
    t = t₀
    for _ in 1:n
        u = u .+ dt .* vf(u, t)
        t += dt
    end
    return u
end

function integrate(vf, u₀, t₀, t₁, itg::RK4Integrator)
    n = nsteps(itg)
    dt = (t₁ - t₀) / n
    u = copy(u₀)
    t = t₀
    for _ in 1:n
        k1 = vf(u, t)
        k2 = vf(u .+ (dt / 2) .* k1, t + dt / 2)
        k3 = vf(u .+ (dt / 2) .* k2, t + dt / 2)
        k4 = vf(u .+ dt .* k3, t + dt)
        u = u .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        t += dt
    end
    return u
end

"""
    integrate_with_divergence(vf, u₀, t₀, t₁, integrator) -> (u₁, Δlogdet)

Integrate the state **and** the log-density correction of a continuous normalising flow:

```math
\\frac{d}{dt}\\log p(u(t)) = -\\operatorname{tr}\\frac{\\partial \\mathrm{vf}}{\\partial u}
\\qquad\\Longrightarrow\\qquad
\\log p(u_1) = \\log p(u_0) - \\underbrace{\\int_{t_0}^{t_1}\\!\\operatorname{tr}\\,\\partial_u \\mathrm{vf}\\,dt}_{\\texttt{Δlogdet}}
```

This is the instantaneous change-of-variables of Chen et al. / FFJORD, and it is what turns a
NeuralODE from a map on *points* into a map on *densities* — i.e. what would let this factor
push a real belief rather than a Dirac.

> [!warning] The trace is computed by a dense finite-difference Jacobian
> ``O(n)`` evaluations of `vf` per step, so ``O(n \\cdot \\mathrm{steps})`` in total. FFJORD's
> whole contribution is the Hutchinson estimator that avoids this; see `flow.md` §4.2. What is
> here is correct, checkable against a linear system, and unusable above small `n`.
"""
function integrate_with_divergence(vf, u₀, t₀, t₁, itg::AbstractIntegrator)
    n = nsteps(itg)
    dt = (t₁ - t₀) / n
    u = copy(u₀)
    t = t₀
    acc = 0.0
    for _ in 1:n
        # trapezoidal in the divergence, Euler/RK4 in the state
        tr0 = _divergence(vf, u, t)
        unew = integrate(vf, u, t, t + dt, _one_step(itg))
        tr1 = _divergence(vf, unew, t + dt)
        acc += dt * (tr0 + tr1) / 2
        u = unew
        t += dt
    end
    return (u, acc)
end

_one_step(::EulerIntegrator) = EulerIntegrator(1)
_one_step(::RK4Integrator) = RK4Integrator(1)

"""
    _divergence(vf, u, t) -> Real

``\\operatorname{tr}(\\partial \\mathrm{vf}/\\partial u)`` by forward differences.
"""
function _divergence(vf, u, t; ε = 1e-7)
    f0 = vf(u, t)
    acc = 0.0
    for j in eachindex(u)
        h = ε * max(one(eltype(u)), abs(u[j]))
        up = copy(u)
        up[j] += h
        acc += (vf(up, t)[j] - f0[j]) / h
    end
    return acc
end
