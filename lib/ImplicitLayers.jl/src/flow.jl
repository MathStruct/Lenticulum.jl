# ---------------------------------------------------------------------------
# Numerical flows, forwards and backwards. Reversing an explicit scheme approximates the
# continuous inverse; it does not invert the discrete forward map, even in exact arithmetic.
#
# See `flow.md`.
# ---------------------------------------------------------------------------

"""
    abstract type AbstractIntegrator

A numerical ODE scheme. Built-in Euler/RK4 are fixed-step explicit methods; optional
[`SciMLIntegrator`](@ref) supports adaptive and implicit solvers.
"""
abstract type AbstractIntegrator end

"""
    EulerIntegrator(steps = 50)

Explicit Euler. First order, and present mainly as the thing `RK4Integrator` is checked
against — its error is visible enough to make the order of accuracy testable.
"""
struct EulerIntegrator <: AbstractIntegrator
    steps::Int
    function EulerIntegrator(steps)
        n = Int(steps)
        n > 0 || throw(ArgumentError("steps must be positive; got $steps"))
        return new(n)
    end
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
    function RK4Integrator(steps)
        n = Int(steps)
        n > 0 || throw(ArgumentError("steps must be positive; got $steps"))
        return new(n)
    end
end
RK4Integrator(; steps = 50) = RK4Integrator(steps)

nsteps(i::AbstractIntegrator) = i.steps

"""
    SciMLIntegrator(alg; kwargs...)

Optional OrdinaryDiffEq solver, loaded by `using OrdinaryDiffEqCore` and a solver package
(e.g. `OrdinaryDiffEqTsit5` or `OrdinaryDiffEqSDIRK`). Keywords go to `solve`; only the final
state is retained. `dt` is a positive magnitude, signed automatically for reverse solves.

Fixed-step `ImplicitMidpoint`/`Trapezoid` can invert their discrete map up to nonlinear-solve
tolerance; adaptive solvers such as `Tsit5` only control integration error. See
[`inversion_kind`](@ref) for the conservative exactness conditions. This wrapper does not
provide SciMLSensitivity gradients (LP-002).
"""
struct SciMLIntegrator{A,K<:NamedTuple} <: AbstractIntegrator
    alg::A
    kwargs::K
end
function SciMLIntegrator(alg; kwargs...)
    opts = (; kwargs...)
    if haskey(opts, :dt)
        dt = opts.dt
        (dt isa Real && isfinite(dt) && dt > 0) ||
            throw(ArgumentError("dt must be a finite positive magnitude"))
    end
    return SciMLIntegrator(alg, opts)
end

"""
    isreversible(integrator) -> Bool

Whether the configured numerical method is symmetric under time reversal, assuming a
deterministic vector field and converged implicit stages. Euler/RK4 and adaptive solvers
return false. Symmetry alone does not ensure the two solves use matching time grids.
"""
isreversible(::AbstractIntegrator) = false
_symmetric_algorithm(alg) = false
function isreversible(itg::SciMLIntegrator)
    opts = itg.kwargs
    # Callbacks, tstops, limiters and other step-changing options need their own proof.
    allowed = (:dt, :adaptive, :abstol, :reltol, :maxiters)
    return _symmetric_algorithm(itg.alg) && get(opts, :adaptive, true) === false &&
           haskey(opts, :dt) && all(k -> k in allowed, keys(opts))
end

"""
    inversion_kind(integrator, t₀, t₁)

`SolverInversion(integrator)` by default. `ExactInversion()` is reserved for a symmetric
fixed-step method whose positive `dt` divides the interval (within floating-point roundoff).
This is exactness of the discrete map, up to roundoff and nonlinear-solve tolerance, not
accuracy against the continuous solution. It assumes dynamics do not change with Lux state.
"""
inversion_kind(itg::AbstractIntegrator, t₀, t₁) = LenticulumCore.SolverInversion(itg)
function inversion_kind(itg::SciMLIntegrator, t₀, t₁)
    if isreversible(itg)
        n = abs(t₁ - t₀) / itg.kwargs.dt
        if isfinite(n) && n >= 1 && isapprox(n, round(n); rtol = 8eps(one(float(n))), atol = 0)
            return LenticulumCore.ExactInversion()
        end
    end
    return LenticulumCore.SolverInversion(itg)
end

_sciml_integrate(args...) = throw(ArgumentError(
    "SciMLIntegrator requires the OrdinaryDiffEq extension: load OrdinaryDiffEqCore " *
    "and a solver package, e.g. `using OrdinaryDiffEqTsit5` or `using OrdinaryDiffEqSDIRK`"))
integrate(vf, u₀, t₀, t₁, itg::SciMLIntegrator) =
    _sciml_integrate(vf, u₀, t₀, t₁, itg)
_sciml_divergence(args...) = _sciml_integrate(args...)
integrate_with_divergence(vf, u₀, t₀, t₁, itg::SciMLIntegrator) =
    _sciml_divergence(vf, u₀, t₀, t₁, itg)

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
> ``O(n)`` extra evaluations of `vf` per divergence evaluation. Built-in methods use
> trapezoidal quadrature; SciML integrates an augmented state. FFJORD's Hutchinson estimator
> avoids the dense trace; see `flow.md` §4.1. What is
> here is correct, checkable against a linear system, and unusable above small `n`.
"""
function integrate_with_divergence(vf, u₀, t₀, t₁, itg::AbstractIntegrator)
    n = nsteps(itg)
    dt = (t₁ - t₀) / n
    u = copy(u₀)
    t = t₀
    acc = 0.0
    tr0 = _divergence(vf, u, t)
    for k in 1:n
        # trapezoidal in the divergence, Euler/RK4 in the state
        unew = integrate(vf, u, t, t + dt, _one_step(itg))
        t_next = t₀ + k * dt
        tr1 = _divergence(vf, unew, t_next)
        acc += dt * (tr0 + tr1) / 2
        u = unew
        t = t_next
        tr0 = tr1
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
