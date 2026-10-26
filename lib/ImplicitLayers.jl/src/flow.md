#implementation

> Numerical flows in either time direction: built-in fixed-step Euler/RK4 and an optional
> OrdinaryDiffEq extension. Time reversal approximates the continuous inverse; only symmetric
> methods on matching grids invert their discrete forward map.

> Sources: Hairer, Lubich, Wanner, *Geometric Numerical Integration*, 2nd ed., II.3 (adjoint and symmetric methods); [OrdinaryDiffEqSDIRK solver documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/implicit/SDIRK/); code: `flow.jl`, `../ext/ImplicitLayersOrdinaryDiffEqExt.jl`, `../ext/ImplicitLayersOrdinaryDiffEqSDIRKExt.jl`, `../examples/flow_roundtrip.jl`, `../test/roundtrip.jl`, `../test/sciml.jl`
>
> Bibliography: [[Bibliography#^hairer2006geometric|Hairer et al. 2006]]
>
> Theory (CT-ML wiki): [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category)

## 1. Continuous and discrete inverses

Where an ODE has a unique solution throughout the interval in both directions, its continuous
flow is a diffeomorphism. A numerical scheme approximates that flow. Running Euler or RK4
with negative steps does **not** invert the discrete forward map, even in exact arithmetic.

For $z'=az$, Euler multiplies by $1+ah$ forwards and $1-ah$ backwards, rather than the discrete
inverse $(1+ah)^{-1}$. Its round trip is $O(h)$. RK4 has fourth-order forward global error
and an $O(h^4)$ round-trip bound under smoothness and bounded-amplification assumptions;
the scalar linear round trip has the sharper $O(h^5)$ order because
$R(x)R(-x)=1+x^6/72+x^8/576$. Thus its round-trip ratio tends to 32 on halving $h$, while its
forward-error ratio tends to 16. [[neuralode]] §4.3 contains the derivation, measured table
and reproduction command. `test/roundtrip.jl` checks both orders and the one-step formulas.

`RK4Integrator(steps=50)` remains the default. Both built-ins reject nonpositive step counts;
their numerical algorithms and divergence quadrature are unchanged. They receive
`SolverInversion(integrator)`, using the core's existing nonexact solver label.

## 2. Optional SciML solvers and reversibility

There is no hard SciML dependency. `SciMLIntegrator(alg; kwargs...)` lives in the base package;
its integration implementation is `ext/ImplicitLayersOrdinaryDiffEqExt.jl`, activated by
`OrdinaryDiffEqCore` and `SciMLBase`. The separate SDIRK extension supplies the symmetry trait.
The solver subpackages bring Core and SciMLBase transitively; installing the full
`OrdinaryDiffEq` umbrella is unnecessary. The test environment resolves Core 4.19.0,
Tsit5 2.1.5 and SDIRK 2.9.8 (Julia 1.13.1); the supported Core/SDIRK major-version ranges
also allow the earlier split ecosystem. The tests verify the versions resolved here.

```julia
using ImplicitLayers, OrdinaryDiffEqCore, OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK
adaptive = SciMLIntegrator(Tsit5(); reltol=1e-9, abstol=1e-12)
symmetric = SciMLIntegrator(ImplicitMidpoint(); adaptive=false, dt=0.02,
                           reltol=1e-12, abstol=1e-12)
u1 = integrate((u,t) -> -5 .* u, [3.0], 0.0, 1.0, symmetric)
u0 = integrate((u,t) -> -5 .* u, u1, 1.0, 0.0, symmetric)
```

Each call constructs an out-of-place `ODEProblem`, solves over `(t₀,t₁)` (reversal allowed),
and returns only the final state. Positive `dt` is a magnitude; its sign follows the time
direction. Storage is forced to `save_everystep=false`, `save_start=false`, `save_end=true`,
empty `saveat`, full-state `save_idxs=nothing` and `dense=false`. Failed or early-terminated solves throw instead of returning
an incomplete endpoint. Without the extension, both state and divergence solves give an
error naming the packages to load. Arbitrary solver keywords otherwise pass through.

`isreversible` defaults to false. Fixed-step `ImplicitMidpoint` and `Trapezoid` implement
symmetric one-step maps (Hairer–Lubich–Wanner, II.3):

$$
\Phi_{-h} = \Phi_h^{-1}.
$$

`inversion_kind(integrator,t₀,t₁)` returns `ExactInversion` only when `adaptive=false`, a
positive `dt` divides the interval within roundoff, the step limiter is the default, and
solve options are limited to `dt`, `adaptive`, `abstol`, `reltol`, `maxiters`. The interval
check prevents a shortened final step from changing the grid under reversal, which matters
for nonautonomous dynamics. Other options conservatively retain `SolverInversion`.
`assemble` also requires empty Lux state before using the exact label.

This exactness is **up to converged nonlinear stages and roundoff** and concerns the discrete
map, not the continuous solution. In particular, midpoint still has second-order global
truncation error. `test/sciml.jl` checks both scalar linear rates $a=0.5,-5$ with `dt=0.02`
and `abstol=reltol=1e-12` (default `NLNewton` scaled convergence factor $\kappa=0.01$),
requiring relative residual below $10^{-10}$ to allow accumulated stage and roundoff errors.
It also checks nonlinear time-dependent dynamics. Tsit5 receives `SolverInversion`; tests
measure its residual across `reltol=10^{-3},10^{-6},10^{-9}`, with `abstol=10^{-3}reltol`.
Tolerance controls local integration error, not a strict global round-trip bound.


`test/sciml.jl` (Julia 1.13.1, versions above) measured the following endpoint residuals,
starting from endpoint 3 and reversing then forwarding on $[0,1]$:

| algorithm/configuration | $a=0.5$ | $a=-5$ |
|---|---:|---:|
| midpoint, fixed $dt=0.02$, tolerances $10^{-12}$ | 0.0 | 8.881784e-16 |
| trapezoid, same configuration | 7.401487e-16 | 8.881784e-16 |
| Tsit5, `reltol=1e-3` | 2.720915e-8 | 5.005610e-5 |
| Tsit5, `reltol=1e-6` | 4.980462e-10 | 1.270538e-7 |
| Tsit5, `reltol=1e-9` | 5.648075e-12 | 1.649269e-10 |

Zero in this run means cancellation to Float64 precision, not a promise of a zero residual
for other problems. The Tsit5 test checks decreasing residuals with logarithmic slopes
between 0.5 and 1.5 over each three-decade refinement, allowing non-asymptotic behaviour.

## 3. Continuous normalising flows

`integrate_with_divergence` computes the instantaneous continuous change of variables:

$$
\frac{d}{dt}\log p(u(t))=-\operatorname{tr}\partial_u f,
\qquad
\log p(u_1)=\log p(u_0)-\underbrace{\int_{t_0}^{t_1}\operatorname{tr}\partial_u f\,dt}_{\Delta\log\det}.
$$

The built-in methods retain their existing trapezoidal divergence quadrature. For SciML,
`[u; logdet]` is integrated as an augmented state with RHS `[vf(u,t); divergence(vf,u,t)]`.
Both directions are checked against $u_1=e^{A(t_1-t_0)}u_0$ and
$\Delta\log\det=\operatorname{tr}(A)(t_1-t_0)$. This is a continuous-flow correction, not the
exact determinant of an arbitrary discrete numerical map. It does not yet transport a
non-Gaussian belief through the factor.

## 4. Difficulties and limits

### 4.1 Dense finite-difference divergence

The divergence uses $n$ forward-difference probes per evaluation, with perturbation
$10^{-7}\max(1,|u_j|)$. This is a small-system tool. The SciML augmented solve applies its
error controller to the accumulator too, but finite-difference noise limits useful tolerance.
The linear test has constant trace and therefore does not establish nonlinear quadrature
accuracy. Built-in logdet quadrature is second-order even when the state uses RK4.

### 4.2 Lux state is evaluation state

The existing `vectorfield` closure updates its `Ref` on every RHS call. Rejected adaptive
trials, nonlinear stage iterations, Jacobian evaluations and divergence probes all update it.
Tests count the calls independently and verify the returned state. If these updates change
the vector field, the result need not represent a deterministic ODE or a reversible flow;
`assemble` conservatively gives nonempty state a nonexact inversion label.

### 4.3 Stiffness and conditioning

Built-in explicit integrators have no error estimate or failure report and can diverge on
coarse or stiff problems. The optional extension offers adaptive and implicit algorithms and
checks solver completion, but cannot repair inverse conditioning. A contracting scalar flow
with $a<0$ amplifies absolute endpoint errors by $e^{|a|T}$ when inverted. This is distinct
from discretisation and persists even with a symmetric method. See [[neuralode]] §4.3.

### 4.4 Gradients remain LP-002

This extension adds integration, not SciMLSensitivity gradients. That is
[[LP-002 The Jacobian side is not implemented]]. SciMLSensitivity's
[`BacksolveAdjoint` documentation](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#SciMLSensitivity.BacksolveAdjoint)
explicitly describes backward re-integration, warns about unstable reverse solutions, and
offers checkpointing to stabilize them. The same reverse-conditioning concern therefore
applies; using SciML does not remove it.

Related: [[neuralode]], [[solve]], [[NeuralODE as an Invertible Factor]],
[[The Equilibrium Family]]
