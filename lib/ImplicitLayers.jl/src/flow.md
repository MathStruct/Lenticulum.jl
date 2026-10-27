#implementation

> Numerical flows in either time direction: built-in fixed-step Euler/RK4 and an optional
> OrdinaryDiffEq extension. Time reversal approximates the continuous inverse; only symmetric
> methods on matching grids invert their discrete forward map.

> Sources: Rackauckas et al. (2020), *Universal Differential Equations for Scientific Machine Learning* (SciMLSensitivity software citation);  Hairer, Lubich, Wanner, *Geometric Numerical Integration*, 2nd ed., II.3 (adjoint and symmetric methods); [OrdinaryDiffEqSDIRK solver documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/implicit/SDIRK/); code: `flow.jl`, `../ext/ImplicitLayersOrdinaryDiffEqExt.jl`, `../ext/ImplicitLayersOrdinaryDiffEqSDIRKExt.jl`, `../examples/flow_roundtrip.jl`, `../test/roundtrip.jl`, `../test/sciml.jl`
>
> Bibliography: [[Bibliography#^rackauckas2020universal|Rackauckas et al. 2020]] · [[Bibliography#^hairer2006geometric|Hairer et al. 2006]]
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

For nonempty Lux state, the `vectorfield` closure updates its `Ref` on every RHS call,
including when SciMLSensitivity is loaded. Rejected adaptive
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

### 4.4 Optional endpoint sensitivities

Load SciMLSensitivity alongside the solver packages to activate
`ImplicitLayersSciMLSensitivityExt`. `SciMLIntegrator(alg; sensealg=nothing,
kwargs...)` keeps sensitivity selection separate from solver keywords; it does
not change `inversion_kind` or time reversal. The default resolves to
`ForwardDiffSensitivity()`: the [SciMLSensitivity manual](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#Choosing-a-Sensitivity-Algorithm)
recommends it for small parameter counts (under roughly 100), using direct AD
through a native Julia solver. It can supply a pullback to an outer reverse AD.
For a continuous adjoint, choose one explicitly:

```julia
using SciMLSensitivity, OrdinaryDiffEqTsit5
itg = SciMLIntegrator(Tsit5(); reltol=1e-10, abstol=1e-12,
    sensealg=GaussAdjoint(autojacvec=ZygoteVJP()))
```

The extension constructs an out-of-place ODE with a flattened Lux parameter
vector as `ODEProblem.p`. Its RHS captures only a zero-valued shape template,
model, offsets and empty state; the learned values come exclusively from `p`.
The sensitivity path requires recursively empty tuples/named tuples for incoming
and returned state. With nonempty incoming state, `_flow` invokes the generic
`AbstractIntegrator` method and preserves legacy evaluation-state threading for
`flow_forward`, `flow_reverse` and inference, even with an explicit `sensealg`.
`flow_logdet` retains its existing evaluation-state path too.

AD with nonempty state is rejected before tracing at `energy_jacobian`,
`energy_gradient`, `free_energy_gradient` and Mycelium's graph gradient and
`parameter_step`. Their `_check_factor_ad(ad, f, st)` hook knows both the backend
and incoming state; `ad=nothing` keeps reference differences available. Adaptive
stages and rejected trials can change state, so this path does not define supported
ODE sensitivities. BatchNorm statistics and other nonempty state remain unsupported
for AD; there is no opt-in static-state promise yet. A layer that starts with empty
state but returns nonempty state is rejected by the sensitivity RHS.
An arbitrary outer DI call bypasses the factor/graph entry checks, just as it does
for Enzyme runtime activity; callers must enforce the empty-state restriction
themselves. Built-in Euler/RK4 keep their existing AD state threading.

`test/lp002_followups.jl` checks Tsit5 endpoint parameter Jacobians, scalar energy
and free-energy gradients with AutoForwardDiff, AutoZygote and AutoMooncake;
Enzyme is explicitly rejected at factor/graph derivative entry points.
`test/sciml_state.jl` runs the same CNF, CountingDynamics, reverse, inversion-label
and inference assertions before and after SciMLSensitivity loads. The follow-up
checks also verify that stateful factor/graph AD throws before evaluating the RHS,
including nested nonempty state. Tests cover
the default plus `InterpolatingAdjoint` and `GaussAdjoint` with `ZygoteVJP()`.
ForwardDiff uses dual-number solves, independently of the selected `sensealg`.
The scalar oracle is $D_a z(T)=Tz_0e^{aT}$ on $T=0.7$, with solver
`abstol=1e-12,reltol=1e-10` and derivative absolute tolerance `2e-8` to allow
accumulated primal/adjoint integration error. A two-dimensional tanh Lux model
is compared with central differences at `3e-7` absolute tolerance. Graph learning
fits the continuous target $e^{0.4}$ from $a=-0.2$ in 100 gradient steps of size
0.2, using the default sensitivity algorithm for all three backends.

Continuous adjoints may store/reconstruct a forward trajectory internally even
though this wrapper returns only the endpoint. The public integration storage
invariants still apply. BacksolveAdjoint can amplify reverse-conditioning error;
SciML does not remove that issue. AD through `flow_logdet`, callbacks, GPU,
non-Tsit5 sensitivities and higher derivatives have no new coverage.

Related: [[neuralode]], [[solve]], [[NeuralODE as an Invertible Factor]],
[[The Equilibrium Family]]
