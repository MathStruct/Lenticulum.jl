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

Under the default `:threaded` policy with nonempty Lux state, the `vectorfield`
closure updates its `Ref` on every RHS call,
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

The extension constructs an out-of-place ODE with flattened Lux parameters as
`ODEProblem.p`. Its RHS captures a zero-valued shape template, model, offsets and
constant incoming state. Learned values come exclusively from `p`.

The default `NeuralODEFactor(...; state_policy=:threaded)` is unchanged: Euler/RK4
advance state at every stage, and nonempty-state SciML uses legacy evaluation-state
threading, including rejected trials. Its SciML AD path still requires recursively
empty incoming and returned state. Reference differences remain available.

Opt into `NeuralODEFactor(...; state_policy=:fixed)` for stateful sensitivities.
Every RHS call receives the same incoming state and discards the returned state.
After the successful solve, evaluate the dynamics once at the final endpoint and
final time with incoming state to obtain the returned state. The same policy is
available for Euler/RK4. This is a different equation from stage threading when
state affects the field; switching policy is always explicit. Layers must treat
incoming state functionally: in-place mutation of captured state is unsupported.
State itself is a constant context, not a differentiated coordinate. `flow_logdet`
rejects this policy; its divergence/state semantics need separate work.

`test/lp002_batch_a.jl` compares affine, state-dependent flows in both directions
against closed forms and fixed-state RK4. Tsit5, Vern7 and SDIRK Kvaerno3/KenCarp4 are
covered with default `ForwardDiffSensitivity`, `InterpolatingAdjoint` and
`GaussAdjoint` (`ZygoteVJP`), using outer ForwardDiff, Zygote and Mooncake.
ForwardDiff uses dual solves independently of the selected sensealg. A fast
scalar mode with coefficient -50 additionally exercises Kvaerno3.

Kvaerno3 with InterpolatingAdjoint fails the affine closed-form reverse gradient
on this stack despite tight solve tolerances. Factor/graph AD entry points reject
that pair for reverse backends with an `ArgumentError`; direct ForwardDiff ignores
the sensealg and remains supported. KenCarp4 with InterpolatingAdjoint and
Kvaerno3 with GaussAdjoint pass the same oracle. Tightening Newton convergence,
using finite-difference Newton Jacobians, and reducing `dtmax` did not repair the
failed pair in diagnostic runs. `examples/lp002_sciml_adjoint_limit.jl` bypasses
validation to reproduce it; arbitrary outer DI callers must enforce this limit.
The typed guard lives in the optional SDIRK/SciMLSensitivity/DI extension and
preserves all primal calls. This is a measured dependency-stack limit, not a
claim that all implicit solver adjoints fail.

Runtime-activity Enzyme **reverse** is tested on Tsit5's default endpoint
sensitivity path, including fixed state and scalar loss gradients with respect
to inputs and parameters. Vector-return split-reverse Jacobians fail in Enzyme’s
ABI wrapper on this stack; `energy_jacobian` rejects that configuration clearly. Plain
`AutoEnzyme()` and forward modes are rejected by factor/graph entry points with
a configuration hint. Direct runtime forward AD fails in OrdinaryDiffEq callback
initialization on the tested stack, before the first RHS evaluation; see
`examples/lp002_enzyme_limits.jl`. EnzymeVJP (the inner RHS VJP choice) is distinct
from the outer backend. No Enzyme support is claimed for other solver/sensealg
combinations. Arbitrary outer DI calls bypass package validation.

For fixed-state explicit sensitivities, and explicit sensitivities with decreasing
time, the wrapper retains a dense forward trajectory internally. SciML’s
endpoint-only checkpoint reconstruction failed in reverse time during batch A;
retaining the trajectory avoids that failure. The public return is still only
the endpoint. Default sensitivity options and primal default results are unchanged.

Continuous adjoints may store/reconstruct a forward trajectory internally even
though this wrapper returns only the endpoint. The public integration storage
invariants still apply. BacksolveAdjoint can amplify reverse-conditioning error;
SciML does not remove that issue. AD through `flow_logdet`, callbacks, GPU,
higher derivatives have no new coverage.

Related: [[neuralode]], [[solve]], [[NeuralODE as an Invertible Factor]],
[[The Equilibrium Family]]


## Device explicit flows (2026-10-27)

Euler/RK4 endpoint tests in `test/device_arrays.jl` use JLArrays with scalar indexing
disabled, both precisions and integration directions. For a linear field they compare
with the exact *discrete* polynomial map raised to the number of steps, and compare
input and parameter gradients with that map. These tests exercise both Lux state
policies, with empty state and functional state updates compared with CPU. Device times/step coefficients use the state element type,
so a Float64 default tspan cannot widen Float32 stages. CPU times/arithmetic are
unchanged. This batch adds no SciML device, divergence/logdet or hardware GPU coverage.


## LP-002 batch C: what “AD through divergence” means (2026-10-28)

The Remaining item first appears in LP-002 commit `6e0bbbb6` (2026-10-26).
It concerns a loss depending on the returned density correction from
`integrate_with_divergence` / `flow_logdet`, not the already tested endpoint-only
residual. For a fixed deterministic vector field, the new derivative would include
both explicit parameter dependence of $\operatorname{tr}D_u f_\theta$ and its
trajectory dependence through $u(t)$. It therefore needs mixed derivatives of the
vector field. Differentiating only the endpoint or detaching the divergence is a
different objective.

The default `_divergence` is a mutable forward-difference loop with
$h_j=10^{-7}\max(1,|u_j|)$. AD of this program, when it works, differentiates the
**finite probe computation**, including the piecewise step-size formula; it does
not replace it with an exact analytic trace. Built-in integrators add their
existing trapezoidal trace quadrature. SciML's augmented-state implementation uses
its separate captured-vector-field RHS; the endpoint sensitivity extension does
not supply a parameter-aware sensitivity rule for this density correction.

`examples/lp002_metric_divergence_limits.jl` separates the raw trace from the
integrated correction. Its nonlinear field $f_\theta(u)=a u^2$ has analytic trace
$2a\sum_j u_j$; the linear field $f_\theta(u)=a u$ has correction $naT$. These
closed forms distinguish probe error from an AD/backend failure. The dated LP-002
batch C section records the actual probe results. These small deterministic CPU
probes do not establish general CNF sensitivity support.

Functional forward-difference probes or an analytic/backend trace would need a
separate opt-in contract, validation of mixed derivatives, and a state policy.
With default nonempty state, every base/probe/stage evaluation advances the Lux
state; the result is an evaluation-history computation, not a fixed vector field.
`state_policy=:fixed` still rejects `flow_logdet`. Stateful, adaptive-SciML,
device and general native-backend divergence sensitivities remain open. No default
trace, quadrature, sensitivity or state-threading algorithm is changed here.
