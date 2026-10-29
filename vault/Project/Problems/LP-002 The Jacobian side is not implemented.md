#open-problem

> **Status: batches A/B retained; batch C adds realised flooding/residual/damped replay and opt-in CPU message fixed-point adjoints (2026-10-28). Final batch C verification is recorded below. Native replay backends, device implicit coordinates and general divergence/metric sensitivities remain limited; full scope is in Remaining.** Related: [[energy]] §"Implementation difficulties" 6, [[Scalar and Multivariate Energy]] §6, [[L-ADR-0002 Energies are vectors, scalarised late]], [Fisher information metric (CT-ML wiki)](https://mathstruct.org/CategoryTheory-ML-Wiki/Fisher-Information-Metric).

> Sources: original implementation; Montoison and Orban (2023), *Krylov.jl*, p. 1; Rackauckas et al. (2020), *Universal Differential Equations for Scientific Machine Learning* (SciMLSensitivity software citation); code: `lib/LenticulumCore.jl/src/energy.jl`, `lib/LenticulumCore.jl/test/parameter_ad.jl`, `lib/ImplicitLayers.jl/test/parameter_ad.jl`, `lib/ImplicitLayers.jl/test/lp002_followups.jl`, `lib/ImplicitLayers.jl/test/solver_numerics.jl`, `lib/Mycelium.jl/src/free_energy.jl`, `lib/ImplicitLayers.jl/test/device_arrays.jl`
>
> Bibliography: [[Bibliography#^montoison2023krylov|Montoison & Orban 2023]] · [[Bibliography#^rackauckas2020universal|Rackauckas et al. 2020]]
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

**Problem.** The vector energy exists so that its Jacobian is available: Gauss–Newton steps, the Fisher metric of AutoBayes Definition 27, and the implicit function theorem. `energy.jl` provides the values and the algebra, but nothing extracts $D_\theta \mathbf F$. Until it does, the main payoff of [[L-ADR-0002 Energies are vectors, scalarised late]] is unrealised.

**Options.**
1. A Jacobian function in `Lenticulum.jl` proper, with one package extension per AD backend; `LenticulumCore` stays free of AD dependencies.
2. Per-factor hand-written Jacobians where a closed form exists (linear-Gaussian, constraint), AD elsewhere.

**Leaning.** Both: closed forms as tests, AD extensions as the general path. The compositional Fisher metric of [Karwowski, Nielsen & Staton](https://mathstruct.org/CategoryTheory-ML-Wiki/Information-Geometry-of-Probabilistic-Programs) is a reference for how the pieces should combine.

**Progress (`review-argon`, 2026-10-23).** `LenticulumCore` now provides `energy_jacobian(f, x, a, y, ps, st; wrt)` (finite-difference reference on the pointwise vector energy) and `gauss_newton_metric(f, x, a, y, ps, st; wrt)` ($J^\top M J$ for `SquaredNorm(M)`), keeping `LenticulumCore` free of AD dependencies. Parameter-space AD extensions ($D_\theta \mathbf F$) and closed-form factor overrides remain open.

## Resolution (2026-10-26)

The maintainer requires automatic differentiation for learnable implicit factors.
`LenticulumCore` extends the existing `energy_jacobian` with `wrt=:ps` and `ad`, adds
`energy_gradient`/`free_energy_gradient` returning Lux-shaped parameter gradients,
and extends `gauss_newton_metric` to parameters. No hard AD dependency: DI is a weak
extension, backends are user-supplied ADTypes objects, and absent backends keep the
central-difference reference. Shared numeric tree flatten/rebuild helpers now also
serve VariationalDiffusion.

Mycelium's `parameter_step` differentiates the sum of local scalar free energies at
fixed messages, then applies its existing update-rule lens, including Nesterov
look-ahead. Variable entropy corrections are constant under this derivative.
`OptimiserFactor` previously required externally supplied gradients; inference
schedules supply no parameter derivative. The new driver stays separate from them.

DEQ `solve_state(...; ad)` uses IFT derivatives, a dense state linear solve by default (opt-in `KrylovIFT` uses GMRES), backend
JVPs in ForwardDiff and a backend VJP in the ChainRules reverse rule. It never
backpropagates through Broyden/Picard iterations. Root state is held fixed through
iterations and advanced once at the root. NeuralODE Euler/RK4 use functional stage
state threading and discretise-then-differentiate. `LuxFactor(...; residual=true)`
opts into endpoint scoring while preserving the default zero-energy wrapper.

## Evidence and limits

- `lib/LenticulumCore.jl/test/parameter_ad.jl`: nested/scalar/empty tree Jacobians,
  scalar gradients, parameter metrics and fixed state, with ForwardDiff, Zygote and FD.
- `test/runtests.jl`, Gaussian parameter derivative test: analytic residual Jacobian,
  precision-weighted gradient and metric, using the FD reference.
- `lib/ImplicitLayers.jl/test/parameter_ad.jl`: linear-cell solved-state parameter/input
  derivatives against $(I-W)^{-1}(\delta Wz^*+\delta Ux+\delta b)$ and FD; a nonzero
  downstream solve loss; discrete linear Euler/RK4 derivatives; stateful stages; Lux
  scoring; graph convergence to a generating ODE coefficient; Nesterov look-ahead.
  Native Enzyme forward/reverse rules and a Mooncake `rrule!!` now cover DEQ solves,
  nonlinear IFT, active warm starts, fixed Lux state and DEQ graph parameter steps.
  Both use DI with ForwardDiff for residual derivatives; no root iterations are traced.
  Mooncake also covers Euler/RK4 and scored LuxFactor paths. Enzyme Euler/RK4 uses
  runtime-activity Reverse or Forward mode: both pass the discrete Jacobian/gradient,
  stateful-stage FD and graph learning checks. Factor derivative and graph training
  entry points reject a mode without runtime activity and give the correct configuration;
  no custom Enzyme flow rule intercepts differentiation. DEQ and scored LuxFactor
  checks still use plain `AutoEnzyme()`. See [[neuralode]].
  Outer Enzyme closures capturing arrays need constant contexts or
  an explicit constant function annotation; see [[solve]].

Remaining: forward-mode Enzyme Jacobian NaNs, an upstream Enzyme defect diagnosed on
2026-10-28 and reported as Enzyme.jl#3884 (section at the end); automatic annotations for arbitrary outer Enzyme closures (reproducer and
explicit annotation workaround below); Kvaerno3/InterpolatingAdjoint reverse
compatibility (guarded accuracy failure); Enzyme forward SciML endpoint AD and broader
solver/sensealg coverage for Enzyme; general AD through CNF divergence and finite-difference
Sampson metric localisation (precise interpretation and probes in batch C); scalable/Krylov
second-order rules, direct input/parameter loss dependence and arbitrary native backend
nesting; hardware GPU validation, dense device factorisation, SciML/device divergence and
native Enzyme/Mooncake device rules; native Enzyme/Mooncake graph replay, implicit graph
device coordinates/evolving message state/higher-order rules and stochastic solver inversion
estimators. Flooding/residual/damped finite replay and CPU Exact/Pathwise implicit graph
adjoints are implemented in batch C; terminal estimators remain as specified in LP-011.
Default state-threaded SciML dynamics with nonempty state still require reference
differences: opt-in fixed-state sensitivities solve a different, explicitly specified equation.

## Runtime-activity flow follow-up validation (2026-10-26)

Julia 1.13.1, CPU. The final runs use
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'`; counts sum the top-level
test summaries. The existing DEQ/Mooncake rules are retained.

| Package | Passed | Pkg.test wall time |
|---|---:|---:|
| ImplicitLayers | 995 | 255.1 s |
| LenticulumCore | 316 | 46.8 s |
| Mycelium | 494 | 11.2 s |

Total: **1,805 passed**, no failures or errors. The implicit environment uses DI
0.7.21, ADTypes 1.24.0, Enzyme 0.13.216, EnzymeCore 0.8.26, Mooncake 0.5.65 and
ForwardDiff 1.4.6. Both runtime-activity Enzyme modes pass the Euler/RK4 closed-form
Jacobians, energy/free-energy gradients, stateful-stage FD comparisons and ODE graph
learning test. Non-runtime modes get the configuration hint; plain `AutoEnzyme()`
continues to pass DEQ and scored LuxFactor checks.

`docs/site/build.sh` passes with 246 HTML pages and no KaTeX errors.
`julia --project=docs docs/make.jl` passes in 435.9 s, including executed tutorials
and the vault, with size warnings for the core API page and search index. Both
builds set `QUARTZ_DIR` to a disposable copy with its checkout/clean EXIT hook removed;
the source checkout is untouched. `git diff --check` passes.

## Initial validation run (2026-10-26, before the native Enzyme/Mooncake rules)

Julia 1.13.1 on CPU. Each suite was run with
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'` (top level:
`--project=.`). Final passing runs, summing top-level test summaries:

| Package | Passed | Pkg.test wall time |
|---|---:|---:|
| LenticulumCore | 316 | 48.1 s |
| Mycelium | 494 | 10.9 s |
| Lenticulum (top-level) | 779 | 60.5 s |
| ImplicitLayers | 857 | 120.4 s |
| VariationalDiffusion | 1,143 | 420.3 s |
| Adversarial | 254 | 7.2 s |
| AlgebraicFactors | 14,399 | 159.0 s |

Total: **18,242 passed**, no failures, errors or broken tests.

Core's AD test environment resolved DifferentiationInterface 0.7.21, ADTypes 1.24.0,
ForwardDiff 1.4.6, Zygote 0.7.14 and Enzyme 0.13.216. The implicit suite tests
ForwardDiff/Zygote (same versions). Enzyme core parameter Jacobians, scalar gradients,
Float32 trees, graded scalarisation and inactive parameters pass. Coordinate dispatch
before tracing avoids Enzyme's static activity conflict between constant and active
arrays. Enzyme/Mooncake solved-state backends are rejected until they have IFT rules;
Mooncake has no new coverage. VD's existing backend tests pass independently.

The graph learning oracle uses Euler with five steps, generates the endpoint with
`a=0.4`, and fits from `a=-0.2` using 100 steps of gradient descent at rate 0.2.
Both tested graph backends reach parameter error below `1e-8` and reduce free energy
below `1e-12` of its starting value. These are assertions in the test, not fitted
reference outputs. DEQ warm starts carry no derivative; fixed state and singular-root
failure are also tested. Gaussian Dirac-endpoint free-energy gradients agree with
the fixed-precision analytic gradients; the precision path avoids a mutating
inverse-of-Cholesky operation that the tested Zygote stack cannot differentiate.

Long runs include VariationalDiffusion (420.3 s total) and AlgebraicFactors (159.0 s,
including 61.2 s for homotopy branches). ImplicitLayers takes 120.4 s; its new DEQ
and discrete ODE testsets take 23.9 s and 17.5 s respectively, including compilation.

Both documentation commands are used: `docs/site/build.sh` and
`julia --project=docs docs/make.jl`. Local builds set `QUARTZ_DIR` to a disposable
copy whose git cleanup EXIT hook was removed, and `npm_config_cache` to a writable
temporary cache. Setting `QUARTZ_DIR` alone only bypasses the outer checkout; the
engine's own helper also has a checkout/clean hook. No new citation was needed;
the existing BacksolveAdjoint documentation statement was checked against the
SciMLSensitivity manual on 2026-10-26.

Both final builds passed (Node 26.12.1). The standalone vault build rendered 246
HTML pages with no KaTeX errors. Documenter executed the tutorials and reported only
size-threshold warnings for the core API page and search index; deployment was skipped
in the local environment. `git diff --check` passed.

## Files changed in the initial LP-002 implementation

- `docs/src/packages/implicitlayers.md`
- `docs/src/packages/lenticulumcore.md`
- `docs/src/packages/mycelium.md`
- `lib/AlgebraicFactors.jl/src/polynomial.jl`
- `lib/AlgebraicFactors.jl/src/polynomial.md`
- `lib/AlgebraicFactors.jl/test/polynomial.jl`
- `lib/ImplicitLayers.jl/Project.toml`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersChainRulesCoreExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersDifferentiationInterfaceExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersForwardDiffExt.jl`
- `lib/ImplicitLayers.jl/src/ImplicitLayers.jl`
- `lib/ImplicitLayers.jl/src/ImplicitLayers.md`
- `lib/ImplicitLayers.jl/src/deq.jl`
- `lib/ImplicitLayers.jl/src/deq.md`
- `lib/ImplicitLayers.jl/src/flow.md`
- `lib/ImplicitLayers.jl/src/luxfactor.jl`
- `lib/ImplicitLayers.jl/src/luxfactor.md`
- `lib/ImplicitLayers.jl/src/neuralode.jl`
- `lib/ImplicitLayers.jl/src/neuralode.md`
- `lib/ImplicitLayers.jl/src/solve.jl`
- `lib/ImplicitLayers.jl/src/solve.md`
- `lib/ImplicitLayers.jl/test/Project.toml`
- `lib/ImplicitLayers.jl/test/parameter_ad.jl`
- `lib/ImplicitLayers.jl/test/runtests.jl`
- `lib/LenticulumCore.jl/Project.toml`
- `lib/LenticulumCore.jl/ext/LenticulumCoreDifferentiationInterfaceExt.jl`
- `lib/LenticulumCore.jl/src/LenticulumCore.jl`
- `lib/LenticulumCore.jl/src/energy.jl`
- `lib/LenticulumCore.jl/src/energy.md`
- `lib/LenticulumCore.jl/src/parameter_tree.jl`
- `lib/LenticulumCore.jl/src/parameter_tree.md`
- `lib/LenticulumCore.jl/test/Project.toml`
- `lib/LenticulumCore.jl/test/parameter_ad.jl`
- `lib/LenticulumCore.jl/test/runtests.jl`
- `lib/Mycelium.jl/src/Mycelium.jl`
- `lib/Mycelium.jl/src/free_energy.jl`
- `lib/Mycelium.jl/src/free_energy.md`
- `lib/VariationalDiffusion.jl/ext/VariationalDiffusionDifferentiationInterfaceExt.jl`
- `src/gaussian.jl`
- `src/gaussian.md`
- `test/runtests.jl`
- `vault/Project/Decisions/L-ADR-0002 Energies are vectors, scalarised late.md`
- `vault/Project/Problems/LP-002 The Jacobian side is not implemented.md`
- `vault/Project/Problems/Problem Log.md`
- `vault/Project/Specifications/L-SPEC-001 The Factor Interface.md`

## SciML sensitivities and matrix-free IFT validation (2026-10-26)

Julia 1.13.1, CPU. `lib/ImplicitLayers.jl/test/lp002_followups.jl` supplies
closed-form and finite-difference oracles for both follow-ups. Final full runs use
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'`; counts sum the
top-level test summaries, and times include test startup and compilation:

| Package | Passed | Pkg.test wall time |
|---|---:|---:|
| ImplicitLayers | 1,086 | 441.2 s |
| LenticulumCore | 316 | 46.0 s |
| Mycelium | 494 | 10.9 s |

Total: **1,896 passed**, no failures or errors. The 91 new assertions cover:

- **47 SciML endpoint checks:** Tsit5 with default `ForwardDiffSensitivity()`,
  explicit `InterpolatingAdjoint` and `GaussAdjoint` (both `ZygoteVJP()`), each
  with outer AutoForwardDiff, AutoZygote and AutoMooncake. For scalar
  $z'=az$, $a=-0.35$, $T=0.7$, $z_0=1.2$, the largest absolute error against
  $D_a z(T)=Tz_0e^{aT}$ was **4.912737e-13**. Solver tolerances are
  `abstol=1e-12,reltol=1e-10`; derivative assertions allow `2e-8` absolute error
  for accumulated integration error. A two-dimensional tanh Lux model agrees
  with central differences within `3e-7`. This initial run incorrectly asserted
  rejection of nonempty incoming state during primal evaluation; the review fix
  below replaces that regression assertion with primal success and AD rejection.
  Nested empty state is accepted, and the sensitivity RHS rejects a layer that
  introduces nonempty state. AutoEnzyme/SciML endpoint AD is
  explicitly rejected at factor/graph derivative entry points.
- **9 graph checks:** continuous target $e^{0.4}$, initial coefficient $-0.2$,
  100 gradient steps of size 0.2. Each of the three supported SciML backends
  reaches coefficient error below `1e-8` and free energy below `1e-12` of its
  initial value. These graph checks use the default sensitivity algorithm.
- **30 small DEQ checks:** GMRES agrees with dense/exact input and parameter
  derivatives, including a nonzero downstream scalar loss and forced Krylov
  nonconvergence. All four existing backends pass, including explicitly selected
  Enzyme forward/reverse modes. The native Krylov Enzyme VJPs enable runtime
  activity internally; the outer DEQ backend can remain plain `AutoEnzyme()`.
- **5 large DEQ checks:** a seeded random contractive 200-dimensional linear
  cell, Picard primal and restarted GMRES IFT. The dense state-Jacobian entry
  point throws if called. All four backends match $(I-W)^{-\top}z^*$; the largest
  relative gradient error was **9.183283e-11** at Krylov `rtol=1e-10,atol=1e-12`.

The implicit test environment resolved SciMLSensitivity 7.119.13, Krylov 0.10.11,
SciMLBase 3.58.0, OrdinaryDiffEqCore 4.19.0, Tsit5 2.1.5, DI 0.7.21,
ForwardDiff 1.4.6, Enzyme 0.13.216 and Mooncake 0.5.65. `DenseIFT()` remains the
default. Krylov's reported convergence is checked against an explicit final
residual; failure throws. Broyden's primal memory and the dense reference APIs
`deq_sensitivity`/`ift_sensitivity` are unchanged. SciML's flattened Lux parameters
are `ODEProblem.p`, with only a zero-valued shape template captured by the RHS.
Sensitivity selection is separate from the options used by `inversion_kind`.

The standalone vault build passes: 246 HTML pages, no KaTeX errors, 7.1 s wall
time. The API documentation build also passes in 89.0 s, using cached tutorial
renderings. Builds use `QUARTZ_DIR` pointing at a
disposable copy with its checkout/clean EXIT hook removed. Source `docs/quartz`
and `meta/` are untouched. Bibliographic metadata for the new software references
was checked against Crossref (Krylov.jl) and arXiv (Universal Differential Equations),
then the four bibliography/metadata scripts were run. Remaining scope is listed above.

## Stateful SciML primal review fix (2026-10-26)

The SciMLSensitivity extension previously threw for nonempty incoming Lux state in
`_flow`, so loading it changed ordinary forward/reverse evaluation and inference.
The CountingDynamics tests ran before the extension loaded, and the follow-up test
asserted this regression instead of preventing it. Nonempty incoming state now
invokes the generic `AbstractIntegrator` flow and retains legacy evaluation-state
threading, including with an explicit `sensealg`. CNF/logdet keeps its existing path.

`_check_factor_ad(ad, f, st)` validates state before entering AD at factor Jacobian,
energy/free-energy gradient and Mycelium graph gradient/`parameter_step` entry
points. The default delegates to the existing two-argument hook, preserving backend
checks such as Enzyme runtime activity. Mycelium passes each factor's own state
subtree. AD with nonempty SciML state raises a clear `ArgumentError`; reference
differences remain available. An arbitrary outer DI call bypasses these entry
checks, as for Enzyme runtime activity, so its caller must enforce the empty-state
restriction. No stateful SciML sensitivity support is claimed.

`test/sciml_state.jl` contains the CNF and CountingDynamics checks shared by
`test/sciml.jl` and the post-SciMLSensitivity follow-up: forward/reverse/logdet state
threading, evaluation counts, inversion labels and inference in both polarities.
`test/lp002_followups.jl` replaces the regression assertion with primal success and
AD rejection; it checks factor/graph errors before RHS evaluation, nested nonempty
state, explicit sensitivity algorithms on primal fallback, and reference differences.

Julia 1.13.1, CPU. Full review-fix runs use
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'`; counts sum the
top-level test summaries:

| Package | Passed | Pkg.test wall time |
|---|---:|---:|
| ImplicitLayers | 1,195 | 475.2 s |
| LenticulumCore | 316 | 46.0 s |
| Mycelium | 494 | 11.1 s |

Total: **2,005 passed**, no failures or errors. The implicit suite includes the
same 25 shared SciML state assertions in both load orders, plus the post-load
extension assertion (26). Its endpoint sensitivity test now has 86 assertions,
and the stateful AD rejection test has 36. The linear endpoint derivative and
200-dimensional Krylov gradient errors remain `4.912736883966318e-13` absolute
and `9.183283481970577e-11` relative respectively. The previous LP-002 SciML and
Krylov work is retained.

## Graph schedule differentiation (2026-10-27)

[[LP-011 Gradient couplings and isfrozen are declared but unused]] implements finite static
schedule derivatives with DI ForwardDiff/Zygote and reference differences. Mixed edge cuts,
freezing, terminal score estimators and straight-through rules are documented in
[[gradient_couplings]]. This closes the static CPU fragment of schedule differentiation;
flooding/residual schedules, other schedule backends and implicit graph adjoints remain.
The existing diagonal driver retains its LP-002 backend support.


## Solver numerics (2026-10-27)

Maintainer decision: implement the solver numerics group of the remaining list.
The unchanged defaults are `BroydenSolver()` and `DenseIFT()`; all new choices are
opt-in. No new external reference is needed: the compact Broyden update and
second-order adjoint below are derived from the existing residual equation;
Krylov conventions were checked against its installed `gmres.jl` documentation.

- `LimitedBroydenSolver(; memory=20, maxiters=100, tol=1e-10, init_scale=1.0)`
  stores at most m transformed secant pairs, two vectors per rank-one correction,
  O(mn) work/storage and no dense inverse. At capacity it drops the oldest correction;
  no multisecant interpolation or convergence guarantee is claimed. With no eviction,
  the iterates reproduce dense Broyden up to roundoff. The first step is Picard at
  unit scale, damped Picard otherwise. Degenerate updates use the existing denominator
  guard; `SolveReport` follows existing conventions. [[solve]] gives the update.
- `KrylovIFT(; precond, ldiv=false, on_solve)` accepts a left operator or a problem
  factory returning `(; left, right)`. It follows Krylov's M/N and mul!/ldiv!
  conventions; factories receive the operator orientation and current problem data.
  `JacobiIFT()` builds an inverse diagonal with n products and O(n) storage. This
  setup can be expensive. Reusing Broyden history is deferred: it would require new
  retained primal workspace and an orientation contract. Every RHS still checks
  its original residual. `on_solve` exposes iteration counts and that residual.
- **Interpretation of “reverse-input”:** the history and [[deq]] say “different
  variable held fixed,” so this means `solve_input(...; ad)`, observing z and
  differentiating the solved x with respect to z and parameters. The input
  cotangent of `solve_state` was already covered. An internal problem adapter
  reuses every first-order backend rule, with dense or Krylov linear solves.
  The AD path fixes incoming Lux state and advances it once at the root; ordinary
  `solve_input` keeps legacy evaluation-state threading. Nonempty state can thus
  yield different equations. Initial guesses, reports and state have no tangent.
- `deq_hvp(loss, f, x, ps, st, v; wrt=:ps, ad=AutoForwardDiff())` is a dedicated
  dense second-order adjoint for scalar `loss(zstar)`, with `:x` and `:joint`
  coordinate selections. Joint order is input then flattened parameters, including
  mixed Hessian blocks. At one converged root it differentiates residuals and loss
  twice, never root iterations. Only CPU real floating-point data, DenseIFT and
  ForwardDiff residual/loss Hessians are supported. It uses quadratic storage in
  all combined coordinates. Other backend selections are rejected explicitly.
  Nested ForwardDiff solves, ForwardDiff over other solve backends and nested
  ChainRules reverse pullbacks get clear `ArgumentError`s. Arbitrary native
  Enzyme/Mooncake tape nesting is not a supported second-order API.

`lib/ImplicitLayers.jl/test/solver_numerics.jl` is the reproducible evidence:
full Broyden traces and allocation scaling; both existing linear hard cases;
reverse-input linear/nonlinear closed forms, active warm starts, fixed state,
singular/nonconverged errors and reference differences; ill-conditioned diagonal
IFT with both preconditioner sides and both application conventions; HVPs against
closed-form linear Hessians including mixed input/parameter blocks, analytic
nonlinear scalar derivatives, and finite differences of first-order IFT gradients.

The remaining solver scope is relative/scaled root tolerances, globalization,
Anderson acceleration, scalable second-order products and general backend nesting.
Retained Broyden preconditioners and reverse-input dense second order are implemented
in batch A below. GPU and complex
implicit differentiation were untested in that run; batch B below adds a JLArrays
first-order real subset. The broader LP-002 Remaining list above
continues to track SciML, graph and metric work separately.


Files changed in the solver numerics group:

- `lib/ImplicitLayers.jl/src/solve.jl`, `solve.md`: compact primal, preconditioner API,
  convergence and storage limits, derivations.
- `lib/ImplicitLayers.jl/src/deq.jl`, `deq.md`: reverse-input problem adapter,
  dedicated HVP entry point, polarity/state contract and second-order theory.
- `lib/ImplicitLayers.jl/src/ImplicitLayers.jl`, `ImplicitLayers.md`: exports and tables;
  the module note also corrects stale claims about missing SciMLSensitivity support.
- `lib/ImplicitLayers.jl/ext/ImplicitLayersKrylovExt.jl`: preconditioners and diagnostics.
- `lib/ImplicitLayers.jl/ext/ImplicitLayersForwardDiffExt.jl`: second-order adjoint,
  nested-primitive guard.
- `lib/ImplicitLayers.jl/ext/ImplicitLayersChainRulesCoreExt.jl`: explicit nested
  reverse guard, preserving the first-order pullback calculation.
- `lib/ImplicitLayers.jl/ext/ImplicitLayersDifferentiationInterfaceExt.jl`: error messages
  cover both solve polarities. Native Enzyme/Mooncake rules are reused unchanged.
- `lib/ImplicitLayers.jl/test/solver_numerics.jl`, `test/runtests.jl`: closed forms,
  reference differences, iteration/allocation measurements and suite inclusion.
- `docs/src/packages/implicitlayers.md`, this note and `Problem Log.md`: public API,
  status, evidence and remaining work.


### Solver numerics measurements

Julia 1.13.1, CPU; emitted by `test/solver_numerics.jl` in the full ImplicitLayers
`Pkg.test()` run on 2026-10-27. Root tolerance is `1e-12` in the comparison table:

| Residual | Dense Broyden | Limited, memory=100 | Limited, memory=2 |
|---|---:|---:|---:|
| Existing contractive 2D linear case | 4 | 4 | 23 |
| Existing noncontractive 2D linear case | 2 | 2 | 2 |
| Quadratic, smaller-root branch | 6 | 6 | 11 |

Full traces agree within `2e-12` absolute/relative tolerance when nothing is
evicted. With memory=2 and ten fixed iterations, warmed total allocations are
**6,566,256 bytes** at n=10,000 and **13,126,256 bytes** at n=20,000.
These are total allocations, not peak resident memory. A dense Float64 inverse
alone at the latter size would require 3.2 GB (decimal).

The IFT oracle has diagonal residual Jacobian with entries from `1e-3` to 1
(condition number 1,000), n=20, GMRES memory=20, `rtol=1e-12,atol=1e-13`.
A warm start at the analytic root isolates linear-solve cost. The largest per-RHS
iteration counts in the scalar parameter-gradient calculation are:

| Outer AD | Unpreconditioned | Jacobi | User left inverse | User right inverse | User left solve (`ldiv=true`) |
|---|---:|---:|---:|---:|---:|
| ForwardDiff | 1 | 1 | 1 | 1 | 1 |
| Zygote | 20 | 1 | 1 | 1 | 1 |
| Enzyme | 20 | 1 | 1 | 1 | 1 |
| Mooncake | 20 | 1 | 1 | 1 | 1 |

ForwardDiff seeds coordinate RHSs, which are already eigenvectors on this diagonal
problem; no improvement is expected there. Reverse mode solves one combined RHS.
On a diagonal operator every preconditioner in the table is the exact inverse, so one
iteration is the expected result by construction. This oracle checks wiring and
orientation; it says nothing about how much Jacobi helps on a non-diagonal Jacobian.
Jacobi setup adds 20 operator products, not included in the GMRES iteration counts;
this experiment demonstrates iteration reduction, not an elapsed-time speedup.
All gradients agree with the dense/closed-form references at the existing `1e-9`
tolerances; the largest relative difference from DenseIFT is **1.937377e-14**.
A separate nonsymmetric case checks factory orientation in both polarities.

For HVPs (linear input/parameter/mixed blocks and nonlinear joint coordinates),
the largest Euclidean error against closed-form Hessian products is
**1.153778e-15**. The largest error against central differences of first-order
Zygote IFT gradients, step `1e-5`, is **9.838706e-11**. The analytic nonlinear
scalar first and second derivatives also pass. Backend-selection and nested-rule
errors are asserted, including the `deq_hvp` hint for nested primitive derivatives.


### Full solver numerics verification (2026-10-27)

All seven package suites pass on Julia 1.13.1, CPU. Commands are
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'`, with `--project=.`
for Lenticulum. Counts sum the top-level summaries; times are the printed
`@time Pkg.test()` wall times, including test subprocess startup and compilation.
Independent ImplicitLayers and package-matrix processes overlap, so these are
validation timings, not controlled performance benchmarks.

| Package | Passed | Pkg.test wall time |
|---|---:|---:|
| LenticulumCore | 314 | 46.0 s |
| Mycelium | 785 | 195.0 s |
| Lenticulum | 1,203 | 343.8 s |
| ImplicitLayers | 1,468 | 672.2 s |
| VariationalDiffusion | 1,408 | 441.3 s |
| Adversarial | 374 | 9.8 s |
| AlgebraicFactors | 15,985 | 165.3 s |

Total: **21,537 passed**, no failures, errors or broken tests. ImplicitLayers
includes **237 new solver-numerics assertions** (42 limited-Broyden, 60 linear
reverse-input, 21 nonlinear reverse-input, 88 preconditioning and 26 second-order).
The existing matrix-free dense-Jacobian guard and SciML coverage remain passing.

Both documentation commands pass with Node 26.12.1 and `QUARTZ_DIR` set to a
disposable copy of `~/Code/MathStruct-quartz`, whose checkout/clean EXIT hook was
removed before use. The initial standalone vault build takes **7.73 s**, renders
**254 HTML pages**, and has no KaTeX errors. The initial
`julia --project=docs docs/make.jl` build takes **107.70 s**, including the vault,
with cached tutorial renderings. Documenter reports existing core-page and
search-index size warnings; deployment is skipped locally. The evidence section
is followed by a final refresh of both builds. `git diff --check` passes.

No new bibliography entry or contributor metadata is added. Source `docs/quartz`
and `meta/` are untouched. No checkout, clean, reset, stash, commit, push or merge
command was used. Logs are in `/tmp/lp002-numerics-clwaymbc/` (the permitted
scratch area for this run).


## Batch A: SciML breadth and solver leftovers (2026-10-27)

Defaults remain dense Broyden, DenseIFT, and stage/evaluation Lux-state threading.
The request's advance-once flow premise differed from the actual implementation:
Euler/RK4 advance state at every stage. `NeuralODEFactor(...; state_policy=:fixed)`
therefore adds the requested semantics explicitly for built-in and SciML solvers.
Each RHS receives constant incoming state; returned state comes from one final
endpoint/time evaluation. State is not differentiated and updates must be
functional. Fixed-state logdet throws. Existing nonempty-state primal fallback,
state-threaded AD errors and inversion labels remain unchanged.

`test/lp002_batch_a.jl` is the reproducible evidence for state-dependent affine
closed forms, both integration directions and tight fixed-state RK4 references.
The solver matrix covers Tsit5, Vern7 and SDIRK Kvaerno3/KenCarp4, default sensitivity,
InterpolatingAdjoint and GaussAdjoint (ZygoteVJP), outer ForwardDiff/Zygote/Mooncake,
plus a fast scalar decay mode. Runtime-activity Enzyme reverse is tested on the
default Tsit5 scalar endpoint loss path. Vector-return split reverse fails in
Enzyme’s ABI wrapper and is explicitly rejected for energy Jacobians. Forward Enzyme fails in OrdinaryDiffEq callback
initialization and is rejected at package derivative entry points.
`examples/lp002_enzyme_limits.jl` preserves the direct minimal reproducer.

`BroydenIFT()` retains the actual `LimitedBroydenSolver` inverse corrections per
solve and applies them as a Krylov left inverse. Adjoint solves swap the two
correction lists to apply the transpose. Public reports remain SolveReport; no
global cache, extra root solve or dense Jacobian is introduced. Both solve
polarities and all four first-order backends are checked. Other root solvers and
`ldiv=true` throw. Warm starts may yield no history, and approximation quality can
be poor; measured iteration counts do not establish an elapsed-time speedup.

`solve_input_hvp(loss, f, z, ps, st, v; wrt=:ps, ad=AutoForwardDiff(), x₀=...)`
reuses the dense second-order adjoint in reverse polarity. `:z` selects observed
state and `:joint` orders state before parameters. Linear mixed blocks, analytic
nonlinear second derivatives and finite differences of first-order IFT gradients
are tested. Dense CPU real data and loss dependence only through the solved
input remain required. This does not add Krylov second order or general nesting.

Automatic annotations for arbitrary outer DI closures remain outside the solve
primitive: Enzyme selects closure activity before dispatching its rule. A global
change would intercept DI/Enzyme-owned methods and could discard intentionally
active closure data. The reproducer raises EnzymeMutabilityException, while
explicit `function_annotation=Enzyme.Const` gives the closed-form derivative.
Library-owned parameter derivatives already use constant callable/context
arguments. The failing and working outer-closure cases are tested.

The diagonal Mycelium parameter driver is checked with fixed-state affine
SciML factors on four backends. LP-011 and [[gradient_couplings]] record that this
extends endpoint coverage, without changing schedule differentiation semantics.
No new citation or contributor metadata is introduced.


### Batch A solver compatibility finding

The full matrix exposed an inaccurate Kvaerno3/InterpolatingAdjoint reverse
sensitivity on the installed stack. Default direct sensitivities and GaussAdjoint
pass with Kvaerno3; KenCarp4 passes with both continuous adjoints. Tight Newton
convergence, finite-difference Newton Jacobians and a smaller maximum step did
not fix the failed pair. A typed optional extension now rejects it at factor/graph
AD entry points, preserving primals and direct ForwardDiff solves.
`examples/lp002_sciml_adjoint_limit.jl` reproduces the discrepancy outside those
guards. Arbitrary outer DI calls must enforce the same restriction. The failure
is retained explicitly as an unresolved upstream compatibility item. The repository
reproducer measures Euclidean gradient errors **3.7626557e-5** in forward time
and **6.2473244e-5** in reverse time (abstol=1e-12, reltol=1e-11). The working
Kvaerno3/GaussAdjoint forward-time error in the same run is **3.3312075e-10**.


### Batch A measured solver behavior

CPU, Julia 1.13.1; `test/lp002_batch_a.jl` in the final full ImplicitLayers run.
The non-diagonal n=12 residual matrix has condition number **96.92901762**.
It is a seeded rotated logarithmic spectrum (0.01 to 1), multiplied by a
nonsymmetric triangular perturbation. All variants share the same forward
limited Broyden solve: memory=60, init_scale=10, root tolerance=1e-8. GMRES uses
memory=12, itmax=200, rtol=1e-11, atol=1e-12. Counts are the largest per-RHS
iteration counts in the scalar parameter-gradient calculation and **include
residual refinements**:

| Outer AD | None | Jacobi | Retained Broyden |
|---|---:|---:|---:|
| ForwardDiff | 12 | 12 | 17 |
| Zygote | 12 | 12 | 11 |
| Enzyme | 12 | 12 | 11 |
| Mooncake | 12 | 12 | 11 |

The improvement in reverse mode is modest; forward mode regresses. Jacobi adds
12 setup products; retained Broyden adds no probes, but needs the retained root
history. No elapsed-time speedup is established. The largest relative gradient
difference from DenseIFT is **1.4396653e-13**. A separate memory=2 case checks
actual eviction, and nonsymmetric two-dimensional fixtures explicitly check H/H'
and both solve polarities. Very tight root tolerances and poorly scaled retained
histories can destabilize this unglobalized Broyden implementation; the original
residual check is never bypassed.

Reverse-input HVP errors are **1.8507996e-15** against closed-form Hessian
products and **1.6291895e-11** against central differences of first-order
Zygote IFT gradients (step 1e-5). These are dense CPU results with fixed state,
not general backend nesting or scalable second-order claims.

### Batch A files changed

- `docs/src/packages/mycelium.md`
- `docs/src/packages/implicitlayers.md`
- `lib/ImplicitLayers.jl/Project.toml`
- `lib/ImplicitLayers.jl/examples/lp002_enzyme_limits.jl`
- `lib/ImplicitLayers.jl/examples/lp002_sciml_adjoint_limit.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersChainRulesCoreExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersDifferentiationInterfaceExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersEnzymeCoreActivityExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersEnzymeCoreExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersForwardDiffExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersKrylovExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersMooncakeExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersSDIRKSensitivityExt.jl`
- `lib/ImplicitLayers.jl/ext/ImplicitLayersSciMLSensitivityExt.jl`
- `lib/ImplicitLayers.jl/src/ImplicitLayers.jl`
- `lib/ImplicitLayers.jl/src/ImplicitLayers.md`
- `lib/ImplicitLayers.jl/src/deq.jl`
- `lib/ImplicitLayers.jl/src/deq.md`
- `lib/ImplicitLayers.jl/src/flow.jl`
- `lib/ImplicitLayers.jl/src/flow.md`
- `lib/ImplicitLayers.jl/src/neuralode.jl`
- `lib/ImplicitLayers.jl/src/neuralode.md`
- `lib/ImplicitLayers.jl/src/solve.jl`
- `lib/ImplicitLayers.jl/src/solve.md`
- `lib/ImplicitLayers.jl/test/Project.toml`
- `lib/ImplicitLayers.jl/test/lp002_batch_a.jl`
- `lib/ImplicitLayers.jl/test/runtests.jl`
- `lib/Mycelium.jl/src/gradient_couplings.md`
- `vault/Project/Problems/LP-002 The Jacobian side is not implemented.md`
- `vault/Project/Problems/LP-011 Gradient couplings and isfrozen are declared but unused.md`
- `vault/Project/Problems/Problem Log.md`


### Full batch A verification (2026-10-27)

All seven package suites pass on Julia 1.13.1, CPU. Commands are
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'` (top level:
`--project=.`). Counts sum top-level test summaries; times are the printed
Pkg.test wall times including subprocess startup and compilation. Processes
for independent packages and diagnostics overlap, so these are validation
wall times, not controlled performance benchmarks.

| Package | Passed | Pkg.test wall time |
|---|---:|---:|
| LenticulumCore | 314 | 46.3 s |
| Mycelium | 785 | 195.8 s |
| Lenticulum | 1,203 | 341.8 s |
| ImplicitLayers | 1,894 | 1203.6 s |
| VariationalDiffusion | 1,408 | 453.6 s |
| Adversarial | 374 | 9.4 s |
| AlgebraicFactors | 15,985 | 172.8 s |

Total: **21,963 passed** in the final passing runs, with no failures, errors or
broken tests. ImplicitLayers has **1,894 assertions**, including **426 new batch A
checks**: 324 solver/state sensitivities and compatibility errors, 62 retained
history/preconditioning, 13 reverse HVP, 12 graph gradients, 2 closure annotations,
7 nested floating-state, 2 stateless Enzyme scalar gradients and 4 stateless
reverse-time continuous-adjoint checks. The largest Euclidean endpoint parameter
gradient error against the supported affine closed forms is **1.8841397e-9**.
Existing matrix-free and default state-threading regression checks pass.

The implicit environment uses SciMLSensitivity 7.119.13, OrdinaryDiffEqVerner
2.4.2, SDIRK 2.9.8, Enzyme 0.13.216, Mooncake 0.5.65 and ForwardDiff 1.4.6.
The initial Core run produced a NaN once in its existing Enzyme endpoint Jacobian
check. Its full rerun passes (314 assertions); ten repeated isolated Enzyme
Jacobians are finite and exact. No Core source change was made. This transient
failure is recorded rather than hidden by the final passing table.

The vault review build passes in **8.35 s**, with **254 HTML pages** and no KaTeX
errors. The API/vault review build (`julia --project=docs docs/make.jl`) passes in
**113.18 s**, using cached tutorial renderings. Both use `QUARTZ_DIR` pointing at
a disposable copy of `~/Code/MathStruct-quartz`, with the checkout/clean EXIT hook
removed before any build. Documenter reports existing core-page/search-index
size warnings and skips deployment locally. Both builds are refreshed after this
evidence is written; final refresh timings are retained in the run logs.

`git diff --check` passes. No new bibliography entry or contributor metadata is
added. Source Quartz checkouts and `meta/` are untouched. No checkout, clean,
reset, stash, commit, push or merge command was run. Logs, the machine-readable
suite summary, diagnostics and the disposable Quartz copy are under
`/tmp/lp002-batch-a/`; final package logs are listed in `verification.json` there.


## Batch B: device-array coverage (2026-10-27)

`lib/ImplicitLayers.jl/test/device_arrays.jl` is the reproducible evidence, using
JLArrays with `allowscalar(false)` throughout, Float32 and Float64. No hardware
GPU is available. GPUArraysCore weak extensions carry storage-specific behavior;
no new public API or mandatory GPU dependency is added. CPU defaults and primal
arithmetic remain unchanged. Device cases that previously threw now gain coverage.

- DEQ roots: Picard, dense Broyden and LimitedBroyden, both solve polarities,
  linear closed forms and CPU references. Device default guesses/storage follow
  the observed array; damping/scales preserve its element type.
- First-order IFT: Zygote and directional ForwardDiff with KrylovIFT, both observed
  and parameter coordinates, both polarities, zero warm-start derivatives,
  nonlinear references, unpreconditioned/Jacobi/retained Broyden and supplied
  left/right operators, including left `ldiv=true`. Jacobi probes use broadcast
  seeds/slices; ForwardDiff dual reconstruction broadcasts columns. GMRES requires
  tolerances in the RHS real type, so the old Float64 keywords failed for Float32.
- DenseIFT: JLArrays has no working LU with scalar indexing disabled. Both
  backends get a clear ArgumentError recommending KrylovIFT; no dense host fallback
  is hidden in the primitive. Hardware device factorisation success is untested.
- NeuralODE Euler/RK4: both directions, discrete polynomial endpoints and
  input/parameter gradients, both state policies, energy/free-energy derivatives
  and graph parameter steps. Device time coefficients preserve Float32. The old
  threaded `foldl` initializer lost Zygote input cotangents. A ChainRules-only stage loop
  fixes that derivative; the original fold primal is retained for native backends,
  and CPU primals compare bit for bit with it. Functional nonempty-state flows
  are also compared with CPU.
- Parameter and graph entry points: array-shaped energy_gradient,
  free_energy_gradient, parameter_step; LP-011 exact/pathwise/diagonal boundaries,
  detach, subset/all freezing, raw static inference AD, and terminal diagonal-
  Gaussian score estimates/standard errors with common host RNG draws. The Gaussian
  oracle includes its variance and exposed graph Bethe entropy correction.

Active-coordinate graph operations now use device gathers and broadcast masks;
host integer metadata has no tangent. Device GradientDescent/Momentum/Nesterov
coefficients are cast to parameter precision, including the default learning rate.
Diagonal Gaussian moments/precision solves/logdet use elementwise algebra and
structural covariance cotangents. Dense Gaussian device precision/covariance is
explicitly rejected. CPU Gaussian factorisation remains unchanged.

Core's device ForwardDiff adapter uses DI pushforwards, one direction per active
coordinate; native ForwardDiff.gradient/jacobian still scalar-index during vector
seeding on this stack. This adapter is for small reference problems, not a scalable
parameter-gradient claim. Device finite differences, scalar leaves in device
parameter trees (use one-element arrays), HVPs and Enzyme/Mooncake device parameter/
solve rules give ArgumentErrors. Enzyme/Mooncake native device internals are not
tested or claimed. SciML device solvers, divergence, complex devices, dense Gaussian
factorisations and native GPU RNGs remain outside this batch. [[gradient_couplings]]
and LP-011 describe the graph subset; other schedules/backends remain open.

The first full ImplicitLayers run exposed an Enzyme forward runtime-activity
BLAS failure after replacing the general flow fold with a loop. The fix retains
the fold primal and uses the loop only for ChainRules reverse; the existing native
flow testset then passes unchanged (80 assertions). Gaussian diagnostics/inversion
and raw replay also exposed typed-reduction, scalar diagonal-inverse and vector-
cotangent failures, fixed without changing CPU primal arithmetic.

No new citation or contributor metadata is introduced. The earlier batch A work
is retained. Verification counts and measurements follow after the full runs.


### Batch B implicit-message replay finding

Reviewing replay exposed a separate wiring gap: DEQ `factor_message` did not
receive the selected backend, so a downstream graph gradient could trace root
iterations. A private constant backend context now reaches the ImplicitLayers
message hook, which calls the existing IFT primitive in the selected polarity.
Tests compare nonzero downstream DEQ/Euler message losses and parameter steps
with closed forms in both directions and both precisions, using ForwardDiff and
Zygote. Raw ChainRules inference also selects the IFT hook. No ordinary inference
or diagonal-gradient default changes. Raw ForwardDiff implicit inference is not
automatically converted to IFT; use the explicit graph gradient driver. Input-
message replay with nonempty DEQ state throws because legacy primal state threading
and the fixed-state IFT equation differ. [[gradient_couplings]] records the boundary.

Raw ForwardDiff DEQ messages now reject dual-valued unannotated solves with an
ArgumentError pointing to the explicit graph gradient driver, preventing accidental
root-iteration differentiation. Ordinary primal messages are unchanged.


### Batch B resumption review (2026-10-27)

The interrupted working tree was retained, including batch A. Two further
Float32 cases were found during review: diagonal Gaussian `logpartition` still
used Float64 constants, and the final `:fixed` flow state evaluation still
received the original Float64 endpoint time. Device-only partition algebra and
the existing storage-specific time helper now cover them. Partition values and
ForwardDiff/Zygote gradients are checked against CPU formulas. A time-dependent
Euler/RK4 fixture asserts the time precision at every call, including the final
state update, and compares endpoints and gradients with the stability polynomial
of a dense augmented linear system. Non-unit root scales/damping and DenseIFT
errors in both solve directions also have regression checks.

The sandbox makes `~/.cache/lenticulum/` read-only. This resumption therefore uses
`~/.julia/lenticulum-runs/lp002-batch-b-resume/` for logs and its disposable Quartz
copy, on disk rather than the RAM-backed `/tmp`. Only one Julia process runs at
a time. Full suites run by including each complete `test/runtests.jl` in its own
test environment, with `--startup-file=no --heap-size-hint=6G`.
Focused native-history reproductions use `--check-bounds=yes`; the final
seven full suites use the standard automatic bounds-checking setting.
The top-level package uses a disposable environment assembled from its declared
test target and local package sources. The wrapper's `verification.json` records
exact commands, counts, wall times and sampled peak RSS. No test file is filtered
or skipped. These timings include loading and compilation, and are not GPU
performance measurements.

### Batch B files

- Optional extensions: `lib/ImplicitLayers.jl/ext/ImplicitLayersGPUArraysCoreExt.jl`;
  `lib/LenticulumCore.jl/ext/LenticulumCoreGPUArraysCoreExt.jl`,
  `LenticulumCoreGPUArraysCoreDIExt.jl`, `LenticulumCoreGPUArraysCoreChainRulesExt.jl`;
  `lib/Mycelium.jl/ext/MyceliumGPUArraysCoreExt.jl`.
- Implicit primitives: `lib/ImplicitLayers.jl/src/solve.jl`, `deq.jl`, `flow.jl`,
  `neuralode.jl`; `ext/ImplicitLayersChainRulesCoreExt.jl`,
  `ImplicitLayersDifferentiationInterfaceExt.jl`, `ImplicitLayersForwardDiffExt.jl`,
  `ImplicitLayersKrylovExt.jl`; corresponding implementation notes and module note.
- Core beliefs: `lib/LenticulumCore.jl/src/gaussian_belief.jl`,
  `gaussian_belief.md`, `energy.md`, `parameter_tree.jl`, `parameter_tree.md`.
- Graphs: `lib/Mycelium.jl/src/free_energy.jl`, `gaussian.jl`,
  `gradient_couplings.jl`, `graph.jl`; their notes and `factors.md`;
  `ext/MyceliumChainRulesCoreExt.jl`.
- Environments/tests: the three packages' `Project.toml` files;
  `lib/ImplicitLayers.jl/test/Project.toml`, `runtests.jl`, `device_arrays.jl`.
- Top-level replay adapters: `src/gaussian.jl`, `composition.jl` and their notes
  thread the new constant backend context through existing Gaussian message rules.
- `lib/VariationalDiffusion.jl/test/runtests.jl`: clarify that JLArrays emulation
  does not establish hardware CUDA support; assertions are unchanged.
- Documentation: `docs/src/packages/implicitlayers.md`, `lenticulumcore.md`,
  `mycelium.md`, `lenticulum.md`; this note, LP-011 and Problem Log.


The first resumed top-level full run failed in the existing Gaussian replay
closed-form oracle: the new private backend argument bypassed three older
Gaussian message specialisations (factor, prior and collapsed composite). The
result traced ordinary Cholesky inverses into LAPACK during Zygote replay.
All three hooks now accept/forward the context and retain their existing
functional matrix-inverse formulas. The failing log is retained as
`Lenticulum-initial-failure.log`; the targeted oracle and full suite are rerun.


The first resumed ImplicitLayers full run reached the retained-history test
with its earlier CPU solver/AD and fixed-state SciML oracles passing, then
Enzyme rejected an abstract augmented-return type. Reading the history through
`Ref{Any}` lost its concrete type. An explicit rule-return-contract trial fixed
reverse mode but triggered an Enzyme LLVM assertion in batched forward mode;
that trial was removed. The final fix returns history as a typed third result
from the private limited-Broyden solve, preserving the existing iteration loop,
residual count, correction vectors and public `SolveReport` return. The
62-assertion history testset is rerun with bounds checking before the complete
suite. `ImplicitLayers-initial-failure.log`,
`enzyme-history-forward-failure.log` and `enzyme-history-contract-abort.log`
retain these findings. No Enzyme device support or new public API is added.


A subsequent empty-parameter-branch probe reproduced host scalar indexing in
functional flattening: the empty subtree contributed a host `Float64[]` to
`vcat`. The storage-specific flatten helper now omits empty device-tree parts
and preserves all-empty device storage. ForwardDiff/Zygote solved-state gradients
with empty branches match dense closed forms; the focused partition/time/tree
regressions pass 188 assertions. This shared-helper change triggers a fresh
sequential verification of all seven packages.


### Batch B device measurements (2026-10-27)

The final complete ImplicitLayers run passes **5,308 assertions**, including
**3,414 device assertions** in `test/device_arrays.jl`, with scalar indexing
disabled. This includes the retained batch A tests. Julia 1.13.1 uses JLArrays
0.3.3, GPUArraysCore 0.2.1, Krylov 0.10.11, ForwardDiff 1.4.6 and Zygote 0.7.14.
No hardware GPU is present. The device cases use two-dimensional systems, bounded
root/linear iterations and eight common-random-number score draws.

For each vector checked by `device_check`, the recorded error is
`norm(Array(result) - reference) / max(1, norm(reference))`. The largest values
in the full run are:

| Element type | Maximum scaled vector error | Comparison tolerance (rtol and atol) |
|---|---:|---:|
| Float32 | 3.0809478630544618e-6 | 2e-5 |
| Float64 | 2.7084487851328933e-11 | 1e-9 |

Every checked vector also asserts its device storage and exact element type.
The scalar log-partition has separate value/type assertions. References are
closed linear/nonlinear IFT formulas, dense CPU derivatives, discrete flow
polynomials, Gaussian expectation/entropy formulas and CPU common-draw score
estimates. The maximum statistic includes these different vector observables;
it is not a solver convergence norm or a hardware performance result.

The full ImplicitLayers wall time is **1,712.61 s**, including loading and
compilation; sampled peak resident memory is **6.861 GiB**. Root and integration
default arithmetic is retained. The full suite rechecks native CPU Enzyme/Mooncake
coverage, unchanged forward-loop semantics, limited-Broyden traces/allocation
scaling, retained-history gradients and both dense second-order adjoints.


### Full batch B verification (2026-10-27)

All seven final package suites pass sequentially on Julia 1.13.1. Each complete
`test/runtests.jl` runs in its package test environment (the top-level declared
test target is resolved into the disposable environment described above), with
`--startup-file=no --heap-size-hint=6G`, standard automatic bounds checking, one
Julia process at a time and single-threaded BLAS. No test file or testset is
filtered. Times include loading and compilation; they are validation timings.

| Package | Passed | Full-suite wall time | Sampled peak RSS |
|---|---:|---:|---:|
| LenticulumCore | 314 | 44.02 s | 0.944 GiB |
| Mycelium | 785 | 190.07 s | 0.858 GiB |
| Lenticulum | 1,203 | 478.17 s | 1.890 GiB |
| ImplicitLayers | 5,308 | 1712.61 s | 6.861 GiB |
| VariationalDiffusion | 1,408 | 391.15 s | 1.982 GiB |
| Adversarial | 374 | 9.00 s | 0.411 GiB |
| AlgebraicFactors | 15,985 | 163.06 s | 1.590 GiB |

Total: **25,377 passed**, no failures, errors or broken
tests. The sequential suite wall times sum to **2988.07 s**;
this excludes earlier diagnostic/failing runs and documentation. The final
ImplicitLayers count includes all 1,894 prior assertions and the 3,414 device
assertions above. Native history and GPU empty-tree regressions pass in the full
run, as do the unchanged CPU and existing device checks of the other packages.

`~/.julia/lenticulum-runs/lp002-batch-b-resume/verification.json` contains exact
commands, counts and timing/RSS records; each package has its own final log.
Earlier failing logs and focused reproductions remain alongside them. The first
resumed failures are repaired, rather than hidden by dropping tests or backends.


### Batch B documentation and final review

Both requested documentation commands pass with Node 26.12.1 and `QUARTZ_DIR`
pointing at the disposable disk copy of `~/Code/MathStruct-quartz`. Its
checkout/clean EXIT hook was removed before any build. The first standalone
`docs/site/build.sh` takes **9.90 s** and renders **254 HTML pages**, without
KaTeX errors. After resolving the docs environment to register the optional
extensions, `julia --project=docs docs/make.jl` passes in **443.15 s**,
re-executing all ten Literate tutorials and building the vault. Sampled peak
Julia RSS is **2.762 GiB**. Documenter reports its existing
core-page/search-index size warnings and skips deployment locally.

Both builds are refreshed after writing this evidence; their final refresh
timings are retained in the run logs. The disposable Quartz copy is removed
after the final builds. Source Quartz checkouts and `meta/` are untouched.
`git diff --check` passes. No new citation or contributor metadata is added, and
no checkout, clean, reset, stash, commit, push or merge command is run.

Hardware GPU execution, dense device LU/Gaussian factorisation, native
Enzyme/Mooncake device rules, SciML/device divergence, device HVPs and broader
graph schedule/message/backend coverage remain outside the verified subset.
ForwardDiff uses the directional device adapter; native vector seeding still
scalar-indexes on this stack. LP-002's broader Remaining list above stays open.


## Batch C: graph schedule adjoints (2026-10-28)

The interrupted A/B/C working tree was retained on `dev-danielb`. No commit,
checkout, clean, reset, stash, push or merge is part of this work. Logs and the
disposable Quartz copy use `~/.julia/lenticulum-runs/lp002-batch-c-resume/`;
`/tmp` is RAM-backed and `~/.cache` is read-only in this sandbox. Julia tests run
one process at a time, with single-threaded BLAS and a 6 GiB heap-size hint.

### Realised schedule computation

`free_energy_gradient(...; schedule, nsweeps, damping=0.0)` extends the
non-diagonal finite replay to flooding and residual schedules. Flooding reads a
snapshot per sweep, with functional Lux state threaded in task order. Residual
scheduling records the reference pass's committed task sequence, including ties
and early exit, then replays it as fixed. Raw ChainRules `infer!` records its
actual reference choices. The derivative is piecewise-defined: it differentiates
the realised sequence, not argmax, convergence decisions or changing polarity/
message-family branches. Pure, deterministic, functional factor evaluations are
required. Outgoing damping is inside the update; variable-to-factor pools remain
undamped. Score estimates retain their single undamped static sweep restriction.

`test/graph_adjoint.jl` compares Gaussian tree/loop values with ordinary message
passing and derivatives with reference differences of the same fixed computation.
`test/graph_adjoint_devices.jl` covers Float32/Float64 diagonal Gaussians and
Diracs with JLArrays scalar indexing disabled. Device damping coefficients retain
payload precision; CPU/default primal arithmetic is unchanged. The old schedules
note incorrectly claimed there was no residual executor; that statement is fixed.

### Converged message adjoint

`adjoint=:implicit` opts into a fresh converged reference inference, using
`schedule`, `maxsweeps=1000`, `tol=1e-10` and `damping=0.0`. The supplied store is
untouched, and `nsweeps` controls only finite replay. With message coordinates $m$,
synchronous BP map $T$ and scalar Bethe objective $L$,

$$
(I-D_mT)^\top\lambda=\partial_mL,\qquad
D_\theta L_{\mathrm{total}}=\partial_\theta L+(D_\theta T)^\top\lambda.
$$

`linear_solver=:dense` assembles the message Jacobian and uses dense LU;
`:krylov` uses matrix-free transposed products and optional Krylov.jl GMRES.
`linear_tol=1e-10` and `linear_maxiter=1000` control the linear solve. Every
original adjoint residual is checked; every ordinary synchronous BP row must satisfy
the root check, even when a tree/residual schedule found it. Detected dense
singularity, nonconvergence, illegal/incomplete schedules and unsupported combinations throw
`ArgumentError`. `parameter_step` forwards these keywords and retains its three
returned entries for deterministic gradients.

Exact/Pathwise, CPU real Gaussian (dense or diagonal precision), numeric Dirac
and uninformative messages, recursively empty incoming/returned message state,
and reference differences/ForwardDiff/Zygote are the supported domain. A diagonal
or detached attachment fixes all legal directed update rows, imposing zero message
variations across that boundary in the linear system. Frozen parameter coordinates
are pinned but input-message derivatives through the factor remain. This gives
the limit of the cut replay, rather than an ordinary uncut-objective derivative.
Agreement with a limiting unrolled gradient additionally requires convergence
of that differentiated replay. The map must be locally differentiable with invertible $I-D_mT$; contraction is
sufficient. Reference convergence alone proves neither uniqueness nor invertibility.

Dense storage is quadratic and factorisation cubic in message dimension. GMRES
keeps at most 30 Krylov vectors, with an AD gradient per transposed product;
ForwardDiff still pays forward-seed cost for that gradient. Both also pay for
reference inference, AD workspace and functional store copies. Neither tapes the
reference iterations. There is no ImplicitLayers dependency. Device implicit
coordinates are explicitly rejected for both linear solvers: a Krylov solver
alone does not supply a device coordinate adapter. Stateful/implicit-score/
higher-order graph adjoints remain open. [[graph_adjoint]] gives the complete
representation, cost and cut contract; [[gradient_couplings]], LP-011 and
L-SPEC-002 record the replay extension.

The tree oracle is an independently assembled dense global Gaussian negative log
partition; loopy references reconverge the Bethe objective at every parameter
perturbation. The tests also compare the unrolled gradient at increasing sweep
counts. The first focused run passes 120 schedule assertions and 55 implicit
Gaussian assertions, then exposes an inherited **test** error (`diff(::Tuple)`)
in the monotonicity check. That assertion now materialises the tuple as a vector;
no inference or derivative formula is changed to accommodate it. Final passing
runs and measurements follow below.

### Meaning of divergence and metric localisation

The two Remaining items first occur in LP-002 commit `6e0bbbb6` (2026-10-26),
after Sampson localisation's `b4d77ab0` implementation (2026-10-25). They are
separate from endpoint sensitivity and from the frozen-metric Gauss–Newton API.

For CNF divergence, a loss on `flow_logdet`'s correction requires derivatives of
$\operatorname{tr}D_u f_\theta$ both directly in parameters and through the
trajectory. The default trace mutates forward-difference probes with a
piecewise relative step. Any successful AD differentiates that finite numerical
program, not an exact analytic trace. Default nonempty Lux state advances at
base/probe/stage evaluations; fixed-state logdet remains unsupported. Adaptive
SciML's augmented-state RHS is separate from its parameter-aware endpoint path.

For Sampson localisation, the total energy derivative includes the changing
covariance $A=J\Sigma J^\top+\lambda R$, not just the residual derivative. With
$w=A^{-1}r$,

$$
d\left(\tfrac12r^\top A^{-1}r\right)=w^\top dr-\tfrac12w^\top(dA)w.
$$

The generic localiser still uses the mutable central-difference Jacobian and
spectral SPD validation. An outer `ad` does not replace that inner algorithm.
Gauss–Newton intentionally freezes the metric and is unchanged. Mixed residual
partials, a checked rank/acceptance policy and a functional/analytic localiser
would be needed for a general AD contract. The independent probe
`lib/ImplicitLayers.jl/examples/lp002_metric_divergence_limits.jl` uses
$r=a y^2-b$ and $(a y^2-b)^2/(8a^2y^2)$, quadratic-field trace $2a\sum u_j$, and
linear integrated trace $naT$. Its measured observations and backend replay
probes are recorded with the final validation below; these items are documented
precisely rather than declared implemented by endpoint tests.

### Batch C files

- Graph source/API: `lib/Mycelium.jl/src/graph_adjoint.jl`,
  `gradient_couplings.jl`, `schedules.jl` (docstrings), `passing.jl`, `free_energy.jl`, `Mycelium.jl`;
  `ext/MyceliumKrylovExt.jl`, `MyceliumChainRulesCoreExt.jl`,
  `MyceliumGPUArraysCoreExt.jl`; package and test `Project.toml`.
- Tests/reproducers: Mycelium `test/graph_adjoint.jl`, `graph_adjoint_devices.jl`,
  `gradient_couplings.jl`, `runtests.jl`, `examples/lp002_replay_backends.jl`;
  ImplicitLayers `examples/lp002_metric_divergence_limits.jl`.
- Core device polarity fix: `lib/LenticulumCore.jl/ext/LenticulumCoreGPUArraysCoreExt.jl`
  and `src/gaussian_belief.md` (reduce stored diagonal entries directly);
  `examples/lp002_enzyme_endpoint_limit.jl` preserves the recurring endpoint probe.
- Implementation notes: Mycelium `graph_adjoint.md`, `gradient_couplings.md`,
  `schedules.md`, `passing.md`, `free_energy.md`, `gaussian.md`, `Mycelium.md`;
  ImplicitLayers `flow.md`; Core `energy.md`.
- Package docs: `mycelium.md`, `implicitlayers.md`, `lenticulumcore.md`;
  project records: this note, LP-011, L-SPEC-002 and Problem Log.

Earlier uncommitted batch A/B changes are retained alongside these files. No new
paper citation or contributor metadata is introduced.


### Batch C native replay and metric/divergence probes

Julia 1.13.1, Enzyme 0.13.216, Mooncake 0.5.65, ForwardDiff 1.4.6,
Zygote 0.7.14. The repository reproducer
`lib/Mycelium.jl/examples/lp002_replay_backends.jl` deliberately bypasses the
public backend guard. Run it with `--project=lib/ImplicitLayers.jl/test`, selecting
`forward`, `reverse` or `mooncake`; append `gaussian` for the Gaussian case.
Both Enzyme modes use runtime activity and `function_annotation=Enzyme.Const`.

- Enzyme forward fails inside `messages_into` with `EnzymeInternalError` on all
  three tested schedules (sequential, flooding, residual).
- Enzyme reverse fails in Base's `setindex_widen_up_to` with `EnzymeNoTypeError`
  on all three. Runtime activity alone does not resolve these replay type failures.
- Mooncake returns the correct **0.7** scalar gradient for the minimal Dirac
  computation on all three. A broader Gaussian replay probe then fails at
  `_improper_check`: reverse mode does not support its `try/catch`. The smaller
  repository Gaussian reproducer retains that boundary. Successful Dirac tracing
  does not establish Gaussian Bethe or general graph support, so the public
  native-backend guard is retained. No new backend is advertised.

`lib/ImplicitLayers.jl/examples/lp002_metric_divergence_limits.jl` directly probes
the existing numerical programs. The references differentiate independent analytic
formulas. Euclidean parameter-gradient errors in this run are:

| Probe | Primal | Analytic primal | Reference differences error | ForwardDiff error |
|---|---:|---:|---:|---:|
| Sampson localisation | 0.14585934669219713 | 0.14585934669725883 | 1.421683e-6 | 1.032439e-11 |
| Quadratic field trace | 0.7000001393808741 | 0.7 | 1.360012e-4 | 1.991056e-7 |
| Linear integrated trace | 0.8399999999057993 | 0.84 | 8.005121e-6 | 1.892910e-10 |

The quadratic trace's ForwardDiff discrepancy includes the forward-probe bias;
the nested reference differences show cancellation amplification. These are CPU
probe measurements, not generic sensitivity guarantees. Zygote localisation fails
at the SPD validation's `maximum(...; init=...)` rule; raw and integrated divergence
fail on probe-array mutation. No default finite-difference, SPD, trace-quadrature
or state policy was changed. The general Remaining items stay open with this
precise meaning and runnable evidence.


### Batch C resumption review findings

The full Mycelium suite caught an operator `mul!` ambiguity with LinearAlgebra's
`AbstractQ` methods; the new method now requires vector operands. The passing
focused run has 467 assertions (194 JLArrays), in 1162.65 s, with sampled peak
RSS 1.875 GiB. A further correlated 2D dense Gaussian oracle is included in the
full rerun. CPU objective constants can promote a Float32 message cotangent;
the graph operator now advertises the RHS element type to GMRES, avoiding a
spurious operator/vector-type warning without changing message coordinates.
The interrupted notes also contained escaped-string damage to TeX; the formulas
are repaired and rendered by the vault build.

The first full Core run in this resumption reproduces the pre-existing Enzyme
endpoint Jacobian NaN recorded in batch A, at `parameter_ad.jl:38`. The failing
log is retained as `LenticulumCore-initial-failure.log`. The complete rerun and
the identity-Jacobian probe are checked separately; no assertion is removed
or relaxed. The full rerun passes all **314** suite assertions, but the extra
probe fails at repetition **2** of the planned 20, again with a NaN in the
first diagonal entry. Its log is `LenticulumCore-repeated-failure.log`. This is
a recurring backend/test nondeterminism finding. The standalone repository
reproducer is `lib/LenticulumCore.jl/examples/lp002_enzyme_endpoint_limit.jl`;
the original fixture is `test/parameter_ad.jl`. This is not a repaired graph-adjoint
defect. The matrix continues with the other six packages. Final results follow
below; a passing full-suite count does not close this additional failure.


### Batch C graph measurements

The complete Mycelium suite passes **1,265 assertions**, including **480 batch C
assertions** (286 CPU and 194 JLArrays), in **1367.48 s** including loading and
compilation. Sampled peak RSS is **2.019 GiB**. The new assertions cover the
correlated 2D dense-precision oracle, dense/GMRES adjoints, realised schedules,
cuts/freezing, parameter steps, singular/nonconverged errors and both device
precisions with scalar indexing disabled. The full ambiguity check passes, and
the operator/RHS type warning no longer occurs.

`test/graph_adjoint.jl` prints the Euclidean error of ExactCoupling's flooding
unroll against its converged implicit loopy Bethe gradient:

| Sweeps | Gradient error |
|---:|---:|
| 4 | 0.17638790415117023 |
| 8 | 0.05881271574020927 |
| 16 | 0.0010820230975263946 |
| 32 | 3.52011227898608e-5 |
| 64 | 2.5264894519687794e-10 |

The largest implicit-gradient error scaled by `max(1,norm(reference))` over
Gaussian tree/loop fixtures, Exact/Pathwise, reference differences/ForwardDiff/
Zygote and dense/GMRES is **1.3032632715663616e-9**. These references are the
independent dense global Gaussian on the tree and fully reconverged Bethe
finite differences on the loop. The test separately checks the correlated dense
Gaussian against its independent global Gaussian derivative.

`test/graph_adjoint_devices.jl` reports largest scaled replay-gradient errors
**3.07626e-7** (Float32) and **7.67305e-16** (Float64) against CPU derivatives of
the same recorded operations. The comparisons allow `3e-5` and `1e-9`,
respectively. They cover diagonal Gaussian tree/loop and Dirac replay; these
numbers are JLArrays emulation results, not hardware GPU performance or implicit
device support. Both implicit linear-solver selections explicitly reject device
coordinates. Full package/build verification follows below.


The final source/note review also corrects two schedule descriptions: the
forward/backward generator accepts a tree fallback with inward/outward lists,
so only its DAG branch always has an empty backward list; flooding removes
message-order dependence for stateless factors, while Lux state still threads
in task order. The passing note now describes all three executors, including
residual priority updates. These are documentation corrections to the checked
implementation; no scheduling arithmetic changes.

The replay note also makes the existing state distinction explicit: raw
ChainRules inference detaches state after each task, whereas the parameter
driver includes functional state dependencies in its finite replay. No state
cotangents are returned by the raw rule. The implicit driver requires empty
message state, as tested above.


### Final batch C package verification (2026-10-28)

All seven complete package suites have passing final runs on Julia 1.13.1.
The interrupted ImplicitLayers run was restarted in full; no testset was
filtered. Suites ran sequentially, one Julia process at a time, with
`--startup-file=no --heap-size-hint=6G`, single-threaded BLAS and each package's
test environment. The top-level declared test target uses the disposable
`root-test` environment described in batch B. Times include loading and
compilation and are validation timings, not performance benchmarks.

| Package | Passed | Full-suite wall time | Sampled peak RSS |
|---|---:|---:|---:|
| LenticulumCore | 314 | 47.02 s | 0.933 GiB |
| Mycelium | 1,265 | 1367.48 s | 2.019 GiB |
| Lenticulum | 1,203 | 368.66 s | 1.864 GiB |
| ImplicitLayers | 5,308 | 1808.31 s | 6.805 GiB |
| VariationalDiffusion | 1,408 | 398.14 s | 1.909 GiB |
| Adversarial | 374 | 9.00 s | 0.416 GiB |
| AlgebraicFactors | 15,985 | 173.16 s | 1.606 GiB |

Total: **25,857 passed**, with no failures, errors or broken tests
in these final complete runs; summed suite wall time **4171.78 s**.
Mycelium adds 480 batch C assertions to the retained 785. ImplicitLayers retains
all 5,308 batch A/B assertions, including 3,414 device assertions.

The final standalone Core suite passes 314 assertions, and the separate
repository endpoint reproducer returns no failures in 20 repetitions in 5.00 s.
This does **not** repair or invalidate the earlier NaNs: the initial complete
suite failed, and the full rerun's appended original-fixture probe failed at
repetition 2. No assertion or backend was removed or relaxed, and no Core CPU
source fix was made. The recurring Enzyme endpoint issue remains open.
`LenticulumCore-initial-failure.log` and `LenticulumCore-repeated-failure.log`
retain the failures; `LenticulumCore-standalone.log` and
`core-endpoint-reproducer.log` distinguish the fresh passing runs.

`~/.julia/lenticulum-runs/lp002-batch-c-resume/verification.json` records the
seven final commands, counts, timings and sampled RSS.
`verification-with-repeat.json` preserves the earlier combined suite/diagnostic
exit status. The interrupted log and focused/native-backend/metric-divergence
probe logs are retained too. A Python status-reader race in the waiting wrapper
was corrected before starting the standalone checks; it did not overlap Julia
processes or affect a suite result. Documentation verification follows below.


### Batch C documentation review

The standalone vault review passes in **11.57 s**, rendering **255 HTML pages**
without KaTeX errors. The first API/vault review build passes in **467.69 s**
and regenerates all ten Literate tutorials. It exposes a newly referenced
pre-existing docstring-placement error: `tree_schedule`'s help was attached to
`_resolve_tree_root`. The docstring now precedes the public generator itself,
with both function bodies unchanged. Final builds below recheck that reference.
Documenter's core-page/search-index size warnings and local deployment skip are
expected. Both builds use the disposable Quartz copy specified above, with its
checkout/clean EXIT hook removed before use; source Quartz trees are untouched.

The corrected API/vault build passes in **483.28 s**, re-executing
all ten tutorials. The `tree_schedule` reference now resolves; no cross-reference
warning remains in this run. The post-fix standalone vault build passes in
**7.78 s**, with **255 pages** and no KaTeX errors. All API source/content is
final at this point; the last evidence-only vault-note changes are refreshed by
the standalone build, recorded in `vault-evidence.log`/`.json`. The disposable
Quartz copy is removed after that final build; logs and verification records
remain under the run directory.

`git diff --check` passes. The final review confirms that schedule source outside
docstrings is unchanged from HEAD. Batch A/B work is retained, no new paper or
contributor metadata is added, and no checkout, clean, reset, stash, commit, push
or merge command was run. Source Quartz checkouts and `meta/` are untouched.
The broader Remaining list stays open with the tested domains and compatibility
limits above; in particular, the earlier Core Enzyme NaNs are not claimed fixed.

## Enzyme endpoint NaN diagnosed (2026-10-28, Claude)

The intermittent NaN in `test/parameter_ad.jl` (`energy_jacobian(...; wrt = :y, ad =
AutoEnzyme())` against the identity) is an Enzyme forward-mode defect; Lenticulum code is
not involved. With Enzyme 0.13.216 (the newest registered release) and Julia 1.13.1,
`Enzyme.jacobian(Forward, (u, A, x) -> u .- A*x, y, Const(A), Const(x))` returned a
non-finite or wrong Jacobian in **139 of 2,000** calls, and the same call in reverse mode in
**0 of 2,000** (`lib/LenticulumCore.jl/examples/lp002_enzyme_endpoint_limit.jl`). Without the
constant product `A*x` (`u .- x`, `2u`, `copy(u)`, `u .- x .- 1`) forward mode gave
0 failures in 2,000 calls each, so the trigger is a constant matrix product inside a
forward-mode derivative. DI's `AutoEnzyme()` Jacobian takes the forward path, so
`energy_jacobian` inherited it. Passing the active vector without also passing it as a
constant context was tried first and did not help (still ~5%).

The Core test now computes its Jacobians with `AutoEnzyme(; mode = Enzyme.Reverse)`;
gradients keep `AutoEnzyme()`. In 1,500 reverse-mode `energy_jacobian` calls
(`wrt = :x, :y, :ps`) none was non-finite. No assertion was relaxed. Users computing
Jacobians with Enzyme should select reverse mode until this is fixed upstream.

Reported upstream as [EnzymeAD/Enzyme.jl#3884](https://github.com/EnzymeAD/Enzyme.jl/issues/3884)
(2026-10-28). A rerun for the report (scratch script, same versions, one thread) gave
141/2000 wrong forward Jacobians, and 0/2000 with the product written out by hand (no BLAS
call). Each wrong Jacobian had a single `NaN` where the tangent of the constant `A*x` belongs,
which suggests an uninitialised shadow of the product, the family of Enzyme.jl#3845.


## Device tests opt-in; review of batches A–C (2026-10-28, Claude)

Since commit 29cc5bdd, `lib/Mycelium.jl/test/graph_adjoint_devices.jl` and
`lib/ImplicitLayers.jl/test/device_arrays.jl` run only with `LENTICULUM_DEVICE_TESTS=true`.
In the 2026-10-28 run they took about 17 minutes together. The suite counts and times above
(25,857 assertions; Mycelium 1,265 in 1367 s) were measured with them included, so a
default `Pkg.test()` now runs fewer assertions.

A read-through of batches A–C found no defects. Checked by hand:
- the implicit graph adjoint, $(I-D_mT)^\top\lambda=\nabla_mL$ followed by $\partial_\theta[L+\lambda^\top T]$, including its invariance under damping;
- the limited Broyden update and its transposed history in `BroydenIFT`;
- flooding buffering and damping order in the replay, against `step!`/`sweep!`;
- the closed forms in `test/lp002_batch_a.jl` and in the tree/loop oracles of `test/graph_adjoint.jl`.

The numbers quoted in this note and in LP-011 match the logged test output.
