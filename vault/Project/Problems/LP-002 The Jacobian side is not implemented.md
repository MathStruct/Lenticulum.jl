#open-problem

> **Status: solver numerics implemented (2026-10-27): limited-memory Broyden, preconditioned Krylov, both solve polarities with first-order ForwardDiff/Zygote/Enzyme/Mooncake rules, and a dedicated dense second-order IFT adjoint. SciML endpoint sensitivities cover ForwardDiff, Zygote and Mooncake with empty Lux state; built-in Euler/RK4 also cover runtime-activity Enzyme. Remaining scope below.** Related: [[energy]] §"Implementation difficulties" 6, [[Scalar and Multivariate Energy]] §6, [[L-ADR-0002 Energies are vectors, scalarised late]], [Fisher information metric (CT-ML wiki)](https://mathstruct.org/CategoryTheory-ML-Wiki/Fisher-Information-Metric).

> Sources: original implementation; Montoison and Orban (2023), *Krylov.jl*, p. 1; Rackauckas et al. (2020), *Universal Differential Equations for Scientific Machine Learning* (SciMLSensitivity software citation); code: `lib/LenticulumCore.jl/src/energy.jl`, `lib/LenticulumCore.jl/test/parameter_ad.jl`, `lib/ImplicitLayers.jl/test/parameter_ad.jl`, `lib/ImplicitLayers.jl/test/lp002_followups.jl`, `lib/ImplicitLayers.jl/test/solver_numerics.jl`, `lib/Mycelium.jl/src/free_energy.jl`
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

Remaining: automatic outer-closure annotations;
nonempty Lux state and non-Tsit5
SciML sensitivity coverage, Enzyme SciML endpoint gradients; AD through divergence and finite-difference metric
localisation; second-order reverse-input/Krylov rules, direct input/parameter loss dependence and arbitrary native backend nesting; GPU coverage for new
implicit rules; broader graph schedule/backend coverage and stochastic solver inversion
estimators (static graph replay and terminal estimators implemented in LP-011, 2026-10-27). These limits do not prevent the tested endpoint
objectives or custom scalar losses on `solve_state` from training.

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
Anderson acceleration, retained Broyden preconditioners, scalable second-order
products, reverse-input second order and general backend nesting. GPU and complex
implicit differentiation remain untested. The broader LP-002 Remaining list above
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
