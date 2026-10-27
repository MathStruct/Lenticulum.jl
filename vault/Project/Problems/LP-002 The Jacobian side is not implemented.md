#open-problem

> **Status: first-order CPU DEQ IFT supports dense and opt-in matrix-free solves with ForwardDiff, Zygote, Enzyme and Mooncake (2026-10-26). SciML endpoint sensitivities cover ForwardDiff, Zygote and Mooncake with empty Lux state; built-in Euler/RK4 also cover runtime-activity Enzyme. Remaining scope below.** Related: [[energy]] §"Implementation difficulties" 6, [[Scalar and Multivariate Energy]] §6, [[L-ADR-0002 Energies are vectors, scalarised late]], [Fisher information metric (CT-ML wiki)](https://mathstruct.org/CategoryTheory-ML-Wiki/Fisher-Information-Metric).

> Sources: original implementation; Montoison and Orban (2023), *Krylov.jl*, p. 1; Rackauckas et al. (2020), *Universal Differential Equations for Scientific Machine Learning* (SciMLSensitivity software citation); code: `lib/LenticulumCore.jl/src/energy.jl`, `lib/LenticulumCore.jl/test/parameter_ad.jl`, `lib/ImplicitLayers.jl/test/parameter_ad.jl`, `lib/ImplicitLayers.jl/test/lp002_followups.jl`, `lib/Mycelium.jl/src/free_energy.jl`
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
limited-memory Broyden and Krylov preconditioning; nonempty Lux state and non-Tsit5
SciML sensitivity coverage, Enzyme SciML endpoint gradients; AD through divergence and finite-difference metric
localisation; higher-order IFT rules; reverse-input solve rules; GPU coverage for new
implicit rules; differentiating graph inference schedules, per-edge couplings and
stochastic inversion estimators. These limits do not prevent the tested endpoint
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
