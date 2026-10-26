#open-problem

> **Status: implemented for first-order core derivatives, DEQ state solves and built-in ODE flows (2026-10-26); remaining scope below.** Related: [[energy]] §"Implementation difficulties" 6, [[Scalar and Multivariate Energy]] §6, [[L-ADR-0002 Energies are vectors, scalarised late]], [Fisher information metric (CT-ML wiki)](https://mathstruct.org/CategoryTheory-ML-Wiki/Fisher-Information-Metric).

> Sources: original implementation; code: `lib/LenticulumCore.jl/src/energy.jl`, `lib/LenticulumCore.jl/test/parameter_ad.jl`, `lib/ImplicitLayers.jl/test/parameter_ad.jl`, `lib/Mycelium.jl/src/free_energy.jl`
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

DEQ `solve_state(...; ad)` uses IFT derivatives, a dense state linear solve, backend
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
  ForwardDiff and Zygote cover the implicit paths. Core also tests Enzyme; Mooncake is untested.
  Solved-state IFT currently rejects Enzyme/Mooncake explicitly, avoiding iteration AD.

Remaining: matrix-free state solves/limited-memory Broyden; first-class SciMLSensitivity
adjoints for `SciMLIntegrator`; AD through divergence and finite-difference metric
localisation; higher-order IFT rules; reverse-input solve rules; GPU coverage for new
implicit rules; differentiating graph inference schedules, per-edge couplings and
stochastic inversion estimators. These limits do not prevent the tested endpoint
objectives or custom scalar losses on `solve_state` from training.

## Validation run (2026-10-26)

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

## Files changed

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
