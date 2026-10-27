#open-problem

> **Status: resolved (2026-10-27), for the declared exact, approximate-message and deliberately non-game domains below. The 2026-10-23 resolution named a function that did not exist.** Related: [[L-SPEC-001 The Factor Interface]], [GitHub issue #8](https://github.com/MathStruct/Lenticulum.jl/issues/8).

> Sources: original to this vault; code: `test/factor_conformance.jl`, `test/game_composition.jl`, `test/conformance_capture.jl`; package `test/conformance.jl` files; `lib/Adversarial.jl/test/transport.jl`, `lib/AlgebraicFactors.jl/test/game_exact.jl`, `lib/VariationalDiffusion.jl/test/bound.jl`.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens).

**Problem.** The factor interface is tested factor by factor. Issue #8 was a mismatch between two parts of the contract (`scalarisation` and `energy`) that no test covered, because each factor's tests exercised only the paths it used. Other factor types (the diffusion, equilibrium, neural-ODE and adversarial factors) may have similar gaps.

**Options.**
1. A conformance test: a function `check_factor(f, ps, st, samples)` in a test utility that runs every applicable part of [[L-SPEC-001 The Factor Interface]] (§1–§5) and the consistency conditions between them, called from each package's tests for each factor type.
2. Keep per-factor tests and add the cross-checks by hand.

**Resolution through 2026-10-25 (history).** The `review-argon` commits claimed option 1 (`check_factor_conformance`) but only added a loop over traits; the function did not exist ([[LP-013 Defects found in the review of master]]). Since bb3af9e9 it does, in `test/factor_conformance.jl`: for a factor and polarity it checks the testable parts of [[L-SPEC-001 The Factor Interface]] §1–§5 (traits, `invert` = `combine(π, likelihood_message)`, the pointwise energy and its scalarisation, game-level `free_energy` and `entropy` in `energyspace`). It runs over every supported polarity of `GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor` and `LikelihoodFactor` (297 checks). Since 2026-10-25 the checker also has a direct-message path (`direct_message = true`, with independently computed site and posterior fixtures) for factors that implement `factor_message` without a lens; `AlgebraicFactors.ToricFactor` runs through it over every supported polarity at arities one to three. **Open as of 2026-10-25:** the factors of `Mycelium`, `ImplicitLayers`, `VariationalDiffusion` and `Adversarial` are not run through it yet.

## 2026-10-27: exact-domain conformance and complete subtype audit

This section supersedes the 2026-10-26 rollout table and its remaining-work list.
[[LP-009 Game-level free energy is declared but not implemented]] supplies the agreed
bounds-plus-exact-cases semantics; the common checker now exercises each implemented
exact domain without inventing a density, posterior entropy or wiring convention.

The default `scope=:wired` retains all existing checks and opt-out defaults. Independent
`expected_free_energy` and `expected_entropy` fixtures additionally compare individual
energy coordinates; wired runs also compare graph free energy with the expected value.
Gaussian distributional observations already fit `game_y`, so they need no special-case
factor dispatch. `scope=:game` checks parameters/states, game spaces, entropy/free-energy
coordinates and default/custom scalarisation, requiring both independent fixtures. It makes
no channel, pointwise, inversion or graph claim. `scope=:bound` requires full evidence,
API keyword arguments and an independently computed finite exact negative log evidence;
it checks an evaluated `FreeEnergyUpperBound`, finiteness, rejection by `scalarise`, and
`bound.value >= exact - 1e-10` (roundoff only). A sampled estimate cannot satisfy the type
check. Neither scope silently substitutes for a wired sink: both explicitly use `p=nothing`.

The Gaussian composition sites/posteriors now come from matrix Gaussian multiplication,
rather than the collapsed implementation's lens. Game fixtures use residual covariance
and Gaussian entropy, including the full joint posterior for soft observations. Tensor
fixtures check every nested branch, with GaussianFactor, GaussianPrior, collapsed pair
and independent product components. Polynomial fixtures use scalar Gaussian conditioning
for all four Dirac/Gaussian input/output pairs, in addition to both wired polarities,
row gauges, bases and damping. Plucker checks both placements of the Gaussian input.
Transport fixtures use the affine output covariance and standard-normal entropy. Diffusion
fixtures propagate the finite reverse chain's mean and variance independently, including
stationary equality, displaced-score strict gaps and nonstationary variances. Negative
checker tests reject a graded fixture with cancelling coordinate errors and a falsely
increased exact NLL; captured intentional failures are asserted, not counted as suite passes.

### Current rollout table

`lens` means the default assembled-lens/message/graph path, `direct` the independently
specified message/posterior path, `game` the explicit game-only scope, and `bound` the
explicit bound-only scope. Test locations identify the owning package or explicitly say top-level.
The table records fixture domains, not a claim about every belief family or dimension.

| Concrete factor | Checker path and fixture domain | Deliberate semantic boundary / test location |
|---|---|---|
| GaussianFactor | lens; both directions | Gaussian/Dirac incoming games; top-level `test/runtests.jl`; exact Gaussian reference |
| GaussianPrior | lens; unary; also tensor game component | Gaussian/point pooling; top-level `test/runtests.jl`, `test/game_composition.jl` |
| LinearConstraintFactor | lens; every supported fixture polarity | linear-Gaussian residual; top-level `test/runtests.jl` |
| NonlinearFactor | lens; both directions | declared Gauss–Newton message algorithm, not an exact nonlinear posterior; top-level `test/runtests.jl` |
| LikelihoodFactor | lens; both directions | declared quadrature/moment projection, not universal exact integration; top-level `test/runtests.jl` |
| ComposedFactor | lens; Gaussian pair both endpoint directions; game with Dirac and soft Gaussian observations | collapsed endpoint law, no internal-channel path game; unsupported components/nested compositions/noise hyperpriors explicitly rejected; `test/game_composition.jl`, Core scope tests |
| TensorFactor | game; independent Gaussian/point branches and nested products | independent game combinator, deliberately no graph/channel wiring or correlated-joint projection; `test/game_composition.jl`, Core scope tests |
| DataFactor | direct; unary | constant-zero clamped-source accounting; Mycelium `test/conformance.jl` |
| PriorFactor | direct; unary categorical, separate Dirac identity | arbitrary `nlogp` has no generic Gaussian expectation; Mycelium `test/conformance.jl` |
| LossFactor | wired sink; scalar/vector terminal point games | named observations have no canonical x/a/y split or outgoing message; pointwise/message gaps retained; Mycelium `test/conformance.jl` |
| RelayFactor | direct; both directions | identity point game; Mycelium `test/conformance.jl` |
| OptimiserFactor | direct; unary | stateful update rule, deliberately no pointwise/statistical game; pointwise/game gaps and rejection tests retained; Mycelium `test/conformance.jl` |
| LuxFactor | lens; forward | conditional point game, zero accounting; ImplicitLayers `test/conformance.jl` |
| DEQFactor | lens; both directions | incoming Dirac game; solved-point lens/anchor game separately checked; soft/singular anchors deliberately refused; ImplicitLayers `test/conformance.jl` |
| NeuralODEFactor | lens; both directions | incoming Dirac game; solved-point lens/anchor game separately checked (including SciML); same anchor boundary; ImplicitLayers `test/conformance.jl`, `test/sciml.jl` |
| NoiseSource | lens; unary fixed draw; game; unconditional standard normal at dimensions 1,3 | repeated stochastic particle draws are not identical-message fixtures; custom samplers provide no density law; Adversarial `test/conformance.jl`, `test/transport.jl` |
| GeneratorFactor | lens; forward point and affine Gaussian games | square nonsingular affine map, identity input adapter, no output conditioning; nonlinear certified bijections expose estimates only, singular maps no Lebesgue density; Adversarial `test/conformance.jl`, `test/transport.jl` |
| RatioFactor | lens; unary Dirac game | particle messages checked separately (LP-016); non-Dirac exact games lack analytic normaliser/reweighted entropy, empirical entropy not substituted; Adversarial `test/conformance.jl` |
| DiffusionFactor | lens; all five binary assignments and unary; bound; analytic Gaussian finite reverse chains | score/prox graph surrogate has no exact posterior game, so wired game gaps remain; full-evidence chain bound is a distinct model, not graph free energy; VariationalDiffusion `test/conformance.jl`, `test/bound.jl` |
| ToricFactor | direct; all supported polarities at arities 1–3 | normalised exact finite-state games; AlgebraicFactors `test/runtests.jl` |
| PluckerFactor | direct; forward point games at r=2,4 and scalar Gaussian images at r=2 | one fixed vector/one proper Gaussian, no output conditioning, positive variance, zero jitter; other images may be singular/non-Gaussian; AlgebraicFactors `test/runtests.jl`, `test/game_exact.jl` |
| PolynomialFactor | lens; both binary degree-one polarities; game; all four Dirac/Gaussian pairs | single branch, invertible output block, SPD residual covariance, proper pooled posterior; nonlinear/branch-mixture densities deliberately not inferred from local metrics; AlgebraicFactors `test/game_exact.jl` |

### Test-only concrete subtypes

These fixtures intentionally implement only the operation being tested; promoting them to
production statistical games would change the purpose of their tests.

| Concrete test fixture | Explicit reason no common full checker is claimed |
|---|---|
| CouplingGuardFactor | parameter/composition coupling rejection only; Core `test/runtests.jl` |
| CircleResidual | pointwise Sampson scalarisation/Jacobian residual only; Core `test/runtests.jl` |
| StatefulResidual | pointwise localisation/state-threading sentinel only; Core `test/runtests.jl` |
| BilinearResidual | mixed pointwise derivative fixture only; Core `test/runtests.jl` |
| LazyPolarityFactor | lazy enumeration/capped-list predicate fixture only; Core `test/runtests.jl` |
| ParameterResidual | parameter-AD fixture with a deliberately pointwise surrogate `free_energy`; no channels/posterior/entropy; Core `test/parameter_ad.jl` |
| FreezeTraitFactor | graph construction rejection/trait fixture only; Mycelium `test/runtests.jl` |
| GaugeResidual | pointwise Sampson gauge obligation only; AlgebraicFactors `test/runtests.jl` |

Audit command (run from the repository root, excluding local working material):

```sh
rg -n -U '^(mutable )?struct [^\n]*<:\s*(LenticulumCore\.)?AbstractLenticulum(Factor|ContainerFactor|WrapperFactor)' --glob '*.jl' --glob '!meta/**'
```

The grep found **22 production types and eight test-only types** (including multiline
supertype declarations and the container subtype). There are no concrete wrapper-factor
subtypes. Every match is accounted for above. The previous table omitted the five top-level
factors and ToricFactor; those omissions are corrected here.

**Remaining scope, deliberately excluded:** general ratio/reweighted entropy, nonlinear
transport exact expectations, singular or non-Gaussian Plucker images, nonlinear/multiple
polynomial branches, arbitrary/nested composition laws, tensor graph wiring/correlated
beliefs, and exact diffusion prox/posterior games or partial-evidence bounds. These are the
explicit semantic boundaries agreed in LP-009, rather than implemented APIs awaiting checker
coverage. All implemented new exact domains now reach the common checker. Existing callers
had no obsolete game/message keyword exemptions for those domains; the obsolete exemptions
were the historical rollout claims. The only real game/message opt-outs left are Optimiser's
non-game, Loss's terminal message and the diffusion surrogate game, with their reasons above.
LP-008 is therefore resolved; extending those semantics would require new laws and fixtures,
not removal of an unexplained checker exemption.

### Verification, 2026-10-27

All seven full package suites passed, with no failures, errors or broken assertions.
Commands were `julia --project=<package> -e 'using Pkg; Pkg.test()'`, using `.` for
Lenticulum and `lib/<Package>.jl` for the others, on Julia 1.13.1 with
`OPENBLAS_NUM_THREADS=1` and `JULIA_NUM_PRECOMPILE_TASKS=2`. Counts sum the package
entry points' top-level test summaries once; elapsed wall times include package setup
and compilation. Some commands ran concurrently, so these are validation timings,
not comparative benchmarks.

| Environment | Passing assertions | Elapsed seconds |
|---|---:|---:|
| Lenticulum | 1143 | 233.48 |
| LenticulumCore | 316 | 47.12 |
| Mycelium | 518 | 11.56 |
| ImplicitLayers | 1231 | 461.65 |
| Adversarial | 374 | 9.65 |
| AlgebraicFactors | 15985 | 166.81 |
| VariationalDiffusion | 1408 | 433.85 |

Total: **20975 passing assertions**. The affected exact-domain
testsets contribute 202 (composition/products), 108 (transport/source), 1411
(polynomial), 99 (Plucker) and 265 (diffusion bounds) assertions. These counts include
existing LP-009 checks as well as the extended common-checker calls; they are not
counts of newly added assertions alone. The bound fixture covers stationary equality
and strict/nonstationary gaps. Intentional checker-rejection tests assert exactly
one captured failure each and reject unexpected errors.

`julia --project=docs docs/make.jl` also passed in **41.32 seconds**, producing
26 API HTML pages and 253 vault HTML pages with cached tutorial renderings. It used
a disposable Quartz engine under `/tmp` with the Git restoration hook removed;
`QUARTZ_DIR` selected that engine and its npm cache was also in the disposable directory.
The original Quartz script, config, layout and static files retain their pre-build
SHA-256 hashes. Deployment was skipped by Documenter in the local environment.

The standalone `docs/site/build.sh` run with the same explicit `QUARTZ_DIR` passed in
**7.25 seconds**, producing **253 HTML pages** without KaTeX errors. A final rebuild
checks this completed verification record. `git diff --check` passes. No checkout,
clean, reset, stash or commit command was run, including in the disposable engine;
its restoration hook was removed before either build. No local working material was touched.

## 2026-10-26 rollout (history)

The single checker stays in `test/factor_conformance.jl` and is included by relative path
from each package. This is the least invasive reuse: test environments already have its
libraries; Mycelium's test project only needs direct LuxCore/Random visibility. It adds
nonempty reason-string opt-outs `pointwise_gap`, `game_gap`, `message_gap`, `polarity_gap`, plus `p=nothing`
for sinks. Defaults retain all checks, and game/graph agreement now compares every energy
coordinate, not just their scalar sum. Independent site/posterior fixtures also work on
the default lens path.

| Factor | Checker path and polarities | Explicit scope/gap |
|---|---|---|
| DataFactor | direct; unary | constant-zero point/source game |
| PriorFactor | direct; unary | categorical closed-form fixture, separate Dirac option-1 check; arbitrary Gaussian expectations still refused |
| LossFactor | sink (`p=nothing`) | scalar/vector terminal games; no message, lens or canonical pointwise split |
| RelayFactor | direct; both directions | identity point game |
| OptimiserFactor | direct; unary | no pointwise energy or statistical game; game methods throw |
| LuxFactor | default lens; forward | conditional point game with zero accounting |
| DEQFactor | default lens; both directions | full incoming Dirac game; solved-point lens game and Jacobian-anchor identity checked separately (LP-009 item 1 resolved) |
| NeuralODEFactor | default lens; both directions | full incoming Dirac game; solved-point lens game and both zero-flow identities checked (LP-009 item 1 resolved) |
| NoiseSource | default lens; unary | injected fixed one-particle draw; game conditioned on a Dirac draw, no cloud entropy |
| GeneratorFactor | default lens; forward | Dirac latent fixture, no distributional entropy |
| RatioFactor | default lens; unary | Dirac game; particle likelihood, posterior energy contribution, and two-ratio graph checked (LP-016 resolved) |
| DiffusionFactor | default lens; all five accepted binary role assignments and unary | fixed-node Gaussian/point fixtures and joint messages; game gap remains; LP-017 enumeration/joint inversion resolved; other backends not certified |
| PluckerFactor | direct; sole forward polarity at r=2,4 | trivially extended Dirac-input game; no assembled lens/inverse fibre/distributional entropy |
| PolynomialFactor | unchanged independent per-polarity tests | no local/game free energy or normalised joint branch model; full checker inapplicable |
| ComposedFactor, TensorFactor | checker inapplicable; Core tests verify scope | parameter/energy-space containers without channels, inversion or evaluated game laws |

**Remaining as of 2026-10-26 (superseded above):** particle game entropy for Ratio (LP-006/009); diffusion game contracts
(LP-009); normalized branch/game laws for Polynomial; wired container semantics.
The rollout tests do not establish conformance for arbitrary belief families or diffusion
backends. No placeholders or new approximation have been introduced for these gaps.


A small DEQ trait defect was fixed during the polarity audit: latent/unobserved pairs were
accepted although neither solve has an observed companion point. `supports_polarity` now
requires one observed channel; two regression assertions cover the rejected splits.


## Earlier validation run, 2026-10-26 (history)

All commands below completed with zero unexpected failures/errors. Broken assertions are
intentional executable records of the conflicts above, not passing conformance claims.

| Environment | Pass | Broken |
|---|---:|---:|
| LenticulumCore | 216 | 0 |
| Mycelium | 459 | 0 |
| Lenticulum (top level) | 775 | 0 |
| ImplicitLayers | 312 | 4 |
| Adversarial | 232 | 3 |
| AlgebraicFactors | 14396 | 0 |
| VariationalDiffusion | 1066 | 4 |

Each package command was `julia --project=lib/<Package>.jl -e 'using Pkg; Pkg.test()'`;
the top level used `julia --project=. -e 'using Pkg; Pkg.test()'`. VariationalDiffusion's
full suite ran once after the changes (its original testset: 889 passes in 6m48s; added
partial conformance: 177 passes, 4 broken). Its touched testset also ran separately as a
script containing the header of `test/runtests.jl` plus `include("conformance.jl")`, with
`--project=lib/VariationalDiffusion.jl/test`: 177 passes, 4 broken. ImplicitLayers' touched
testset was also run as a header-plus-testset script while iterating; final full-suite
counts above include the subsequent domain-rejection tests.

Broken counts: DEQ/NeuralODE incoming-versus-solved energy (four); Ratio particle pooling,
target exclusion and expectation mismatch (three, LP-016); Diffusion enumeration and joint
posterior dimension (four, LP-017).

The diffusion counts above record the earlier LP-008 run. LP-017 subsequently replaced
its four broken assertions with passing checks and added complete enumeration, a lazy
iterator, structural predicates and joint point/Gaussian/mixture tests; see
[[LP-017 DiffusionFactor omits supported polarities]]. Ratio's three broken assertions
were later replaced by LP-016 (below). At the LP-017 run, ImplicitLayers recorded the
incoming/solved energy mismatch as passing inequality checks. LP-009 item 1 now replaces
those checks with two explicit games and the identity linking them.

## Files in the 2026-10-26 rollout (history)

Source and test changes (abbreviated paths on a package line are relative to that package):

- `test/factor_conformance.jl`
- `lib/Mycelium.jl/src/factors.jl`, `test/Project.toml`, `test/runtests.jl`, `test/conformance.jl`
- `lib/ImplicitLayers.jl/src/luxfactor.jl`, `src/deq.jl`, `test/runtests.jl`, `test/conformance.jl`
- `lib/Adversarial.jl/src/generator.jl`, `src/ratio.jl`, `test/runtests.jl`, `test/conformance.jl`
- `lib/AlgebraicFactors.jl/src/plucker.jl`, `test/runtests.jl`
- `lib/VariationalDiffusion.jl/test/runtests.jl`, `test/conformance.jl`
- `lib/LenticulumCore.jl/test/runtests.jl`

Companion notes: Mycelium `factors.md`; ImplicitLayers `luxfactor.md`, `deq.md`,
`neuralode.md`; Adversarial `generator.md`, `ratio.md`; AlgebraicFactors `plucker.md`,
`polynomial.md`, `AlgebraicFactors.md`; VariationalDiffusion `factor.md`; LenticulumCore
`statistical_game.md` (all under their package `src/`). Package pages under
`docs/src/packages/`: `mycelium.md`, `implicitlayers.md`, `adversarial.md`,
`algebraicfactors.md`, `lenticulumcore.md`, `variationaldiffusion/factors.md`.
Project records: LP-008, LP-009, new LP-016 and LP-017, Problem Log, and L-SPEC-001.
The existing LP-011 implementation is preserved.

LP-016 subsequently replaced its three broken assertions with passing checks: RatioFactor
now emits a log-potential, excludes target inputs, and averages its graph energy under the
reweighted cavity. The two-ratio graph test checks multiplication once per factor. Particle
game entropy remains unsupported; the earlier counts above record the pre-LP-016 run.


## LP-009 item 1 validation, 2026-10-26

The incoming factor and solved-point lens games now have separate methods; DEQ/NeuralODE
run the full common game checker without exemptions in both polarities. Linear tests derive
`1/8` (DEQ) and `2` (zero flow) using `energy_jacobian` to form the Gaussian anchor precision.
A quadratic DEQ checks cubic absolute energy error for the Gauss–Newton approximation.
Matrix anchors, custom channel names, stopped solves, reverse-flow residuals and state
threading have separate checks in `lib/ImplicitLayers.jl/test/conformance.jl`.

All seven full package suites passed with no broken assertions, using the package commands
above and their repository `test/runtests.jl` entry points:

| Environment | Pass |
|---|---:|
| LenticulumCore | 236 |
| Mycelium | 494 |
| Lenticulum (top level) | 775 |
| ImplicitLayers | 480 |
| Adversarial | 254 |
| AlgebraicFactors | 14396 |
| VariationalDiffusion | 1143 |
