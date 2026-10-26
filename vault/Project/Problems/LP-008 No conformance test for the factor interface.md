#open-problem

> **Status: open, rollout extended with explicit gaps (2026-10-26). The 2026-10-23 resolution named a function that did not exist.** Related: [[L-SPEC-001 The Factor Interface]], [GitHub issue #8](https://github.com/MathStruct/Lenticulum.jl/issues/8).

**Problem.** The factor interface is tested factor by factor. Issue #8 was a mismatch between two parts of the contract (`scalarisation` and `energy`) that no test covered, because each factor's tests exercised only the paths it used. Other factor types (the diffusion, equilibrium, neural-ODE and adversarial factors) may have similar gaps.

**Options.**
1. A conformance test: a function `check_factor(f, ps, st, samples)` in a test utility that runs every applicable part of [[L-SPEC-001 The Factor Interface]] (§1–§5) and the consistency conditions between them, called from each package's tests for each factor type.
2. Keep per-factor tests and add the cross-checks by hand.

**Resolution through 2026-10-25 (history).** The `review-argon` commits claimed option 1 (`check_factor_conformance`) but only added a loop over traits; the function did not exist ([[LP-013 Defects found in the review of master]]). Since bb3af9e9 it does, in `test/factor_conformance.jl`: for a factor and polarity it checks the testable parts of [[L-SPEC-001 The Factor Interface]] §1–§5 (traits, `invert` = `combine(π, likelihood_message)`, the pointwise energy and its scalarisation, game-level `free_energy` and `entropy` in `energyspace`). It runs over every supported polarity of `GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor` and `LikelihoodFactor` (297 checks). Since 2026-10-25 the checker also has a direct-message path (`direct_message = true`, with independently computed site and posterior fixtures) for factors that implement `factor_message` without a lens; `AlgebraicFactors.ToricFactor` runs through it over every supported polarity at arities one to three. **Open as of 2026-10-25:** the factors of `Mycelium`, `ImplicitLayers`, `VariationalDiffusion` and `Adversarial` are not run through it yet.


## 2026-10-26 rollout

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
| DEQFactor | default lens; both directions | game gap: incoming residual differs from solved-point energy, asserted by a test |
| NeuralODEFactor | default lens; both directions | same game gap, independently tested with a constant flow |
| NoiseSource | default lens; unary | injected fixed one-particle draw; game conditioned on a Dirac draw, no cloud entropy |
| GeneratorFactor | default lens; forward | Dirac latent fixture, no distributional entropy |
| RatioFactor | default lens; unary | Dirac prior only; particle contract fails, [[LP-016 RatioFactor emits a particle posterior as a likelihood]] |
| DiffusionFactor | default lens; all five accepted binary role assignments and unary | fixed-node Gaussian/point fixtures and joint messages; game gap remains; LP-017 enumeration/joint inversion resolved; other backends not certified |
| PluckerFactor | direct; sole forward polarity at r=2,4 | trivially extended Dirac-input game; no assembled lens/inverse fibre/distributional entropy |
| PolynomialFactor | unchanged independent per-polarity tests | no local/game free energy or normalised joint branch model; full checker inapplicable |
| ComposedFactor, TensorFactor | checker inapplicable; Core tests verify scope | parameter/energy-space containers without channels, inversion or evaluated game laws |

**Remaining:** full particle conformance for Ratio (LP-016); DEQ/ODE and diffusion game
contracts (LP-009); normalized branch/game laws for Polynomial; wired container semantics.
The rollout tests do not establish conformance for arbitrary belief families or diffusion
backends. No placeholders or new approximation have been introduced for these gaps.


A small DEQ trait defect was fixed during the polarity audit: latent/unobserved pairs were
accepted although neither solve has an observed companion point. `supports_polarity` now
requires one observed channel; two regression assertions cover the rejected splits.


## Validation run, 2026-10-26

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
remain. The current ImplicitLayers checker records the incoming/solved energy mismatch
as a passing inequality check; its suite has no broken assertions in the LP-017 run.

## Files in this change

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
