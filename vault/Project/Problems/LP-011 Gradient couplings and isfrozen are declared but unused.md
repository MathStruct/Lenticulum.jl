#open-problem

> **Status: option 1 extended to realised flooding/residual/damped replay and opt-in CPU implicit graph adjoints (2026-10-28); broader estimator/backend/device support remains.** Related: [[L-SPEC-002 Stop Gradient]], [[Factors are Parameterized Statistical Games]], [[The Type Discipline of a Factor Graph]] §4. Code: `abstract_types.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/graph.jl`.

> Sources: repository audit of `src/`, `lib/` (source, extensions, tests, examples), `test/`, `docs/literate/` and `bench/` on 2026-10-26; code: `lib/LenticulumCore.jl/src/abstract_types.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/graph.jl`, `polarity_resolution.jl`; tests: both packages' `test/runtests.jl`, the `LP-011` testsets.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

**Original problem (verified before the guards).** The design says the choice of gradient coupling is made per edge (`connect!(…; coupling = DiagonalCoupling())`, `ComposedFactor(…; coupling)`), and that a frozen factor (`isfrozen`) differs from a non-learnable one in the backward pass. Neither coupling field had a runtime reader; `isfrozen` had only its default definition and a top-level conformance assertion checking `Bool`. No production override returned true and no test/example/tutorial/benchmark selected a non-default coupling. A user setting `coupling = ScoreFunctionCoupling()` therefore got no error and no gradient effect.

**Options.**
1. Implement them as part of the graph-level backward pass ([[Backward Signatures]]) and the stop-gradient specification.
2. Until then, warn (or throw) when a non-default coupling is set, so the setting is not silently ignored.

## Option 2 implemented (2026-10-26)

- `connect!`, direct/parameterized `Edge`, and keyword/positional/parameterized
  `ComposedFactor` now throw `ArgumentError` for non-default couplings, explaining that the
  setting would be silently ignored and naming LP-011 / L-SPEC-002. Fields and types remain.
- `build` does not call `validate`; both now reject every graph factor declaring
  `isfrozen == true`, naming the factor. This includes `islearnable == false`: the existing
  optimiser-list filter does not implement parameter freezing with preserved input
  cotangents, and no graph backward path supplies that guarantee.
- Checks occur at construction/validation only; message passing is unchanged. Accepting
  `DiagonalCoupling()` preserves existing behavior rather than inserting AD stops.
- The two packages' `LP-011` testsets cover all three non-default built-in coupling families
  (`PathwiseCoupling`, `ScoreFunctionCoupling` with and without baseline, `ExactCoupling`),
  a custom subtype, constructor bypasses, independent build/validate rejection for both
  learnability values, errors naming the frozen factor, and default paths.
- No existing non-default tests or examples needed conversion: the audit found none.

**Still open.** Option 1, the actual gradient and freezing semantics, remains with
[[L-SPEC-002 Stop Gradient]].

## Option 1 implementation (2026-10-27)

The maintainer decisions are recorded in [[L-SPEC-002 Stop Gradient]]; the option-2
history above is retained. All four built-in coupling annotations are admitted. The
parameter driver now reads them, with explicit static-schedule replay for non-diagonal
edges and estimates with standard errors for terminal score edges. Unknown annotations
still throw. `detach` is per attachment, freezing is per factor parameter subtree, and
straight-through is a separate factor with ForwardDiff and ChainRules rules.

[[gradient_couplings]] gives the mixed Jacobian semantics, public keywords, estimator
objective and rejected combinations. `isfrozen` is accepted by build/validate and pins
parameters without suppressing upstream derivatives. Optimiser state contains only active
coordinates. The LP-008 conformance work present at task start is retained.

The closed-form tests exposed an existing terminology error: fixed cavity messages do
not imply fixed posterior beliefs. Today's diagonal derivative is retained and explicitly
labelled fixed-cavity. True posterior EM and exact evidence gradients agree at an exact
posterior. A collapsed Gaussian composite also differs semantically from an exposed
internal graph wire; its three analytic endpoint coupling choices coincide.

Remaining domains: flooding/residual replay, graph damping in reverse mode, other schedule
AD backends, broader device message algebra, nonterminal/bidirectional score sampling, stochastic solver inversion
estimators, and an implicit graph adjoint. Unsupported combinations raise ArgumentError;
this implementation does not claim those capabilities.

## Files for option 1

- Core: `Project.toml`, `src/LenticulumCore.jl`, `abstract_types.jl`,
  `statistical_game.jl`, `parameter_tree.jl`, their notes and `open_model.md`;
  new ChainRulesCore/ForwardDiff extensions; `test/runtests.jl` coupling cases.
- Mycelium: `Project.toml`, `src/Mycelium.jl`, `graph.jl`, `polarity_resolution.jl`,
  `messages.jl`, `passing.jl`, `schedules.jl`, `free_energy.jl`, new `gradient_couplings.jl` and notes;
  new ChainRules extension; test project, construction/conformance tests and new
  `test/gradient_couplings.jl`.
- Lenticulum: `src/gaussian.jl`, `composition.jl` and notes,
  `ext/LenticulumChainRulesCoreExt.jl`, `test/runtests.jl`, new
  `test/gradient_couplings.jl`.
- Documentation: package pages for Core/Mycelium/Lenticulum, Backward Signatures,
  Factors are Parameterized Statistical Games, L-SPEC-002, LP-011, LP-002,
  Problem Log, generated bibliography backlinks, and an appended L-SPEC-001 entry
  for the new structural factor.

The pre-existing LP-008 changes in other package tests, the shared checker,
`conformance_capture.jl`, LP-008 and its specification history are retained.

## Estimator measurements (2026-10-27)

`lib/Mycelium.jl/test/gradient_couplings.jl`, CPU, `Xoshiro(11)`, 4,000 draws
per estimate, Gaussian mean 0.8, unit variance and terminal cost `(x+0.2)^2/2`.
The exact mean gradient is 1.0. Common random draws give:

| Baseline | Estimated gradient | Standard error |
|---|---:|---:|
| none | 1.0309944917 | 0.0455415826 |
| fixed 2.0 | 0.9993480511 | 0.0317498956 |

The measured variance ratio is **0.4860359190**. This is a run measurement, not a
claim that every baseline improves variance. The categorical test (`Xoshiro(13)`,
4,000 draws) estimates `-2.2137735682 ± 0.0651344159` (one standard error), compared
with `-3 + log(0.7/0.3)`: expected point cost plus the toy graph's analytic Bethe
entropy correction. Both estimator means satisfy their stated standard-error tests.

`test/gradient_couplings.jl` samples the exact internal Gaussian posterior with
`Xoshiro(47)`, 3,000 draws. A fixed baseline equal to the reference collapsed NLL
(1.4130896953500565 in that fixture) cancels the posterior score term; its componentwise
standard errors are asserted below `1e-12`. The expectation gradient and analytic
endpoint gradient agree within the estimator's standard errors and numerical tolerance.

## Full package verification (2026-10-27)

Julia 1.13.1, CPU. Each command is
`julia --project=<package> -e 'using Pkg; @time Pkg.test()'` (top level: `--project=.`).
Counts sum the top-level test summaries. Times are the Pkg.test wall times, including
compilation; independent runs overlapped, so their sum is not elapsed task time.

| Package | Passed | Pkg.test time |
|---|---:|---:|
| LenticulumCore | 314 | 46.42 s |
| Mycelium | 785 | 195.21 s |
| Lenticulum | 1,203 | 353.58 s |
| ImplicitLayers | 1,231 | 481.71 s |
| VariationalDiffusion | 1,408 | 436.91 s |
| AlgebraicFactors | 15,985 | 170.76 s |
| Adversarial | 374 | 10.16 s |

**Total: 21,300 passed; no failures, errors or broken tests.**
The final root and Mycelium runs include explicit Zygote composite-score checks and a
common-random-number comparison of reference-difference and ForwardDiff score estimates.
The Mycelium tests also verify one committed state advance rather than one per replicate.
The inherited LP-008 tests are included in these counts.

## Documentation verification (2026-10-27)

The full `julia --project=docs docs/make.jl` refresh passed in **423.44 s**, executing
all ten Literate tutorials after the final schedule-docstring correction. The docs
environment was then resolved to register the new Core/Mycelium optional extensions;
Pkg reported no packages added or removed. A final cached API/vault rebuild passed in
**38.52 s**. Documenter reported only the core-page/search-index size warnings and
skipped deployment in the local environment.

The standalone `docs/site/build.sh` run passed in **7.76 s**, rendering **254 pages**
without KaTeX errors. A subsequent verification with the full test record passed in
**7.20 s**, again with 254 pages and no KaTeX errors. Both
commands set `QUARTZ_DIR` to a disposable copy whose checkout/clean EXIT hook was removed.
The source `docs/quartz` checkout and `meta/` were untouched; no git checkout, clean,
reset, stash or commit was run. The four bibliography scripts were run; no new work
was added to refs.bib, and only generated existing-work backlinks changed.
`git diff --check` passes.


### LP-002 batch A endpoint state policy (2026-10-27)

NeuralODE factors can opt into `state_policy=:fixed`: constant incoming Lux state
per solve and one final endpoint update. The existing diagonal graph parameter
gradient and `parameter_step` driver use the factor's policy and fixed state
subtree. `lib/ImplicitLayers.jl/test/lp002_batch_a.jl` checks these gradients
against a state-dependent affine closed form with ForwardDiff, Zygote, Mooncake
and runtime-activity Enzyme reverse. Default evaluation-state threading is
unchanged; state derivatives, schedule replay backends and graph adjoint scope
are unchanged. [[flow]] records the solver/backend restrictions.


## LP-002 batch B device subset (2026-10-27)

JLArrays tests with scalar indexing disabled extend finite static replay to
Float32/Float64 parameter arrays under ForwardDiff/Zygote. Dirac and diagonal
Gaussian messages cover exact/pathwise/diagonal boundaries, detach, subset/all
freezing, raw static inference AD and optimiser steps. Terminal diagonal-Gaussian
score estimates and standard errors compare with CPU using common explicit host
RNG draws. This adds no native GPU RNG, dense device Gaussian factorisation,
categorical device score rule, new schedule or new replay backend.

The fixes remove host scalar parameter gathers and incompatible diagonal
cotangents, and preserve Float32 in device optimiser coefficients. The quadratic
Gaussian oracle includes both expected variance and Bethe entropy terms.
Evidence: `lib/ImplicitLayers.jl/test/device_arrays.jl`; details and limits are in
[[gradient_couplings]]. CPU primal/default arithmetic is retained.


The same batch found that DEQ replay messages lacked explicit backend selection.
Replay now passes a constant backend context to ImplicitLayers' IFT message hook;
nonzero downstream DEQ/ODE graph losses are checked in both message directions.
Raw Zygote static inference uses the hook too. ForwardDiff implicit graphs should
use the explicit graph gradient API; raw `infer!` has no automatic IFT selection.
Nonempty-state reverse-input DEQ replay is explicitly rejected. See
[[gradient_couplings]] for this distinction from ordinary inference.

Raw ForwardDiff DEQ messages now reject dual-valued unannotated solves with an
ArgumentError pointing to the explicit graph gradient driver, preventing accidental
root-iteration differentiation. Ordinary primal messages are unchanged.


The resumed batch B CPU regression found and repaired backend-context dispatch
for GaussianFactor, GaussianPrior and GaussianComposite too. Their existing
functional inverse formulas are retained; the old hook signatures would instead
trace ordinary Cholesky inverses during replay. The closed-form top-level
marginal-likelihood oracle is included in the final full-suite rerun.


The final full ImplicitLayers run passes 5,308 assertions, including 3,414
JLArrays assertions across implicit rules, flows and graph paths. The largest
recorded vector errors scaled by `max(1,norm(reference))` are 3.0809478630544618e-6
(Float32) and 2.7084487851328933e-11 (Float64), within the tests' element-type
tolerances. These are emulator/reference measurements, with scalar indexing
disabled; hardware execution and the broader graph domains above remain open.
See LP-002 for full package/build verification and the measurement definition.


Final batch B verification passes all seven complete suites sequentially:
25,377 assertions, including the device coverage above. The full API/vault build
re-executes all ten tutorials and passes; the standalone vault also passes with
254 pages and no KaTeX errors. LP-002 records exact counts, timings, memory,
repaired regressions and the disposable-Quartz/log locations.


## LP-002 batch C: graph schedule adjoints (2026-10-28)

[[gradient_couplings]] now specifies finite flooding/residual/damped replay.
Residual task choices are recorded outside AD and replayed as a fixed sequence: a
piecewise realised-schedule derivative. Raw Zygote inference uses its actual
reference count/choices, including damping. The all-diagonal default is unchanged.

`free_energy_gradient` and `parameter_step` accept `adjoint=:implicit` with
reference convergence and dense/Krylov linear-solve controls. [[graph_adjoint]]
gives the message-space IFT equation and cut-row semantics. CPU Gaussian/Dirac,
Exact/Pathwise, empty message state, reference differences/ForwardDiff/Zygote are
the supported subset. Tree tests compare the dense Gaussian closed form and
converged exact unroll. Loopy tests compare reconverged Bethe differences and
measure convergence of finite unrolls. JLArrays replay covers the new schedules;
implicit device coordinates are explicitly rejected.

Native Enzyme/Mooncake replay probes and the divergence/metric interpretation
are recorded with reproducers in LP-002's batch C section. Terminal score and
straight-through restrictions stay as specified above. Tests are
`lib/Mycelium.jl/test/graph_adjoint.jl` and `graph_adjoint_devices.jl`.


The resumed batch C Mycelium suite passes 1,265 assertions, including 480 new
graph assertions (194 JLArrays), in 1367.48 s. The loopy unrolled-gradient error
falls from 0.1763879042 at four flooding sweeps to 2.526489452e-10 at 64.
Implicit gradients agree with the independent Gaussian/reconverged references
to a largest scaled error of 1.303263272e-9. The correlated 2D dense Gaussian
coordinate test and full ambiguity check also pass. Native Mooncake succeeds
on the minimal Dirac probe but fails on Gaussian Bethe entropy; Enzyme fails
with runtime activity. The public backend guard is retained. LP-002 records the
reproducers, remaining suite/build verification and the recurring Core Enzyme
endpoint-Jacobian NaN, which is not a repaired graph-adjoint defect.


Final batch C package verification passes all seven complete suites sequentially:
25,857 assertions, including Mycelium's 480 batch C checks and the retained
ImplicitLayers device coverage. The fresh Core suite and its 20-repeat standalone
endpoint probe pass, but earlier Enzyme NaNs remain recorded and unresolved; no
CPU backend fix or relaxed assertion is claimed. LP-002 records per-package
counts, timings, memory and the separate diagnostic logs.

Both requested documentation commands pass; the corrected API/vault build
regenerates all ten tutorials in 483.28 s, and the standalone vault renders
255 pages without KaTeX errors. Final evidence-only note edits are refreshed
in the vault before the disposable Quartz copy is removed. See LP-002 for logs,
measurements and the full batch C file list.
