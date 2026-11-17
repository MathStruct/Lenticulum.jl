#open-problem

> **Status: resolved (2026-11-18).** The maintainer's session reviewed the three commits a Mistral
> cloud agent added to PR #23 (`dev-danielb` → `dev`): the fixes for GitHub issues I15–I22
> (`74274922`) and two commits that repaired its tests after CI (`407a4196`, `9d94e548`). All three
> were accepted. One defect was found and fixed: the I21 rewrite of the `SampleBelief` moments
> refused weights given as a tuple, which AdversarialFactors passes. The I15 test now also checks
> that a second sweep changes nothing when the tree is rooted at the data.
> Related: [[Feature Backlog]], [[LP-054 Findings of the review of I24]],
> [[lib/LenticulumCore.jl/src/gaussian_belief|gaussian_belief]].

> Sources: a reading of the source diff of `74274922` and the test diffs of `407a4196` and `9d94e548`
> by the maintainer's session (Claude); the eight package suites on the rebased branch
> (`~/.cache/lenticulum/rebase/`), then the affected suites after the fix (`~/.cache/lenticulum/pr23/`).
> code: `lib/LenticulumCore.jl/src/gaussian_belief.jl` (`_samples_matrix`, `_moment_weights`);
> tests: `lib/LenticulumCore.jl/test/sample_moments.jl`, `lib/AdversarialFactors.jl/test/lp018.jl`,
> `lib/Lenticulum.jl/test/i15_nonlinear_tree.jl`.

**Problem.** The PR's other eleven commits are the maintainer's own, each reviewed when it was
committed. The cloud agent had no Julia, so its commit was tested only by CI, and CI does not run
the weekly packages on every push. Each of its commits was to be accepted, amended or rejected.

| # | commit / finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `74274922`, I21: `belief_mean`/`belief_cov` of a `SampleBelief` compute `S * w`, and `float.(b.weights)` keeps a tuple a tuple: `MethodError: *(::Matrix{Float64}, ::Tuple)`. AdversarialFactors' LP-018.9 test errored | yes: reproduced (AdversarialFactors suite, 2 errors) | `collect` the weights; the samples are stacked by `reshape` (scalars) or `reduce(hcat, …)` (vectors) instead of a splatted `hcat`, which compiles a method per sample count | tuple weights in `sample_moments.jl`; LP-018.9 |
| F2 | `74274922`, I20: `_plain_batch_arrays` requires `Array` with a float element type, so device and view inputs now also take the per-replica path, not only duals | not a defect: the per-replica path is the general one, and the batched kernel builds Float64 host arrays anyway. Possibly slower for plates on device | kept | `plate_block_duals.jl` |
| F3 | `74274922`, I15–I19, I22: read and judged correct. I16 refuses ties with differing fixed noise (conservative). I18 is reached only when both sites are flagged improper. I22's `which` uses the types that `em_step` passes | no defect | accepted | the agent's tests, which pass |
| F4 | `407a4196`: the sample-moment gradient oracle was wrong for `sum(C)`, now `tr(C)`, whose gradient is `2 wᵢ (xᵢ − m)`; the `exact_report` reason follows I15; the device covariance comparison and the P29 noise were corrected | right | accepted | — |
| F5 | `9d94e548`: rooted at `:data`, one sweep of the I15 tree is a true fixed point, because the backward message is sent after the cavity is known. The test now expects the marginal to move only for root `:prior` | right, but the `:data` branch checked only that x left its prior mean | added `m2 ≈ m1` for root `:data` | `i15_nonlinear_tree.jl` |

Verified after the fix: LenticulumCore, AdversarialFactors, Lenticulum, and Mycelium with
`LENTICULUM_DEVICE_TESTS=true` (I18/I19 on JLArrays) pass (`~/.cache/lenticulum/pr23/`).
MyceliumLearning, VariationalDiffusion, AlgebraicFactors and ImplicitLayers passed on the rebased
branch before the fix. The fix changes only how sample points are stacked and accepts tuple weights,
and these suites were not rerun after it. Not run: the vault build. A plate on device was not
benchmarked for F2.
