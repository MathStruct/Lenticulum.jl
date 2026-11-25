#problem

> **Status: closed (2026-11-25).** Q4 (element type from the input; Codex) and what its review found. Mistral
> found three places where a Float32 call still returned Float64 arrays, all right and fixed with regression tests:
> GaussianPrior's messages with a learned covariance kept the stored mean, its pointwise energy with a learned
> covariance read the stored covariance, and integer offsets were widened by `float.`. The session found a fourth:
> both `mixture_message` methods in `switching.jl` were not converted at all. The session's benchmark found Zygote
> 16 % slower through the Kalman chain, fixed by a rule for `_covariance_at`. The shared element-type test checks
> `factor_message`, `energy` and `local_free_energy` only, which is why none of the four showed there; the other
> message paths are row Q19. Related: [[Engineering Standard]], [[Backlog Q Quality and Engineering]] (Q4, Q19),
> [[Feature Backlog]].

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort, 2 h 8 min; brief
> `~/.cache/lenticulum/codex-queue/q4-brief.md`, answer and review brief in `../Lenticulum.jl-Review-2026-11-25/q4-eltypes/`,
> logs in `scratch/q4/` of `../Lenticulum.jl-codex`). Mistral Large (`vibe --agent review`, \$0, 100 turns, three
> findings recorded, probes 3–5 written but not recorded; the session ran them: `scratch/q4r/` of
> `../Lenticulum.jl-mistral`). code: `lib/LenticulumCore.jl/src/gaussian_belief.jl` (`_compute_eltype`,
> `_at_eltype`, `_zeros_like`), `lib/LenticulumCore.jl/test/support/eltype_matrix.jl`, `lib/*/test/eltypes.jl`,
> `lib/Lenticulum.jl/test/lp091.jl`.

## What Q4 does

The working element type of a factor call is the promotion of the element types of its parameters, its incoming
messages and its observed inputs (`LenticulumCore._compute_eltype`; integers and empty containers do not count).
Allocations take it (`_zeros_like`, `similar`), stored constants are converted where used (`_covariance_at` in
`gaussian.jl` for the covariances of every Gaussian-family factor, the noise priors' hyperparameters, the
likelihoods' cutpoints and scales), and precision-tied tolerances follow it. In LenticulumCore, Lenticulum and
AlgebraicFactors, 47 source files changed. A shared test (`eltype_matrix.jl`) runs every concrete factor type,
found by reflection, in Float32, Float64, BigFloat and first-order ForwardDiff duals against the Float64 call, and
checks that every returned array has exactly that element type. Float32 parameters with Float64 messages are
checked on the free energy. Lenticulum: 326 pass, 35 broken (slow tier, 1 min 18 s); AlgebraicFactors: 55 pass,
3 broken; LenticulumCore: 3 pass. The broken ones are all BigFloat through a decomposition that has only a LAPACK
method (eigen, SVD, the Gauss–Hermite tridiagonal eigensolver); each has its reason in `test/eltypes.jl`.

Float64 results are bitwise unchanged: Codex recorded the 99 operations of the fixtures before the change and
compared after (`scratch/q4/record.jl`); the session reran it after its fixes (99/99). Codex's benchmark
comparison of LenticulumCore and Lenticulum against `dev-danielb` had no flag.

One deliberate change in behaviour: a Gaussian message whose only dual input is an offset now has a dual
precision with zero partials, where the precision stayed primal before (`test/forms_crossing.jl` changed
accordingly; a mixed belief, dual η and primal Λ, is still fed to the consumers on half the trials).

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| 1 | GaussianPrior's messages (`factor_message`, `likelihood_message`, both `factor_messages`, `_gradient_message`) build `Gaussian(f.μ, Σ)` with the stored mean: with a learned covariance and Float32 parameters, η is Float64 and Λ Float32 | right (probe 1; `lp091.jl` "finding 1" failed) | `_prior_mean` converts the mean to the covariance's type when the covariance is learned; with a fixed covariance there is no numeric input and the stored types stay |
| 2 | GaussianPrior's `energy` with a learned covariance passes the stored `f.Σ` to `_noise_metrics`, so a Float32 call returns Float64 | right (probe 1; "finding 2" failed) | passes the converted `Q` |
| 3 | `float.(ps.c)` and `float.(ps.b)` widen an integer offset to Float64 in a Float32 call (`LinearConstraintFactor` messages; the same in `gaussian.jl`'s flat path and `switching.jl`) | right (probe 1; "finding 3" failed). Integer offsets come only from a caller's own parameters | `_offset_at(c, Q)` (`constraint.jl`) converts to the promotion of the offset's and the covariance's types, at all twelve sites; the two `residual` functions convert to the call's type |

Mistral's other probes found no defect: dual preservation and gradients against central differences on the
changed paths (probe 2, including `_poisson_upper_tail`'s first-order expansion and a nested-dual Hessian of an
offset); BigFloat and mixed message paths (probe 3); the stored covariance rounded to the parameters' type
(probe 3b, as rule 1 says); inputs with no numeric leaves, `Bool` and `Int` leaves, plates, `SampleBelief`,
Zygote against ForwardDiff (probe 4); the three new test files as scripts (probe 5). Probe 4 also showed that
Float32 messages lose accuracy as the covariance's condition number grows (0.35 relative at $10^5$), which is
Float32's precision, not a defect, and that a dual hidden in a `Ref` is not seen by `_numeric_eltype` (no factor
input holds one).

## Found by the session

- **`mixture_message` not converted.** `SwitchingConstraintFactor`'s and `LinearConstraintFactor`'s
  `mixture_message` (`switching.jl`) passed the stored `f.Q` and `eltype(f.Q)`: a Float32 call returned Float64
  (`lp091.jl`, the fourth testset, failed). Both use `_covariance_at` and `_offset_at` now.
- **Coverage of the element-type test.** It runs `factor_message`, `energy` and `local_free_energy`. The other
  message paths (`mixture_message`, `mean_field_message`, the plate methods `factor_messages`, `factor_belief`,
  `_gradient_message`) are not run in other types; all four defects sat there or in fixtures it does not build
  (a learned prior covariance, integer parameters). Row Q19.
- **Speed.** Codex's benchmark ran Float64 only and covered LenticulumCore and Lenticulum. The session compared
  MyceliumLearning and Lenticulum against `dev-danielb` (`benchmark/compare.jl`, one thread). ForwardDiff through the
  Kalman chain: unchanged (0.988, then 1.001 interleaved), so the dual precision above costs nothing measurable there.
  Zygote: `kalman/zygote_gradient` 1.195, then 1.163 interleaved, with 41,430 → 43,819 allocations per gradient:
  Zygote traced `_covariance_at`'s type selection on every message. A rule for it
  (`ext/LenticulumChainRulesCoreExt.jl`: the further arguments get no tangent, the covariance the projected
  cotangent) brings it to 1.016 and 41,499 allocations. `regime/mstep`'s flagged improvement (0.547) is noise: its
  code did not change. Reports in `scratch/q4/lp091/bench-*.md` of `../Lenticulum.jl-codex`.

## Gates (2026-11-25, this machine, `JULIA_NUM_THREADS=auto`, `Pkg.test()`)

Codex, on its version: LenticulumCore, Lenticulum (both tiers), AlgebraicFactors, Mycelium, MyceliumLearning,
VariationalDiffusion, AdversarialFactors, ImplicitLayers passed (`scratch/q4/*.log`, each ending in Aqua with no
failure). The session, after its fixes and again after the Zygote rule (Lenticulum's sources and its ChainRules
extension only): Lenticulum fast tier, slow tier (`LENTICULUM_SLOW_TESTS=only`) and AlgebraicFactors passed
(`scratch/q4/lp091/`, `scratch/q4/lp091b/`); `lp091.jl` 10/10; Float64 still 99/99 bitwise (recorder run after the
fixes, before the rule, which changes no forward value).

Not run: device tests (Q7), Julia 1.12, the docs builds after the session's note edits (Codex built both on its
version), the suites of Mycelium, MyceliumLearning, VariationalDiffusion and ImplicitLayers after the session's
fixes (they do not depend on Lenticulum).
