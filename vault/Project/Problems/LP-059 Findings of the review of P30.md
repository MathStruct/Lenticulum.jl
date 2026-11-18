#open-problem

> **Status: resolved (2026-11-18).** The review of P30 (Lenticulum's linear-Gaussian factors in the
> element type of their parameters) found no defect in its three recorded probes; of its two
> unrecorded probes, run by the maintainer's session, one showed a defect: an Int `df` of
> `WishartPrecisionPrior` widened a Float32 prior to Float64. Fixed, with a test. Before the review,
> the session had found and fixed four Float64 sources in LenticulumCore and one in Mycelium that the
> implementation had left as broken tests.
> Related: [[Feature Backlog]] (P30), [[LP-058 Findings of the review of P8]],
> [[lib/Lenticulum.jl/src/gaussian|gaussian]] "Element type".

> Sources: implemented by Codex (GPT, `codex exec -C ../Lenticulum.jl-codex`, branch `codex/p30`),
> completed by the maintainer's session (Claude: the LenticulumCore and Mycelium sources);
> reviewed by Mistral Large 4 (`vibe --agent review` in `../Lenticulum.jl-mistral`, \$0, stopped at
> its 100-turn limit). Brief, probes and answer: `../Lenticulum.jl-Review-2026-11-18/p30/`. code:
> `lib/Lenticulum.jl/src/gaussian.jl`, `lib/Lenticulum.jl/src/constraint.jl`,
> `lib/Lenticulum.jl/src/hard_constraint.jl`, `lib/Lenticulum.jl/src/noise.jl`,
> `lib/LenticulumCore.jl/src/extended_gaussian.jl`, `lib/Mycelium.jl/src/free_energy.jl`,
> `lib/Mycelium.jl/src/gaussian.jl`; tests: `lib/Lenticulum.jl/test/p30_element_type.jl`.

**Problem.** The requirement given to the reviewer: with Float32 parameters, noise, priors and data,
every message, marginal and free-energy term of `GaussianFactor`, `GaussianPrior`,
`LinearConstraintFactor` and `HardConstraintFactor` is Float32 or `false`, on single nodes, plates,
mean-field nodes and with learnable noise; no conversion (promotion decides); Float64 results
unchanged; ForwardDiff, Zygote and Int inputs keep working; the rank memo holds Float32 safely.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| S1 | (the session, before the review) Codex's test left 218 checks `@test_broken`: extended messages and hard-constraint messages widened in LenticulumCore (`_conjoin` took `sqrt` of Int weights, `preimage` floated an Int offset, `extended(::TrivialBelief, n)` was Float64, `_logsum` and `_scale` started from `0.0`, `marginalise` built a Float64 selection), and Mycelium's `variable_corrections` turned a point's `false` entropy into the Int `0` (`-false === 0`) | yes, 218 broken checks | fixed in LenticulumCore and Mycelium; the broken markers became tests | `p30_element_type.jl`, all checks strict (2121 pass) |
| R1 | probe 1: loopy Float32 graph with extended messages and damping `0.5f0` stays Float32, agrees with Float64 to 1e-7; a Float64 damping weight widens by promotion | no defect (promotion, requirement 2) | recorded in `extended_gaussian.md` "Element type" | probe `probe1_loopy_f32.jl` |
| R2 | probe 2 (run by the session): Float64 trees printed bit for bit under the old (`3d0bc449`) and the new code | no defect: every number identical; only exact zeros `0.0` became `false` (`0` for the Int-zero case), as designed | — | `probe2_f64_same.jl`, `out_old.txt`, `out_new.txt` |
| R3 | probe 3 (run by the session; it needed one import repaired): `WishartPrecisionPrior(4, Float32 scale)` stores `df = 4.0` (Float64), so a Float32 prior's complexity is Float64 | yes, reproduced (`probe3.out`: "float32 wishart int-df complexity: … Float64") | an Int `df` takes the scale's float type (`noise.jl`) | `p30_element_type.jl`, "P30 exact integer inputs" |
| R4 | probe 3: the message from a Float32 prior to the factor stays Float32 in a graph with Float64 data | no defect (that message does not read the data; the reviewer's check was too strict, as it noted) | — | — |

Its probe 3 ended in an error of its own (`tree_schedule` on a graph with a loop); the Int-`noise` case it
meant to check was not run. An Int `noise` gives a Float64 graph (`float(Int)`), by promotion.

Not reviewed: the fixes S1 and R3 were not sent to a second review.
