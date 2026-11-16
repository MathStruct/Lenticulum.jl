#open-problem

> **Status: resolved (2026-11-16).** The review of P25 (ForwardDiff through a relation factor's flat
> path when the flat directions move, at constant rank) found one defect: nested duals reached
> `_range_complement` and failed with a `MethodError` instead of the `ArgumentError` the requirement
> asks for. Fixed, with a regression test. A suspected defect in the covariance was the reviewer's
> own wrong identity.
> Related: [[Feature Backlog]] (P25), [[lib/Lenticulum.jl/src/gaussian|gaussian]] "Flat directions that move".

> Sources: a review of the uncommitted P25 change by Mistral (`vibe --agent review`, `--max-price 3`,
> $1.18 at list price), in the helper worktree `Lenticulum.jl-mistral` at `8a6a9f41` with the diff
> applied; the brief, the probes and the answer file in `../Lenticulum.jl-Review-2026-11-16/p25/`.
> code: `lib/Lenticulum.jl/src/gaussian.jl`, `src/constraint.jl`, `src/switching.jl`, `ext/LenticulumForwardDiffExt.jl`; tests: `lib/Lenticulum.jl/test/forms_crossing.jl`.

**Problem.** The requirement given to the reviewer: under first-order ForwardDiff, the flat path's
messages are differentiable when a flagged input's precision is dual or the coefficients are dual,
equal to central differences of the value path at constant rank; a tangent that changes the rank at
first order, and nested duals, throw an `ArgumentError`; Float64 values unchanged.

The run stopped at its 100-turn limit with one entry in its answer file. The probes it left were run
by the maintainer's session (`probe2_v3.jl`, `probe4_edge_cases_fixed.jl`, `probe6_final_check2.jl`,
`probe7_nested_duals.jl`, the last repaired as `probe7_repaired.jl`: its line 44 multiplied a
5×2 by a 2×5 matrix transposed).

| # | probe | right? | what was done | test |
|---|---|---|---|---|
| 1 | `_moving_flat_split`: constant-rank tangent accepted, rank-changing refused, nested duals refused | passed | — | P25 refusal testset |
| 2 | `_range_complement`: constant rank, rank change, equal columns separated or kept parallel, log pdet against central differences | passed | — | P25 refusal testset |
| 4 | full-rank Λ dual, Λ = 0 with a rank-changing tangent, η outside range(Λ), values equal to `_flat_split` | passed | — | `probe4_edge_cases_fixed.jl` only (full-rank and η cases) |
| 6 | "Σ is √Λ⁺": the probe tested ΣΣᵀ = Λ⁺ next to Σ = Λ⁺; Σ = Λ⁺ to 7e-16 | not a defect: the second identity is wrong | — | — |
| 7 | nested duals in `_range_complement` | **right**: `MethodError` (no `svd` for them), not `ArgumentError` | typed float method and an `ArgumentError` fallback; the split's message for nested duals made explicit | P25 refusal testset (failed 20 times before the fix) |

Not probed: the plate methods (`factor_messages` with shared metric) under moving flat directions,
and inference through a graph with a dual `A` upstream of a flat path (only single messages are tested).

**Reviewer's tooling.** The command line given to copy ran; the reviewer spent its turns on
re-runs of debug variants (26 scripts) and did not append an entry after each probe as asked.

**Leaning / resolution.** Fixed (finding 7).
