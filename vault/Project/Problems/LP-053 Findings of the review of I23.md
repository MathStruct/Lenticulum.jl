#open-problem

> **Status: resolved (2026-11-17).** The review of I23 (an iterated `infer!` with dual parameters no
> longer throws in `ConvergenceReport`) found one defect that the requirement covers: a
> `ResidualSchedule` chose the next task by comparing dual residuals, and a dual breaks ties by its
> partials, so the tangents could change the order of the updates and the stopping sweep. It was
> reproduced and fixed (the choice is made on the primal value). The derivatives were right in every
> case the reviewer ran.
> Related: [[Feature Backlog]] (I23, I24), [[LP-051 Findings of the review of P9]] (where I23 was
> found), [[lib/Mycelium.jl/src/passing|passing]] "A relative convergence criterion".

> Sources: a review of the uncommitted I23 change by Mistral Large (`vibe --agent review`, \$0) in the
> helper worktree `Lenticulum.jl-mistral` at `3ec67646` with the diff applied: three probes, recorded
> in a finished answer file (vibe stopped at its turn limit after writing it). Codex was out of quota.
> The brief, the answer and the probes are in `../Lenticulum.jl-Review-2026-11-17/i23/`.
> code: `lib/Mycelium.jl/src/passing.jl` (`ConvergenceReport`, `_iterate!`, `sweep!` for
> `ResidualSchedule`); test: `lib/Lenticulum.jl/test/p9_relative_tol.jl` ("I23").

**Problem.** The requirement given to the reviewer: an iterated `infer!` or `propagate!` with dual
parameters (first order and nested) runs; the report's `residual` is the primal `Float64`; whether a
run has converged is decided on the primal residual, so the stopping sweep does not depend on the
tangents; results without duals are unchanged; at a converged fixed point of loopy Gaussian BP the
derivative of the means matches the dense solve.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `ResidualSchedule()` with `damping = 0.3` stops at sweep 111 for tangent 1.0 and 110 for −2.5 and 0.0; `ResidualSchedule(; maxtasks = 3)` at 412 against 413 (probe 2). `sweep!` picks the task with `res > best_res` / `res == best_res` on dual residuals | yes: reproduced, both cases (the same tasks in the same order failed for the three tangents; flooding passed) | the task is chosen on `_primal(res)` | "I23", the tangent loop: same report and the same recorded task order for tangents 1.0, −2.5, 0.0 |
| 2 | Under duals `AndersonSchedule` runs without its memory (13 to 89 sweeps more than the float run): `_anderson_state` mixes only `Vector{Float64}` messages and returns `nothing` otherwise, so the sweep is the inner schedule's | right by the code (`anderson.jl`, `_anderson_state`); not a defect: the derivatives match the dense solve | recorded in `passing.md` ("A relative convergence criterion") as a limit | — |
| 3 | Probes 1–2: the I23 testset alone; flooding, residual, residual with `maxtasks = 3`, Anderson, hybrid, sequential, damped and plate schedules under four keyword sets: no throw, `residual isa Float64`, converged, Jacobian of the means within 3.3e-13 of the dense solve (plates: 2/3 per replica, exact to 1.2e-16) | no defect | — | the reviewer's probes |
| 4 | `ResidualSchedule(; maxtasks = 3)` with `:marginals` stops after 2 sweeps with trivial marginals, under duals as under floats | known: I24 ([[LP-052 Findings of the review of I22]] F1) | — | the existing `@test_broken` |

Not run by the maintainer: the device tests; Zygote through an iterated `infer!` (not changed: the
report was built from Float64 residuals there before); the vault build and `docs/make.jl`.
