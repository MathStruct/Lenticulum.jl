#open-problem

> **Status: resolved (2026-11-17).** The reviews of I22 (`criterion = :marginals` needs two quiet
> sweeps in a row) found no defect in the change. Mistral found one false "converged" that the
> change does not cover and that was there before it: a `ResidualSchedule` whose `maxtasks` is
> below the number of tasks commits only part of a sweep, which can be messages to factors alone, so
> `:marginals` still stops after 2 sweeps with every marginal trivial. It was reproduced. It is a
> `@test_broken` and a backlog row (I24); a sound rule for partial sweeps is a design question.
> Related: [[Feature Backlog]] (I22, I24), [[LP-051 Findings of the review of P9]] (where I22 was found),
> [[lib/Mycelium.jl/src/passing|passing]] "Convergence on marginals".

> Sources: two reviews of the uncommitted I22 change. Mistral Large (`vibe --agent review`, \$0) in the
> helper worktree `Lenticulum.jl-mistral` at `5505123b` with the diff applied: nine probes, all
> recorded in a finished answer file. Codex (`codex exec review --uncommitted`): no finding, tests not
> run. The brief, both answers and the probes are in `../Lenticulum.jl-Review-2026-11-17/i22/`.
> code: `lib/Mycelium.jl/src/passing.jl` (`_iterate!`); test: `lib/Lenticulum.jl/test/p9_relative_tol.jl`.

**Problem.** The requirement given to the reviewers: `:marginals` must not report convergence while
information is still travelling towards the marginals; `:messages` and the tree fast path are
unchanged; the hybrid schedules, which default to `:marginals`, keep converging.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `ResidualSchedule(maxtasks = 1, 2, 3)` with `:marginals` on chains of 6 and 10 variables reports "converged after 2 sweeps, residual 0.0" with every marginal still `TrivialBelief` (probe 4); `:messages` and the default `ResidualSchedule()` are correct | yes: reproduced with the reviewer's probe, and on the divider (`maxtasks` 1 and 3). Predates I22 (with one quiet sweep it stopped after 1). Not fixed: two quiet partial sweeps carry no guarantee, and how many would is not bounded by an argument | backlog I24 | `@test_broken` in `p9_relative_tol.jl` |
| 2 | Probes 1–3, 5, 6, 8: the divider under flooding now converges to the exact means (absolute and relative); chains of 4–20 and rings under flooding converge to the dense solve; `maxsweeps = 1, 2` report not converged; warm start, tree fast path, damped tree; Anderson, default residual, spanning tree, damped flooding; hybrid and hybrid plate schedules converge to their oracles; damped rings report honestly | no defect | — | the reviewer's probes |
| 3 | Unverified (reviewer): compensating changes of two messages to one variable could leave its marginal unchanged and defeat the rule on a loopy graph | not shown; stated as an assumption in `passing.md` | — | — |
| 4 | Unverified (reviewer): the default `ResidualSchedule()` on a graph with more than 10 000 tasks is a partial sweep too, so F1 applies there | right by the code (the same path as F1); not run | covered by I24 | — |

Not run by the maintainer: the device tests; the vault build and `docs/make.jl`.
