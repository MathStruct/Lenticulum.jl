#open-problem

> **Status: resolved (2026-11-17).** The review of P9 (a relative convergence criterion,
> `propagate!(...; rtol)`) found one defect in the change: a zero change carrying a dual tangent,
> against a zero bound, gave 0/0. It was reproduced and fixed. The suites found a second one after
> the review: a new argument check refused `tol = 0` without `rtol`. It was fixed (finding 5). The review also found one defect that predates
> P9: an iterated `infer!` with dual parameters throws when it builds its report. That one is a
> backlog row (I23).
> Related: [[Feature Backlog]] (P9, I22, I23), [[lib/Mycelium.jl/src/passing|passing]] "A relative convergence criterion".

> Sources: a review of the uncommitted P9 change by Mistral Large (`vibe --agent review`, \$0), in the
> helper worktree `Lenticulum.jl-mistral` at `91005dbc` with the diff applied. vibe 3.1.2 ran this
> time (it failed on P29 earlier today). Six probes: five recorded in the answer file, the sixth
> written but not recorded, and the maintainer ran it. The brief, the answer and the probes are in
> `../Lenticulum.jl-Review-2026-11-17/p9/`.
> code: `lib/Mycelium.jl/src/messages.jl`, `gaussian.jl`, `blocks.jl`, `passing.jl`, `plates.jl`;
> test: `lib/Lenticulum.jl/test/p9_relative_tol.jl`.

**Problem.** The requirement given to the reviewer: with `rtol = 0` nothing changes. With `rtol > 0`
each group of a message's (or a marginal's) parameters changes by at most `tol + rtol·s`, and the
residual is in units of that bound. This holds for every schedule, plates included, and leaks
nothing outside `propagate!`. It must not see the units of the model.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | `_ratio` decided "no change" with `d <= 0`, and its comment said that reads a dual's value. It does not: `Dual(0, 1) <= 0` is `false`, because ForwardDiff breaks ties by the partials (probe 6c). A zero change with a tangent, against a zero bound (`tol = 0`, a group zero on both sides), gave `Dual(NaN, NaN)` | yes, reproduced: `NaN` on the old code, `0` after the fix (a scratch script run in both worktrees) | `_primal(d) <= 0`, with `_primal` the identity and its dual method in `MyceliumForwardDiffFlatExt` | "relative_residual: groups, scales, limits", the dual line |
| 2 | An iterated `infer!` with dual parameters fails with a `MethodError` in the `ConvergenceReport` constructor (`residual::Float64` from a dual), with `rtol = 0` as well as `rtol > 0` (probe 6) | yes (the probe's output). It predates P9: the absolute path fails the same way | backlog row I23; not fixed here | — |
| 3 | Probes 1–5: argument checks, `rtol = 0` identical to no keyword on five schedules × two criteria, no leak of the scoped value (also after a throw and in spawned tasks), block fast paths equal to the replica path for every block type (690 checks), edge cases of `relative_residual`, the same sweeps at units $10^{-6}$ to $10^4$ | no defect | — | — |
| 4 | At units $c \ge 10^5$ the sequential schedule throws because a message loses positive definiteness to rounding (probe 5b) | predates P9 (the absolute criterion throws the same way); not a convergence defect | — | — |
| 5 | Found by the suites after the review, not by the reviewer: the new argument check refused `tol = 0` without `rtol`, which MyceliumLearning's graph-adjoint test uses to run to `maxsweeps` (it errored). The brief had stated this check as a requirement, which contradicts its own point 1 (`rtol = 0` changes nothing), so the reviewer confirmed it | yes, `test/graph_adjoint.jl:64` errored | the checks apply only when `rtol > 0` (`rtol < 0`; `tol < 0` with `rtol > 0`) | "loopy Gaussian BP …", `tol = 0` without `rtol` runs to `maxsweeps`; MyceliumLearning's suite |
