#open-problem

> **Status: resolved (2026-11-17).** The review of I24 (`:marginals` under partial residual sweeps)
> found that the fix stopped false convergence, but its `reason` overclaimed: it said every pending
> message residual was within the bound, while only the residual at the start of the last sweep was
> checked, and that sweep's own updates left some up to 2.7 times the bound. This was reproduced and
> fixed: a run about to stop now scans every task once and goes on if one is above the bound. The
> marginals were within the tolerance in every case the reviewer ran.
> Related: [[Feature Backlog]] (I24), [[LP-052 Findings of the review of I22]] (where I24 was found),
> [[lib/Mycelium.jl/src/passing|passing]] "Partial residual sweeps (I24)".

> Sources: a review of the uncommitted I24 change by Mistral Large (`vibe --agent review`, \$0) in the
> helper worktree `Lenticulum.jl-mistral` at `3655c0c1` with the diff applied: twelve probes and a
> finished answer file. Codex was out of quota. The brief, the answer and the probes are in
> `../Lenticulum.jl-Review-2026-11-17/i24/`.
> code: `lib/Mycelium.jl/src/passing.jl` (`_iterate!`, `_max_pending`); test:
> `lib/Lenticulum.jl/test/p9_relative_tol.jl` (the I24 lines).

**Problem.** The requirement given to the reviewer: under `:marginals`, a `ResidualSchedule` whose
`maxtasks` is below the number of tasks reports convergence only when, besides two quiet sweeps of
the marginals, the largest pending message residual is within the bound; the marginals are then at
the fixed point within the tolerance; full residual sweeps, other schedules and `:messages` are
unchanged.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | At a reported convergence the largest pending residual exceeded `tol` (absolute criterion): divider, `maxtasks = 1`, up to 2.68 × `tol` (probes 1, 3, 3b, 7). The check used the sweep's residual, a bound at the start of the last sweep, while the `reason` claimed it at the report | yes: reproduced (after the report, 4 of 6 cases above `tol`, and warm starts too) | at the stop, `_max_pending` evaluates every task without committing it; above the bound, the run goes on | I24 lines: `tol` 1e-11, 5e-11, 1e-9 × (`maxtasks` 1, damping 0; 3, 0.3), the pending scan after the report and after a warm start; the relative criterion |
| F2 | After every reported convergence the marginals moved by at most 0.014 `tol` under one full sweep (divider, rings n = 5, 8, chain, mixture chain) | no defect | — | the reviewer's probes 3c, 5, 6, 7 |
| F3 | The LP-052 defect is gone: `maxsweeps = 2` reports not converged; `maxtasks` 1 to 17 converge to the dense solve on the divider and on rings; a chain and a switching mixture chain (`MixtureReduction(16)`, `MeanField()`) against enumeration | no defect | — | probes 2, 3, 3c, 5, 6 |
| F4 | Unchanged: `maxtasks` equal to the task count and the default; flooding and spanning tree; `:messages`; warm start; damping; a run limited by `maxsweeps` | no defect | — | probes 2, 3, 4 |
| F6 | Missing tests: the pending residual at the report, warm starts, damping, graphs other than the divider | partly right | warm start, damping and the relative criterion added to the I24 lines; rings and mixtures rest on the reviewer's probes | — |

Not run by the maintainer: the device tests; rings and mixture graphs under the final change (the
reviewer ran them under the first one); the vault build and `docs/make.jl`.
