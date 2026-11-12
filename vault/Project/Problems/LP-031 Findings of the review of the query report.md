#open-problem

> **Status: resolved (2026-11-12).** Four defects in the flags of `query_report` (P11), all
> reproduced by a test that failed before the fix and passes after it: the wrong machine epsilon
> for `:ill_conditioned`, `:not_lowest` decided by rank instead of energy (missed once, raised
> falsely once), and a round trip with no stable point left unflagged without a tolerance.
> Related: [[Feature Backlog]] (P11), [[LP-030 Findings of the review of round_trip]].

> Sources: a review of the uncommitted query report by Mistral (`vibe --agent review`), run with
> Julia in the helper worktree `Lenticulum.jl-mistral` (2026-11-12). Its answer is in
> `../Lenticulum.jl-Review-2026-11-12/P11-mistral.md`; its scripts are `scratch/p11/probe.jl`,
> `probe2.jl`, `probe3.jl` and `probe4.jl` in that worktree.
> code: `lib/VariationalDiffusion.jl/src/report.jl`; test: `lib/VariationalDiffusion.jl/test/runtests.jl`,
> "query report: what an answer can be trusted for (P11)", the block marked LP-031.

**Problem.** `query_report(m, z₀, ρ, ps, st)` answers a query and lists flags for what failed: the
solve, other roots, a root of lower energy, the conditioning, the round trip, coverage. Each flag
must mean what the docstring says.

Mistral ran the testset (29 of 29 at the time) and probed precision vectors of type `Any`, views,
Float32 and Float64, no free coordinates (dense and Krylov), no hard clamps, given `starts` and
`solution`, Krylov conditioning, and the shared singularity rule of `implicit_pullback`. None threw.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `:ill_conditioned` took ε from the state's type; the pullback takes it from $J_{FF}$'s, which is promoted with ρ. A Float32 state with Float64 precisions at $\operatorname{cond}_1 \approx 6\cdot10^7$ was flagged though `implicit_pullback` accepts it | right: reproduced | ε of $J_{FF}$'s element type | Float32 state, `ρ = [1e-3, 1e3]`: no flag, the pullback runs |
| F2 | `:not_lowest` was `rank > 1`: with the answer missing from the roots (rank 0) and a lower root found, no flag | right: reproduced (two-component mixture, start near the light component, the other start near the heavy one) | decided by the query energies: a root below the answer's by more than $\sqrt\epsilon$ relative | `rank == 0` and `:not_lowest` |
| F3 | the same rule flagged rank 2 when both roots have exactly equal energy | right: reproduced (equal weights, `noisefree_nodes`) | the same fix | `rank == 2`, flags `[:other_roots]` only |
| F4 | a reversed query with no stable point (distance `Inf`) was flagged only when `round_trip_tol` was given, against the docstring | right: reproduced | `Inf` is flagged always; a finite distance only against the tolerance | the reversed query from the ring's centre |

**Leaning / resolution.** All four fixed, each with its regression test. Not tested: the margin
$\sqrt\epsilon$ of F2 against a Float32 network (it was chosen above the energy error of a root
solved to the tolerance, not measured), and `:ill_conditioned` at a real fold of a learned relation.
