#open-problem

> **Status: resolved (2026-11-16).** The review of P24 (ForwardDiff and Zygote through a cluster
> node's free energy at fixed messages) ran six probes and found no defect. Two probes were weaker
> than reported (below); what they were meant to show is covered by the suites. Related:
> [[Feature Backlog]] (P24, P26), [[lib/Mycelium.jl/src/cluster|cluster]] "Derivatives at fixed messages".

> Sources: a review of the uncommitted P24 change by Mistral (`vibe --agent review`, `--max-price 3`,
> $0.41 at list price), in the helper worktree `Lenticulum.jl-mistral` at `c2284852` with the diff
> applied; the brief, the answer file and the probes in `../Lenticulum.jl-Review-2026-11-16/p24/`
> (the probes' outputs went to its session only, not to files).
> code: `lib/Mycelium.jl/src/cluster.jl`, `cluster_batch.jl`, `ext/MyceliumForwardDiffExt.jl`,
> `ext/MyceliumChainRulesCoreExt.jl`, `lib/Lenticulum.jl/src/regime.jl`; tests: `lib/Lenticulum.jl/test/regime_plate.jl`.

**Problem.** The requirement given to the reviewer: ForwardDiff and Zygote differentiate
`local_free_energy` of a `ClusterFactor` and of a `RegimeFactor` at fixed messages in `ps`, the
belief's dependence on `ps` included, per node, through `scalar_free_energy` and on a plate with
block rows; Float64 values and the batched paths unchanged; flat directions refused with an
`ArgumentError`, never a wrong number. Asked especially for dropped or wrong Zygote tangents.

| # | probe | right? | what was done | test |
|---|---|---|---|---|
| P1 | a regime factor with Gaussian messages: ForwardDiff and Zygote against central differences, fixed-belief gradient differs | passed, as the testset does | — | `test/regime_plate.jl` (P24 testset) |
| P2 | the clustered mixture of factor analysers through `scalar_free_energy` on block rows | passed | — | `test/regime_plate.jl` (MFA testset, seed 1) |
| P3 | Float64 values unchanged | the probe checks only that values are finite, not that they equal the previous ones; equality of the batched and per-replica Float64 paths is tested by the existing testsets, which pass | — | `test/regime_plate.jl`, `test/cluster.jl` |
| P4 | flat directions refused | passed: `ArgumentError` under both | — | `@test_broken` in the P24 testset |
| P5 | a cluster of {prior of x, regime factor} with a Gaussian message on z | passed, as the testset does | — | P24 testset |
| P6 | Float32 parameters | reported passing; not rerun here. With Float32 leaves `_numtype` promotes to Float64, so the batched path stays on | — | `probe6.jl` only |

No defect was found. Not probed: tied parameters across clusters of a graph without plates, a
cotangent on one graded part only (the testset weights all three), Zygote with messages that
depend on `ps` (they get a `@not_implemented` tangent by design).

**Reviewer's tooling.** The brief gave the exact `julia` command line without `cd` and without
redirection; every probe ran. It wrote the answer file as asked and finished within its turns.

**Leaning / resolution.** No change.
