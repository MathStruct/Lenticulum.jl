#open-problem

> **Status: resolved (2026-11-16).** The review of P26 (ForwardDiff and Zygote through a cluster's
> joint precision with flat directions, at constant rank) ran eight probes and found no defect. One
> probe was weaker than reported (below); what it was meant to show is covered by the suites.
> Related: [[Feature Backlog]] (P25, P26), [[lib/Mycelium.jl/src/cluster|cluster]] "Flat directions under a derivative".

> Sources: a review of the uncommitted P26 change by Mistral (`vibe --agent review`, `--max-price 3`,
> $0.66 at list price), in the helper worktree `Lenticulum.jl-mistral` at `73277da7` with the diff
> applied; the brief, the answer file and the probes in `../Lenticulum.jl-Review-2026-11-16/p26/`.
> code: `lib/Mycelium.jl/src/cluster.jl`, `ext/MyceliumForwardDiffFlatExt.jl`; tests: `lib/Lenticulum.jl/test/regime_plate.jl`.

**Problem.** The requirement given to the reviewer: at constant rank, the derivative through the
diffuse limit equals that of the value path, also with repeated eigenvalues on the range; a tangent
that changes the rank to first order is refused with an `ArgumentError`; Float64 values unchanged;
nested duals and other number types still refused on the flat path.

| # | probe | right? | what was done | test |
|---|---|---|---|---|
| P1 | Float64 values of `_psd_factor`, `_psd_solve`, `_psd_inv`, `_psd_logdet` unchanged (rank 0, rank deficient, definite) | the probe has no assertion: it prints the values and reports "matched". Unchanged Float64 values are shown by the suites: Lenticulum `test/cluster_flat.jl` (closed forms to $10^{-12}$) and the cluster testsets pass | — | `test/cluster_flat.jl` |
| P2 | ForwardDiff against central differences, constant rank (rank 1 and 2, rotating, repeated eigenvalues) | passed | — | P26 testset |
| P3 | rank changes 0→1, 1→2, 2→3 refused; constant rank accepted | passed | — | P26 testset (1→2) |
| P4 | rank-0 J (every direction flat) under ForwardDiff; 0→1 refused | passed | — | `probe4.jl` only |
| P5 | one flat direction, `_check_range` refusing a linear term along it | passed | — | P26 testset, `test/cluster_flat.jl` |
| P6 | Hessians (nested duals), BigFloat, Rational refused with `ArgumentError` | passed | — | `probe6.jl` only |
| P7 | the new fields of `_Flat` (`K` for `λ`) under every `_psd_*` | passed | — | the suites |
| P8 | the testset's regime factors, P24's former limit, Zygote against ForwardDiff | passed, as the testset does | — | P26 testset |

No defect was found. Not probed: `cluster_marginal` of a flat class under ForwardDiff (the brief
named it; the probe checked the field accesses only), and a cluster with several precision classes
whose flat directions differ in direction but not in number.

**Reviewer's tooling.** The brief gave the exact `julia` command line without `cd` and without
redirection; every probe ran, and it wrote the answer file as it went.

**Leaning / resolution.** No change.
