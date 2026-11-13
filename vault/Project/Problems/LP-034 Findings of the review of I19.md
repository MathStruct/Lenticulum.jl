#open-problem

> **Status: resolved (2026-11-13).** No wrong result, crash or missing check. Mistral's four probes
> compared the batched messages, beliefs and free energies of tied cluster nodes over a plate with
> the per-replica ones (several factors per cluster, several precision classes, Gaussian inputs
> with points, discrete and continuous targets, pruning categorical inputs, N = 2, scalar points,
> flat directions, rows whose factors differ) and the errors of both paths; all agree. The review
> wrote no answer (it stopped at its turn limit); its probes, after fixes to their own code, are
> the evidence. Two became regression tests. Related: [[Feature Backlog]] (I19),
> [[LP-032 Findings of the review of I6]].

> Sources: a review of the uncommitted I19 change by Mistral (`vibe --agent review`, 100 turns,
> stopped at the limit without an answer file), run with Julia in the helper worktree
> `Lenticulum.jl-mistral` at `1770cc13` with the diff applied; probes in `scratch/i19/` of that
> worktree, copied with the corrected ones (`-fixed`), the brief and the run's output to
> `../Lenticulum.jl-Review-2026-11-13/`; the logs of the corrected runs in `~/.cache/lenticulum/i19/probes/`.
> code: `lib/Mycelium.jl/src/cluster_batch.jl` (`_batch_joint`, `_as_clusters`, `_same_struct`),
> `lib/Mycelium.jl/src/cluster.jl` (`_continuous_parts`, `_discrete_parts`, `_belief_parts`);
> tests: `lib/Lenticulum.jl/test/regime_plate.jl`, `test/cluster_flat.jl`, `test/cluster.jl`.

**Problem.** I19 replaced the per-replica joints of batched cluster nodes over a plate by one
stacked joint whose data terms are computed for all replicas together, and assembled each
replica's message and belief from its columns (`cluster.md`, "Batching in one stacked joint").
The requirement: for every row `_cluster_batchable` accepts, the results equal the per-replica
path's up to rounding, and the same errors are thrown.

| # | probe | result | what was done | test |
|---|---|---|---|---|
| P1 | multi-factor clusters (a sum constraint and a `RegimeFactor`) over a plate: messages to every channel, beliefs, graded free energies, against per replica; N = 2 and 5, scalar and vector points, Gaussian inputs with points | 1 188 checks pass | — | covered by `cluster.jl` and `regime_plate.jl` (random instances) |
| P2 | rows of `RegimeFactor`s: N = 2, pruning categorical inputs, observed switches (not batchable), a row with one unequal factor, `_as_clusters` sharing one cluster | 906 checks pass after two fixes to the probe (a tolerance written `atol = 1e-9 && …`, which Julia parses as the tolerance; a generic comparison that read a `Dict`'s unset slots and threw `UndefRefError`) | the unequal-factor part adopted | "RegimeFactor plate: unequal factors in a row go replica by replica" |
| P3 | errors: every state of a categorical input impossible, every outcome impossible, a linear term in a flat direction, unequal flatness, a mislabeled categorical on a later replica (LP-032) | 27 checks pass: both paths throw | — | — |
| P4 | flat directions in tied clusters over a plate: messages, beliefs, energies against per replica and the closed form; a tilt in the flat direction refused | 27 checks pass after fixes to the probe (an unbalanced parenthesis; `(mix = …)` without a comma, read as a keyword; a message asked for an internal variable, which both paths rightly refuse) | adopted | "Flat directions: tied clusters over a plate, batched" |

Found by the suites, not by the review: Lenticulum's ChainRulesCore extension still marked its old
`_as_clusters` non-differentiable; with `_as_clusters` now Mycelium's (and marked there) the method
was defined twice and overwritten, with a warning on every load. The mark in Lenticulum's
extension was removed.

**Leaning / resolution.** Nothing to fix in the change. The review did not run the suites (the
maintainer's session ran all seven packages) nor the benchmarks.
