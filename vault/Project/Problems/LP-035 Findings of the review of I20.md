#open-problem

> **Status: resolved (2026-11-15).** No wrong result, crash or missing check. Mistral reported one
> finding, that cluster nodes' block inputs with different categorical labels per replica would
> be batched where the per-replica path refuses them; its own probe did not run, and repaired it
> shows the inputs refused. A categorical block holds one set of labels for all its columns, so
> such inputs arrive as a vector block, which the block path sends through the columns. Not
> reproduced; the case became a regression test. Related: [[Feature Backlog]] (I20),
> [[LP-034 Findings of the review of I19]], [[LP-032 Findings of the review of I6]].

> Sources: a review of the uncommitted I20 change by Mistral (`vibe --agent review`, `--max-price 1`;
> it stopped at that price after 71 turns, with an answer file kept as it went), run with Julia in
> the helper worktree `Lenticulum.jl-mistral` at `2d554500` with the diff applied; the brief, the
> diff, the output, the answer file and the probes (with the repaired ones, `-fixed`) in
> `../Lenticulum.jl-Review-2026-11-15/`; the logs of the maintainer's runs of the probes in
> `~/.cache/lenticulum/i20/mprobe_*.log`.
> code: `lib/Mycelium.jl/src/cluster_batch.jl` (`_block_batchable`, `_uniform_shape`, the block-input
> `factor_messages`, `_discrete_block`, `_continuous_block`), `lib/Mycelium.jl/src/cluster.jl`
> (`_continuous_message`), `lib/Mycelium.jl/src/plates.jl` (`_group_plan`, `_group_trees`),
> `lib/Mycelium.jl/src/free_energy.jl` (`_energy_row`); tests: `lib/Lenticulum.jl/test/regime_plate.jl`.

**Problem.** I20 grouped untied factors without parameters in plate sweeps and free-energy rows,
and let cluster nodes read a plate sweep's inputs as blocks and return their messages as blocks
([[plates]] "Untied factors without parameters", [[cluster]] "Block inputs"). The requirement given
to the reviewer: the grouped and block paths give the edge-by-edge and column paths' results
exactly, for every input shape, and fall back when they cannot.

| # | finding or probe | right? | what was done | test |
|---|---|---|---|---|
| F1 | `_uniform_shape(::CategoricalBlock, …)` compares the columns' log-probabilities but not their labels, so inputs whose labels differ between replicas (LP-032) would be batched | no: a `CategoricalBlock` has one `labels` field for all columns, and `belief_block` stacks categorical beliefs with different labels into a `VectorBlock`, for which `_uniform_shape` returns false. Mistral's probe `probe_label_defect2.jl` failed on a missing `using LinearAlgebra`; repaired, it prints `_block_batchable: false` | none in the code | "a mislabeled categorical input on a later replica is refused": the stacked inputs are a vector block, not block-batchable, and `factor_messages` on them throws |
| P1 | the rewritten `_continuous_message` (a linear scan) against the old one (a `Dict`), on random inputs: one assignment, several, equal columns, different precisions | passed (`probe_continuous_message.jl`, rerun by the maintainer: "All tests passed") | — | covered by "block inputs give the column path's messages exactly" |
| P2 | block against column inputs for a `RegimeFactor` with points | passed after repairing the probe (a `reshape` of a 2 × 2 matrix to 2 × 2 × 2) | — | covered by the same testset |

Not checked by the review (its answer says so): the I20 testsets themselves, untied state that is
non-empty, and `fit_em`'s key selection. The maintainer's session ran all seven suites, the two
I20 testsets (four mutations of the block path fail 18 to 69 of their tests) and the benchmarks.

**Leaning / resolution.** Nothing to fix in the change. The price limit, not the turn limit, ended
this run: `--max-price 1` counts list prices even for the free model (3.9 M input tokens, 97%
cached, $1.01), so it stops a review at about 70 turns.
