#open-problem

> **Status: resolved (2026-11-15).** One finding, right and fixed: the new shortcut "a point block
> times a mixture block is the point" did not check that both blocks have the same number of
> replicas, so blocks of different sizes gave a block of one of the sizes. Inside the library a
> plate's rows always have one size, so no sweep met it; the shortcut now refuses such blocks with
> a `DimensionMismatch`, and a test shows it. Related: [[Feature Backlog]] (I21),
> [[LP-035 Findings of the review of I20]].

> Sources: a review of the uncommitted I21 change by Mistral (`vibe --agent review`,
> `--max-price 3`, finished after 53 turns, $0.83 at list price), run with Julia in the helper
> worktree `Lenticulum.jl-mistral` at `4ffd955c` with the diff applied; the brief, the diff, the
> output, the answer file and the probes in `../Lenticulum.jl-Review-2026-11-15/i21/`; the
> maintainer's run of its probe in `~/.cache/lenticulum/i21/mprobe_req1.log`.
> code: `lib/Mycelium.jl/src/blocks.jl` (`ComponentMixtureBlock`, `combine_blocks`, `_same_replicas`),
> `lib/Mycelium.jl/src/cluster_batch.jl` (`_continuous_block`), `lib/Mycelium.jl/src/free_energy.jl`
> (`_energy_rows`); tests: `lib/Lenticulum.jl/test/regime_plate.jl`.

**Problem.** I21 added a mixture block with a precision per component, returned by plates of tied
cluster nodes, and kept the plate's energy rows in the graph per store layout ([[blocks]] "Mixtures
with a precision per component", [[free_energy]] "Batched energies over plates"). The requirement
given to the reviewer: the block's columns, distance and products equal the per-belief rules, the
block path equals the per-replica path, and the kept rows equal rows found afresh for every store.

| # | finding or probe | right? | what was done | test |
|---|---|---|---|---|
| F1 | `combine_blocks(::PointBlock, ::Union{MixtureBlock,ComponentMixtureBlock})` and the reverse order return the point block without checking the replica counts; the column path truncates in one order and throws a `BoundsError` in the other | yes, for blocks of different sizes (none arise in a sweep). A test refusing them failed 8 times on the change as reviewed (`test_combine_nreplicas.jl` shows the behaviour) | both methods check the counts and throw `DimensionMismatch` | "ComponentMixtureBlock: columns, distance and products against the per-belief rules": blocks of $n$ and $n + 1$ replicas, both block types, both orders |
| P1 | `replica` of a `ComponentMixtureBlock` against hand-built per-replica mixtures, proper and improper | its checks pass; the probe then built a block whose `logw` had the wrong shape, which the constructor refused (the probe's error, not the change's) | — | covered by the I20 block testset (which now returns this block, improper components included) |
| — | `block_distance`, the energy-row cache, unchanged free energy and EM | "passed by code analysis", not run by the reviewer | — | the maintainer's tests: the distance against `belief_distance` with only the precisions moved, the cache across stores of one and of another layout, the closed form; four mutations of the change fail 2 to 57 tests |

Not checked by the review (its answer says so): several threads using the cache at once, detached
edges, a whole EM run. The maintainer's session ran all seven suites and the benchmarks.

Found by the maintainer while testing the cache, not by the review: a plate schedule on a store
whose block rows are numbered otherwise than `MessageStore(g)`'s gives wrong results silently (P22
in the [[Feature Backlog]]); it predates I21.

**Leaning / resolution.** F1 fixed with its test. The other operations that take a point block
(`combine_blocks(::PointBlock, ::GaussianBlock)`) have the same missing check and the column path
truncates; left as they were.
