#open-problem

> **Status: resolved (2026-11-19).** The review of P33 (the cluster batch path, accelerated EM and
> sample reweighting in the element type of their inputs) found no defect. The maintainer's session
> changed one thing before the review: the block form of `_cluster_batchable` built a column per
> replica on every call to check their joint types; it now checks the first column only, which suffices
> because `_block_batchable` admits only blocks (one array, one element type, per variable).
> Related: [[Feature Backlog]] (P33), [[LP-061 Findings of the review of P32]],
> [[lib/Mycelium.jl/src/cluster|cluster]] "Element type".

> Sources: implemented by Codex (GPT, `codex exec -C ../Lenticulum.jl-codex`, branch `codex/p33`, one
> run); reviewed by Mistral Large 4 (`vibe --agent review` in `../Lenticulum.jl-mistral`, \$0, stopped
> at its 100-turn limit with an answer file of four probes). Brief, probes and answer:
> `../Lenticulum.jl-Review-2026-11-19/p33/`; the session's logs: `~/.cache/lenticulum/p33/`.
> code: `lib/Mycelium.jl/src/cluster_batch.jl`, `messages.jl`, `lib/LenticulumCore.jl/src/parameter_tree.jl`;
> tests: `lib/Mycelium.jl/test/element_type.jl`, `lib/MyceliumLearning.jl/test/element_type.jl`,
> `lib/LenticulumCore.jl/test/parameter_ad.jl`.

**Problem.** The requirement given to the reviewer: the stacked joint of a cluster plate serves any
plain floating type (Float32 messages, beliefs, free energies and Zygote gradients equal to the
replica-by-replica ones; replicas with different joint types and duals go replica by replica; wider
data widens, never rounds); an empty parameter subtree no longer widens accelerated EM's flattened
Float32 vector; sample weights take the type of what they are computed from; Float64 bit for bit
unchanged.

| # | probe | result |
|---|---|---|
| 1 | Float32 regime plate (N = 30) through sweeps, gradients (Zygote, ForwardDiff, `free_energy_gradient`), accelerated `fit_em`, batched moment statistics | passed: batch path taken, Float32 throughout, Float64 agreement 2e-7 (energy) to 4e-5 (gradient) |
| 2 | Float64 bit for bit against `d3adbd6e` (regime plate, P33 plate, accelerated EM with empty subtree, flatten round trip, sample reweighting) | passed: 128 lines identical; under the old code Zygote through `combine` with a `SampleBelief` threw (mutation), the new code differentiates |
| 3 | mixed replica types, Int points, duals, wide Float64 points | passed: mixed and dual not stacked; Int points Float32 exact; wide points keep their bits |
| 4 | `_flatten_active`/`_unflatten` on 9 trees with empty subtrees, ForwardDiff/Zygote; accelerated EM on 80 points | passed: Float32 μ within 7.9e-8 of Float64 |
| 5 | (session, `../Lenticulum.jl-Review-2026-11-19/p33/probe5.jl`) zero density everywhere still throws `ArgumentError`, weighted and unweighted; Float32 samples with Float64 weights give Float64 | passed |

Not probed: a `GaussianBlock` whose vector of precisions mixes element types. `_block_batchable` admits
it only when every precision `isequal`s the first, and the stacked joint reads replica 1's precision
anyway, so checking the first column loses no value (argued from the code, not tested). A wholly empty
parameter tree still flattens to `Float64[]` (documented in `parameter_tree.md`).
