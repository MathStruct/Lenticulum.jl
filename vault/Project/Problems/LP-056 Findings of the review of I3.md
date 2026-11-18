#open-problem

> **Status: resolved (2026-11-18).** The review of I3 (weighted spanning trees and alternating trees
> in `spanning_tree_schedule`) found no defect against the requirement. Four observations; one
> (an unsupported `weights` type raised a `MethodError`) was turned into an `ArgumentError` with a
> test; two are older than the change and recorded; one is intended.
> Related: [[Feature Backlog]] (I3), [[lib/Mycelium.jl/src/schedules|schedules]] "Choice of tree and
> alternating trees", [[Composing Optimisers]] §4.4.

> Sources: the change was implemented by Codex (GPT) in `Lenticulum.jl-codex`, branch
> `codex/i3-spanning-trees`, and reviewed by Mistral Large (`vibe --agent review`, \$0) in the helper
> worktree `Lenticulum.jl-mistral` at `0eef33ca` with the diff applied: six probes and a finished
> answer file. The brief, the answer and the probes are in `../Lenticulum.jl-Review-2026-11-18/i3/`.
> code: `lib/Mycelium.jl/src/schedules.jl` (`_spanning_forests`, `_forest_components`,
> `spanning_tree_schedule`, `_tree_sweeps`); tests: `lib/Mycelium.jl/test/runtests.jl` ("weighted
> and alternating spanning forests"), `lib/Lenticulum.jl/test/i3_spanning_trees.jl`.

**Problem.** The requirement given to the reviewer: the default (`weights = nothing`, `ntrees = 1`)
unchanged task for task; `weights` (a callable on factor indices or a Dict by name, finite) give a
maximum-weight spanning forest by Kruskal with the lower edge index breaking ties; later trees prefer
edges used fewer times; `:cover` until every edge is in a tree; the layout off-tree tasks, then per
tree inward and outward; within a tree segment every message after the tree inputs it reads;
`tree_schedule`, `plate_schedule`, `hybrid_schedule` unchanged.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | Default identical to the old implementation (copied from `HEAD`) on 400 random graphs × every root: 6 484 pairs, 0 mismatches, including mixed directions, parallel edges, edgeless factors | no defect | — | probe 1 |
| F2 | Tree 1 is a maximum-weight forest: equal to the brute-force maximum on 219 small graphs and to an independent Kruskal | no defect | — | probe 2 |
| F3 | Forests, layout and the order within each segment (a generic checker of "inputs sent earlier") on 22 692 schedules; `:cover` terminates and covers | no defect | — | probe 3 |
| F4 | One sweep exact on Gaussian trees (dims 1–3, every root); `infer!` exact means on diagonally dominant loopy graphs for every option (320 runs) | no defect | — | probe 4, `i3_spanning_trees.jl` |
| F5 | `_tree_sweeps` field-for-field unchanged (with and without `skip`); `tree_schedule`, `hybrid_schedule`, `plate_schedule` run | no defect | — | probe 5; Lenticulum `plates.jl` |
| O1 | An empty graph throws `BoundsError` | right, older than the change (the old function does the same) | recorded, not changed | probes 5, 6 |
| O2 | `weights` neither a Dict nor callable (a number, a vector, a string) threw `MethodError` | right; not required by the specification | `ArgumentError` with a message | Mycelium "weighted and alternating spanning forests" (`1.0`, `[1, 2]`, `"w"`) |
| O3 | `root = (:variable, 99)` throws `BoundsError` | right, older than the change (`_resolve_tree_root` does not check a tuple's index) | recorded, not changed | probe 5 |
| O4 | `ntrees = Int32(2)` is refused | intended ("a positive `Int`") | — | probe 2 |

The maintainer's own check before the review: the tree oracle of Codex's tests (random Gaussian
trees) cannot tell a weighted order from the breadth-first one, since on a tree every weighting gives
the same forest. Added: a loopy graph whose off-tree constraints have noise `1e12`, where one sweep of
the weighted order must be exact; replacing it by the breadth-first order makes 224 of 408 checks
fail. The benchmark's rows where the weighted tree equals BFS were checked to be the same forest
(`_spanning_forests` masks), not weights being ignored.

Not run by the reviewer: the benchmark, the full Lenticulum suite (the maintainer ran it), device tests.
