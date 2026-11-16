#overview

> Section S of the [[Feature Backlog]]: edits, moves, gates, scores and a search driver. Open rows only; a row that is
> done moves to [[Backlog Done 2026-11]] with its id. Status, priority, ids and the column *needs*
> are explained in [[Feature Backlog]], which also holds the dependency graph.

> Sources: original to this repository; the sources of each row are in its *source* column.

| id | feature | status | needs | priority | source | why |
|---|---|---|---|---|---|---|
| S1 | Template edits of plated graphs, stable ids | planned | — | low | [[L-SPEC-011 Plate Edits]] | edits and `as_plate` take about 15% of a search today |
| S2 | Moves and warm starts as library code (beam search, final refit) | proposed | — | low | `lib/Lenticulum.jl/examples/structure_clusters.md` | the example's beam with refit reaches the best fit in 62% of the gradient search's time; a reusable driver would serve other families |
| S3 | Collapsed M-steps as a library feature (integrate out the intermediates of a switch-over-sum) | proposed | — | medium | `lib/Lenticulum.jl/examples/structure_clusters.md` | the per-factor M-step crawls with nearly deterministic switches |
| S4 | Gated inclusion of edges (spike-and-slab): a binary gate and a slab per candidate edge, inferred with the rest; a structure edit becomes the rounding of a gate | proposed | — | medium | [[The Regulariser and the Loss]] §2, §6; Daniel, 2026-11-05 ("valuable but rarely used, or to regularise?") | separates whether an edge exists from how large it is, so a rare but large edge is not shrunk away |
| S5 | Per-edge diagnostics: usage (posterior mass through the edge), gain when used, net value against its complexity, bootstrap stability | proposed | — | medium | Daniel, 2026-11-05 | tells a rare but valuable connection from one to prune; also says what an out-of-core store must keep loadable |
| S6 | Structure scores beyond BIC: a Laplace evidence (the Hessian at the fit) and a held-out free energy | proposed | — | medium | [[The Regulariser and the Loss]] §6 | BIC is the crude Laplace term; held-out scores also judge parts that are not normalised densities |
| S7 | A structure-search driver in the library: move sets (edits, gates, cluster merges), the scores above, beams, restarts and a final refit | proposed | S2, S6 | medium | [[The Regulariser and the Loss]] §5–§6; `lib/Lenticulum.jl/examples/structure_clusters.md` | every search today is example code; merges the "moves and warm starts" row above |
