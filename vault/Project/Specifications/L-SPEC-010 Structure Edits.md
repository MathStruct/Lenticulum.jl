#design

> **Status: draft (2026-11-03, Claude), implemented as described.** Related: [[Extending Mycelium]] §6, [[L-SPEC-009 Cluster Nodes]], [[Composing Stochastic Factors]] (structure search on `experiment-qm`). Code: `lib/Mycelium.jl/src/edits.jl`; example: `lib/Lenticulum.jl/examples/structure_edits.jl`.

> Sources: original to this repository. code: `lib/Mycelium.jl/src/edits.jl`; tests: `lib/Lenticulum.jl/test/structure_edits.jl`, `lib/Mycelium.jl/test/edits.jl`; example: `lib/Lenticulum.jl/examples/structure_edits.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

Changing the structure of a built graph (variables, factors, edges) and carrying its messages over,
so that structure moves can be scored by the graph's free energy. The moves, the search and the
score of a model family are the caller's.

## The contract

**§1. Editing.** `GraphBuilder(g)` returns a builder holding a copy of `g`. On it `variable!`,
`factor!` and `connect!` add as at construction; `remove_factor!(b, name)` removes a factor and its
edges, `remove_variable!(b, name)` a variable and its edges, `disconnect!(b, factor, channel)` one
edge, and `replace_factor!(b, name, factor; freeze, rule, params)` puts a new factor in a node,
keeping its edges on channels the new factor has. `build(b)` makes the new graph. Names identify
nodes; ids are renumbered. A graph with plates is refused.

**§2. Messages.** `transfer(s, g_old, g_new)` returns a store for `g_new` with `s`'s messages, in both
directions, on every edge whose factor name, channel and variable name are in both graphs, the pins
of the variables in both, and `s`'s iteration. Other edges start empty.

**§3. Scoring.** Nothing new: the free energy of the edited graph, exact when inference is (cluster
nodes, trees).

## Checked by

`lib/Lenticulum.jl/test/structure_edits.jl`: edits reproduce the graphs built directly (nodes and edges by name, exact
free energy to $10^{-14}$) for adding and removing a switch; `transfer` keeps the surviving messages and
the warm start reaches the cold start's fixed point. `lib/Lenticulum.jl/examples/structure_edits.jl`: greedy search over
$L$ switches on E3's 8-mode grid (200 points, exact scores through tied per-point cluster nodes, fits by
gradient steps at the exact beliefs) finds $L = 3$ by three additions, the branch's path, in 166 s (clusters as a plate). `lib/Lenticulum.jl/examples/structure_clusters.jl`: the switch over sums ($J$ clusters,
labels tied to switch outcomes by `TableFactor`s) with all four of the branch's moves; split and
merge re-add the label and outcome variables with new state counts. On the branch's tree (200
points) greedy goes (1, 0) → (1, 1) → (2, 1) → (2, 2) → (2, 3), BIC 1 355.84 in 441 s, while the
generating (3, 2) reaches 1 299.15: a worse structure, as on the branch's 16-mode grid. With a compound move (a
data-driven split, then a removal) the search continues to (3, 2) at 1 299.17 (970 s, 81 fits).

## Open

- Editing plates directly; stable ids (append-only growth with masks) instead of rebuilding. An edited
  graph whose replicas are named `x[i]` gets its plate back with `as_plate` (2026-11-04).
- Moves and warm starts are the caller's; the example's compound split-and-remove with a data-driven
  split is one choice that works on the tree (`lib/Lenticulum.jl/examples/structure_clusters.md`), not a general rule.
- Fitting: closed-form M-steps from cluster beliefs (`MyceliumLearning.fit_em`, SQUAREM) since 2026-11-04,
  about 7× faster per warm fit than 300 gradient steps; a search is now limited by compiling each
  new structure's inference code and by slow EM on over-parameterised candidates (`em.md`). Frozen
  paths inside cluster nodes: done 2026-11-04.
