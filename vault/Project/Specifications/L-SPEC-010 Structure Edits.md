#design

> **Status: draft (2026-11-03, Claude), implemented as described.** Related: [[Extending Mycelium]] §6, [[L-SPEC-009 Cluster Nodes]], [[Composing Stochastic Factors]] (structure search on `experiment-qm`). Code: `lib/Mycelium.jl/src/edits.jl`; example: `examples/structure_edits.jl`.

> Sources: original to this repository. code: `lib/Mycelium.jl/src/edits.jl`; tests: `test/structure_edits.jl`, `lib/Mycelium.jl/test/edits.jl`; example: `examples/structure_edits.jl`
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

`test/structure_edits.jl`: edits reproduce the graphs built directly (nodes and edges by name, exact
free energy to $10^{-14}$) for adding and removing a switch; `transfer` keeps the surviving messages and
the warm start reaches the cold start's fixed point. `examples/structure_edits.jl`: greedy search over
$L$ switches on E3's 8-mode grid (200 points, exact scores through tied per-point cluster nodes, fits by
gradient steps at the exact beliefs) finds $L = 3$ by three additions, the branch's path, in 734 s.

## Open

- Plates; stable ids (append-only growth with masks) instead of rebuilding.
- The branch's split and merge moves for the switch over sums ($J > 1$).
- Fitting is slow against closed-form M-steps; frozen paths inside cluster nodes.
