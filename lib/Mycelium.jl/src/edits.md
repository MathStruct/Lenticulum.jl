#implementation

> Structure edits: a built graph is opened as a builder (`GraphBuilder(g)`), changed by name
> (`variable!`, `factor!`, `connect!`, `remove_factor!`, `remove_variable!`, `disconnect!`,
> `replace_factor!`) and built again; `transfer` carries the messages of the edges both graphs share.
> A greedy search over switch wirings done this way finds the generating structure, scored by the
> graph's exact free energy (`examples/structure_edits.md`).

> Sources: original to this repository. code: `edits.jl`; tests: `../test/edits.jl`, Lenticulum `test/structure_edits.jl`; example: Lenticulum `examples/structure_edits.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

Implements: [[L-SPEC-010 Structure Edits]], [[Extending Mycelium]] §6.

## How

A `FactorGraph` stays immutable; an edit makes a new one. `GraphBuilder(g)` copies the graph's
nodes and edges into a builder. The builder's `variable!`, `factor!` and `connect!` add as before;
the removals drop nodes or edges and renumber what is left (`_keep!`); `replace_factor!` swaps the
factor object of a node and drops its edges on channels the new factor does not have. Nodes and
edges are identified by name throughout, so ids are free to change. Plates are refused: their
replica records hold ids.

`transfer(s, g_old, g_new)` makes a store for the new graph and copies every message, both
directions, of an edge with the same factor name, channel and variable name in both graphs, and
the pins of the variables both have. It does not decide which messages are still right: next to an
edit they are a warm start that the schedule replaces.

This is the rebuild-and-carry-over design. The alternative in the dynamic-graphs analysis
(append-only growth with deletion masks) would keep ids stable and avoid the copy; an edit here
costs a pass over the graph, which in the search example is negligible against fitting.

## Tests

- Lenticulum `test/structure_edits.jl`: adding a switch to the additive wiring by edits (the sum
  node replaced with one more channel, a switch and its variables added) gives the graph built
  directly with one more switch: the same nodes and edges by name and the same exact free energy
  (every factor in one cluster) to $10^{-14}$; so does removing one. `disconnect!` and the refusals.
  After adding a switch, the transferred store keeps every old message, and a loopy sequential
  schedule from it reaches the cold start's fixed point (marginal means to $10^{-8}$). On this
  small graph it takes the same 4 sweeps as from cold; pins survive by name.
- Mycelium `test/edits.jl`: removal with renumbering, `transfer` of the surviving messages,
  replacing a factor keeps its edges, disconnect and reconnect, the refusals.

## What it does not do

- Plates, and edits of a plate's replicas.
- Choosing which messages to drop: `transfer` keeps every shared edge's messages.
- Moves and scores: the search, its moves and the BIC live in the example, not in Mycelium.

## Review (Gemini via `agy`, 2026-11-03)

Valid and fixed: `transfer` matched edges by names only, so a variable removed and re-added under
its name with another dimension received its old messages and pins; the variable's space is now
part of the key (tested). Not bugs: `replace_factor!`'s defaults (`freeze`, `rule` and `params` of a
node are never `nothing`), and the shared message objects (messages are not mutated in place).
