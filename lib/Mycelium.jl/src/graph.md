# graph.jl — implementation note

Implements: the bipartite factor graph, edge directions, and the two acyclicity predicates.
Theory: [[Factor Graphs]].

## Structure

```
VariableNode(id, name, space)        -- a wire; no behaviour
FactorNode(id, name, factor)         -- an AbstractLenticulumFactor
Edge(factor, channel, variable, direction, coupling)
```

An edge attaches one **named channel** to one variable. Channels are named rather than
positional because a factor has no distinguished input — which is decided per message by
[[Polarity Resolution]].

Built through a mutable `GraphBuilder` (`variable!`, `factor!`, `connect!`, `build`) so the
adjacency indices cannot drift out of sync with the edge list.

## The two predicates

| | on | decides |
|---|---|---|
| `istree(g)` | the **undirected** bipartite graph | message passing is **exact** |
| `isdag(g)` | the **directed** multigraph | the graph is **Lux-compatible** |

`isdag` builds arcs from the directions: emitting gives factor → variable, absorbing gives
variable → factor, **bidirectional gives both** — so a single bidirectional edge is already a
2-cycle. That is not a bug; `isdag(g)` is precisely "this graph could have been written in Lux",
and a bidirectional edge is exactly what "implicit" means.

The supervised test graph is a **tree that is not a DAG**, which is the combination the whole
library exists for. Asserted in the test suite so the distinction cannot quietly collapse.

## `euler_characteristic`

$\chi = |F| + |V| - |E|$; `1` iff a connected tree. It reappears in [[free_energy]] as the sum
of the Bethe counting numbers, and the test suite asserts the two computations agree — the
cheapest possible check that the graph and the free-energy accounting have not drifted apart.

## Implementation difficulties

### 1. `isdag` uses a recursive DFS

`visit` recurses, so a graph deeper than the stack will overflow. Fine for the sizes this is
built for (factor graphs are wide, not deep) but it is an unguarded assumption; an explicit
stack would be strictly better and is a five-line change.

### 2. `FactorGraph` carries two vestigial type parameters

`_v` and `_f` hold the tuples of node names, giving `FactorGraph{V,F}` type parameters that
nothing currently dispatches on. The intent was to make graph-wide `NamedTuple` construction
type-stable; the payoff is not realised because `variables` and `factors` are
`Vector{VariableNode}` / `Vector{FactorNode}` (abstractly typed elements), so the whole
structure is dynamically typed anyway.

**Either commit or remove.** Committing means making the node vectors tuples and the whole graph
a compile-time object, which would make `infer!` type-stable and is probably right for small
graphs but bad for large ones. Removing means deleting `_v`/`_f`. Leaving it half-done, as now,
is the worst option and is recorded so it is not mistaken for a working optimisation.

### 3. A channel is a port, not a bus

`connect!` refuses to attach the same `(factor, channel)` twice. Fan-out is achieved by
connecting *several factors* to one variable, which is the graph-level form of the copier of
[[Copiers Cups and Caps]] — and a variable of degree $d$ **is** a copier. This is why the Bethe
counting number corrects by exactly $d_v - 1$: the number of extra consumers.

### 4. `isconnected` assumes at least one variable

The traversal starts at `(:variable, 1)`. A graph of factors with no variables is handled by a
special case, but a graph with variables in one component and factors in another is only caught
by the final `all(seen_v) && all(seen_f)`. Correct, but the special-casing is inelegant enough
to be worth flagging.

### 5. `learnable_factors` is the only place the traits are used so far

`LenticulumCore.islearnable` is consulted to list which factors need an optimiser. `isfrozen` is
declared but not yet consulted anywhere — the distinction (frozen factors still propagate
cotangents; non-learnable ones have no parameter wire) matters only once the gradient machinery
exists.

Related: [[Factor Graphs]], [[polarity_resolution]], [[free_energy]]
