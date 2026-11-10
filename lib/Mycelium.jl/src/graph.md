#implementation

> Graph construction and topology; unsupported gradient and frozen-factor declarations fail before message passing.

> Sources: code: `graph.jl`
>
> Theory (CT-ML wiki): [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model)

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

An `Emitting()` edge carries no variable-to-factor message. A factor whose
`local_free_energy` reads those messages declares `uses_cavity == true`; graph free energy
then requires `Absorbing()` or `Bidirectional()` on every attached edge. In particular a prior
that also emits needs `Bidirectional()` for free-energy evaluation; see [[free_energy]].

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

### 2. `FactorGraph` carried two vestigial type parameters (removed 2026-11-02)

They made the graph's type grow with the graph, so a plate of each size recompiled every method
that took the graph (`plates.md`, "What changed after the review"); `FactorGraph` is now one
concrete type. The original note:

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
[[Open Models and Latent Channels]] — and a variable of degree $d$ **is** a copier. This is why the Bethe
counting number corrects by exactly $d_v - 1$: the number of extra consumers.

### 4. `isconnected` assumes at least one variable

The traversal starts at `(:variable, 1)`. A graph of factors with no variables is handled by a
special case, but a graph with variables in one component and factors in another is only caught
by the final `all(seen_v) && all(seen_f)`. Correct, but the special-casing is inelegant enough
to be worth flagging.

### 5. Historical construction guards (2026-10-26; superseded below)

`connect!` and the `Edge` inner constructor accept only `DiagonalCoupling()`. Other coupling
values throw `ArgumentError` explaining that per-edge gradient couplings are not implemented
(LP-011, L-SPEC-002), so their settings would be silently ignored. The inner constructor
also guards direct and explicitly parameterized construction. Fields and types are retained.

`build` does not call `validate`; structural validation remains optional. Both call a shared
frozen-factor check, throwing `ArgumentError` naming every graph factor whose `isfrozen` is
true. This includes non-learnable factors: `islearnable` only filters `learnable_factors`,
and neither that filter nor parameter/state setup establishes frozen backward semantics.
Allowing that combination would imply support for behavior the graph does not implement.

The `LP-011 graph construction guards` testset in `../test/runtests.jl` covers both
learnability values, independent build/validate rejection, the named error, default graphs,
all three non-default built-in coupling families, score-function baselines, a custom subtype
and direct/parameterized `Edge` rejection. Failed `connect!` leaves the builder reusable.
These are construction/validation checks only: message passing is unchanged, and accepting
the default coupling does not itself insert stop-gradients into AD.

Related: [[Factor Graphs]], [[polarity_resolution]], [[free_energy]]

## LP-011 implementation (2026-10-27)

`Edge` now includes `detach::Bool`. `connect!` accepts all four built-in
couplings. `FactorNode` includes graph-local `freeze` paths; its old three-argument
constructor remains valid. `factor!` checks path syntax, and gradient/update entry points
check paths against actual parameters. Build/validate no longer reject `isfrozen=true`.
The construction-time rejection account above describes the 2026-10-26 option-2 state.

Evidence: `lib/MyceliumLearning.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].


## Device freeze masks (2026-10-27)

`_factor_parameters` delegates coordinate pinning to a storage-specific helper.
CPU keeps its prior comprehension; GPUArraysCore uses broadcast masks and
`stop_gradient` for frozen entries. This avoids host scalar reads during raw
static inference AD. Freeze indices are constant metadata. JLArrays replay and
subset/all freeze tests live in `parked/ImplicitLayers.jl/test/device_arrays.jl`;
see [[gradient_couplings]] for coverage limits.

## Parameter keys and plates (2026-11-02)

Each `FactorNode` now carries a parameter key, `params`, the entry of `ps` and `st` it reads: its
name unless `factor!(...; params = key)` ties it to other nodes. `parameter_keys(g)` lists the
distinct keys in order of first appearance; `LuxCore.initialparameters`, `initialstates` and
`parameterlength` have one entry per key, from the first node reading it (`_key_node`).
`tied_factors(g, key)` lists the nodes of one key. `build` calls `_check_ties`: tied nodes must
have the same factor type and `freeze`, and a key may not be the name of a node that reads
another key. The 3-, 4- and 5-argument `FactorNode` constructors set the key to the name, so code
that builds nodes by hand is unchanged.

`FactorGraph` and `GraphBuilder` also hold the graph's plates (`plates.jl`, `plates.md`).
`connect!` checks for a duplicate port in a set (`GraphBuilder.ports`) instead of scanning the edges,
so building is linear in the number of edges. `show` now says "forest" for a disconnected graph whose components are trees (`_ncomponents`), the
usual shape of a plate with shared parameters only; it said "loopy" before.

## Parameters of an unfrozen node (2026-11-04, LP-023)

`_factor_parameters` returns `ps` directly for a learnable node with nothing frozen. Before, it
built the offset tree and index of every parameter coordinate on each message, only to find them
all active. Frozen and non-learnable nodes take the old path. Not timed.

## Pinning frozen coordinates (2026-11-04)

`_pin_coordinates` and `_mask_gradient` tested `i in indices` for each coordinate, a search of the
index vector, so pinning was quadratic in the parameter count on every message of a frozen node.
They now build a mask (`_active_mask`, outside AD), and `_active_indices` caches the index by the
freeze and the shape of the parameter tree. In `examples/structure_edits.jl` (a few dozen
parameters per node) neither is measurable: 192.5 s with the frozen sum, 195.0 s without.

