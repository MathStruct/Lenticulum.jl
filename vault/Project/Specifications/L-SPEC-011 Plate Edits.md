#design

> **Status: planned (2026-11-04, Claude); not implemented.** Related: [[L-SPEC-010 Structure Edits]], [[L-SPEC-006 Plates]], [[L-SPEC-009 Cluster Nodes]], [[Feature Backlog]]. Code (today's workaround): `lib/Mycelium.jl/src/edits.jl` (`GraphBuilder(g)`, which refuses plates), `lib/Mycelium.jl/src/plates.jl` (`as_plate`).

> Sources: original to this repository. code: `lib/Mycelium.jl/src/edits.jl`, `lib/Mycelium.jl/src/plates.jl`; examples: `examples/structure_edits.jl`, `examples/structure_clusters.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

Structure edits of a graph with plates, applied to the plate's template once rather than to every
replica by name, and a node numbering that survives edits. Today an edit unrolls the plate into
named replicas (`GraphBuilder(g)` refuses plates, so the caller builds the graph without one),
edits every replica, rebuilds the whole graph, and `as_plate(g, name, n)` reconstructs the plate
by replaying the named replicas through `plate!`. This specification proposes the replacement. It
leaves open the representation of removed nodes (§4) and whether cluster nodes are edited directly
(§6).

**Why, and how much.** In `examples/structure_edits.jl` with EM fits (200 replicas, 2026-11-04,
an instrumented scratch copy), the edits took 3.9 s and clustering plus `as_plate` 2.9 s of a
35-second search; the fits took the rest. The edits themselves are linear in the replicas, but each
one rebuilds the graph and replays the plate. The gain is therefore at most about 15% of that search
today, and it grows as fits get faster. A second motivation is correctness by construction: a
template edit cannot make the replicas disagree, while an edit by name can, and `as_plate` then
refuses the graph.

## The contract (proposed)

**§1. Builders of plated graphs.** `GraphBuilder(g)` accepts a graph with plates. Nodes outside
plates are edited as in L-SPEC-010 §1. An edit by name of a single replica's node is refused with
an `ArgumentError` naming the plate, since it would break the uniformity L-SPEC-006 §2 requires. An
explicit `unplate!(b, plate)` drops the plate record, keeps the replicas as ordinary nodes, and
allows such edits.

**§2. Template edits.** Each applies to every replica of plate `plate` in one call, and keeps the
plate record (`plates(g)`) consistent:

- `plate_variable!(b, plate, name, space)` adds a variable to every replica (`replica_name(name, i)`);
- `plate_factor!(b, plate, name, factor; params, freeze, rule, tie = true)` adds a factor to every
  replica, tied by default as in `plate!`;
- `plate_connect!(b, plate, factor, channel, variable; direction, coupling, detach)` connects every
  replica's `factor` to its `variable` (a template name), or to a shared node outside the plate;
- `plate_remove_factor!`, `plate_remove_variable!`, `plate_disconnect!` and
  `plate_replace_factor!` mirror L-SPEC-010 §1 on every replica;
- `plate_resize!(b, plate, n)` adds or removes replicas at the end. A new replica copies the
  template, and its data factors take data from a caller-supplied function of the replica index.

A template edit must leave every replica with the same nodes and edges in the same order. That
holds by construction, and `build` checks it as `plate!` does.

**§3. Shared nodes.** A template edit may connect to nodes outside the plate (a global variable,
say). Removing such a node is refused while a plate connects to it, unless the plate's edges to it
are removed in the same builder.

**§4. Stable ids.** Node and edge ids survive edits: an edit appends new nodes and marks removed
ones dead instead of renumbering. Passing over dead entries is cheap: schedules, stores and plate
records skip them. An explicit `compact(g) -> (g′, map)` renumbers and returns the map from old to
new ids. With stable ids `transfer(s, g, g′)` is the identity on surviving edges, with no lookup by
name. Whether dead entries are masks in the existing vectors or a free list is open.

**§5. Messages and blocks.** A store carried across a template edit keeps its block rows
(L-ADR-0010) for unchanged template positions. A new template position starts empty, and a removed
one is dropped. Replica columns are unchanged unless `plate_resize!` changes `n`, which appends or
truncates columns.

**§6. Cluster nodes.** Template edits apply to the unclustered graph. `cluster(g, plate, name,
factors)` is then rerun, which is cheap next to fitting: it builds one cluster per replica from
the template. Editing inside a cluster node directly (adding a factor to a cluster) is out of
scope.

**§7. Equivalence.** A sequence of template edits produces a graph with the same nodes, edges and
plate records, after `compact`, as building the edited structure directly with `plate!`. Its free
energy and messages are then the same to rounding.

## Checked by (planned)

- Template edits against direct construction, for adding and removing a switch and resizing a
  label (the two structure examples' moves): nodes, edges, plate records after `compact`, and free
  energies to $10^{-14}$.
- `transfer` across a template edit keeps every surviving message, and the warm start reaches the
  cold start's fixed point (as `test/structure_edits.jl` does by name today).
- Refusals: an edit by name of one replica, removing a shared node a plate still uses, replicas
  that disagree after `unplate!` edits when `build` re-plates them.
- `examples/structure_edits.jl` and `examples/structure_clusters.jl` rewritten with template edits
  give the same search paths and scores, faster by the edit and `as_plate` time measured above.

## Open

- Masks or a free list for dead ids (§4), and what `nfactors`/`nvariables` count.
- Whether template edits should also work on graphs with several plates sharing nodes (today one
  plate per clustered graph is supported, L-SPEC-009 §8).
- Priority: low while fits dominate a search (§Scope); see [[Feature Backlog]].
