#implementation

> Belief-message schedules and their graph constraints; realised-schedule
> parameter derivatives are defined separately.

> Sources: code: `schedules.jl`
>
> Theory (CT-ML wiki): [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)

Implements: the schedule types and their generation from graph structure. Pure combinatorics —
this file computes *which* messages in *what* order, and nothing else.
Theory: [[Schedules]].

## Two atomic tasks

`:to_factor` on edge $e$ (legal iff the edge absorbs) and `:to_variable` on $e$ (legal iff it
emits). `islegal(g, task)` is the predicate; every generator filters on it.

## The generators

| generator | requires | produces |
|---|---|---|
| `flooding_schedule` | nothing | all legal tasks, double-buffered |
| `tree_schedule` | `istree` | inward (reverse BFS) + outward (BFS) |
| `forward_backward_schedule` | `isdag` or `istree` | topological sweep for a DAG; tree inward/outward sweeps otherwise |
| `spanning_tree_schedule` | nothing | off-forest messages, then one or more forests’ inward/outward sweeps in place (BFS or weighted Kruskal) |
| `AndersonSchedule(inner; m)` | a static `inner` | `inner`'s sweep with Anderson mixing of the last $m + 1$ sweeps (`anderson.md`) |

`tree_schedule` roots at a leaf, BFS to record which endpoint of each edge is the *child*, then
emits child→parent messages in reverse BFS order and parent→child in BFS order. That ordering is
what makes each message computable exactly once from already-final inputs.

## Historical gradient limitation (2026-10-26; superseded below)

The schedule docstrings describe a planned cotangent pass, not an implemented one. Only
`DiagonalCoupling()` is accepted at graph construction; other annotations raise
`ArgumentError` (the Mycelium `LP-011 graph construction guards` testset). Scheduling and
message execution are unchanged by these guards; see [[graph]] and [[statistical_game]].

## Implementation difficulties

### 1. Pruning — the bug the tests found

The first version emitted all $2|E|$ tree tasks and threw a `PolarityError` at the loss node:
an `Absorbing` edge cannot carry a factor → variable message. Now the generators filter, and
`TreeSchedule.pruned` counts what was dropped.

Surfacing the count rather than swallowing it is the important part. `pruned > 0` is *normal* —
it means part of the graph is explicit rather than implicit, and a loss factor genuinely has
nothing to say to its input in belief terms. But it is also exactly what a **mis-wired** graph
looks like (an implicit factor accidentally given a unidirectional edge), and the two are
indistinguishable without the number.

In the supervised test graph, 4 of 12 messages are pruned: two `Emitting` data edges and two
`Absorbing` loss edges each lose one direction.

### 2. `forward_backward_schedule`'s DAG backward list is empty

`isdag` excludes any graph with a bidirectional edge, and the backward sweep only covers
bidirectional edges. Thus the DAG branch always returns `backward = []`.

That is **correct**, and it is the sharpest available statement of the Lux/Lenticulum boundary:
a strictly unidirectional DAG has no backward *belief* flow, only cotangent flow. But it does
mean the DAG branch cannot use that field. The separate tree fallback wraps
`tree_schedule(g).inward` and `.outward`, so a bidirectional tree can have a nonempty
backward list. The previous note's claim that it was empty in every reachable case
was stale; `forward_backward_schedule` explicitly contains this fallback.

### 3. Residual scheduling evaluates all candidates

The previous note incorrectly said this type had no executor. `passing.jl`
already has a priority loop: probe every legal task, commit the largest residual,
break ties by oldest message (then task order), and stop early at zero residual.
A sweep commits at most `min(maxtasks, length(all_tasks(g)))` tasks. This costs
quadratic message evaluations per full sweep. Unsupported distances returning
`Inf` prevent numerical convergence certification. Probe state updates are
discarded; factors must use functional state.

### 4. Tree rooting is arbitrary and it matters

`tree_schedule` roots at `first(leaves(g))`. On the supervised graph that is a `DataFactor`,
which cannot absorb — so the entire inward sweep towards it is wasted work, and the useful
messages all happen in the outward sweep or at the far end.

Rooting at a **sink** (a loss) would be better: information would flow inward towards the thing
that consumes it. There is no principled rule implemented, and `leaves` returns nodes in id
order, so the choice is effectively "whatever was declared first". A `root = :loss` keyword is
supported but undocumented at the call site, and choosing a good default root is an open
question.

### 5. `tasks(::TreeSchedule)` allocates a `vcat` on every sweep

`sweep!` calls `tasks(sched)`, which for `TreeSchedule` and `ForwardBackwardSchedule` builds a
fresh concatenated vector each time. Trivial to fix (iterate the two vectors), noted because it
is in the inner loop.

Related: [[Schedules]], [[passing]], [[graph]]

## LP-011 implementation (2026-10-27)

Static sequential/tree/forward-backward schedules can be replayed
by the parameter gradient driver, with mixed boundary cuts. Flooding/residual replay
currently throws rather than changing their update semantics. Ordinary scheduling is
unchanged; the old statement that non-default annotations fail at construction is historical.

Evidence: `lib/MyceliumLearning.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].


## Batch C schedule derivatives (2026-10-28)

[[gradient_couplings]] now replays flooding with an immutable sweep snapshot and
residual schedules with recorded committed task choices. Damping is inside each
replayed outgoing update. The residual derivative is piecewise-defined: its
chosen sequence and stopping decisions are fixed. The implicit adjoint in
[[graph_adjoint]] uses these schedules only to find and verify a BP fixed point.
Tests: `test/graph_adjoint.jl`, `test/graph_adjoint_devices.jl`.

The final API build exposed a misplaced docstring: the `tree_schedule` help
block preceded `_resolve_tree_root`, so Documenter attached it to that internal
helper and could not resolve a reference to the public generator. The block now
precedes `tree_schedule` itself. This corrects help/API rendering without changing
either function body.

`ResidualSchedule` rejects `maxtasks < 1` (LP-019 item 9): a sweep of no tasks computes no
message and has residual 0, which `propagate!` read as convergence, so a unary `DataFactor`
graph returned `TrivialBelief()` with a converged report.

## Component sweeps (2026-11-02)

The breadth-first inward/outward construction of `tree_schedule` is now `_tree_sweeps(g, root)`,
which covers the component containing `root`; `tree_schedule` checks the whole graph is a tree
and calls it. `plate_schedule(g)` (`plates.jl`) calls it per component.

## Hybrid schedules (2026-11-03)

`hybrid_schedule(g)` ([[L-SPEC-007 Hybrid Schedules]]): the belief-propagation clusters (components
of the graph without its `MeanField()` nodes, found by `_cluster`) each get the inward and outward
sweeps of `_tree_sweeps(g, root; skip)`, which now can skip factor nodes; a sweep runs every cluster,
then each mean-field node's legal messages, each followed by its target's cluster. `_hybrid_tasks`
builds it for a set of variables, so `plate_schedule(g)` can build it for replica 1's component.
`HybridSchedule` sweeps like a `SequentialSchedule`; its only difference is `propagate!`'s default
convergence criterion (`_default_criterion`).

Repeating every cluster's sweep at the start of each sweep costs a pass per sweep but keeps the
schedule stateless: the first pass on a fresh store is the start from the clusters' own factors,
and later ones change nothing that is converged.

## Spanning-tree schedules (2026-11-09)

`spanning_tree_schedule(g; root)` is the first static schedule for loopy graphs besides flooding.
It takes `_tree_sweeps` from the root (and from a node of every other component; `_tree_sweeps`
already ignores an edge that closes a loop, since its far end has been seen), puts the messages of
the edges left out first, then every component's inward sweep, then every outward sweep, and returns
a `SequentialSchedule`. On a tree it is `tree_schedule`'s lists (tested), but `propagate!` iterates
it instead of stopping after one sweep, because it is not a `TreeSchedule`.

Why: on the resistive divider (one loop of six edges) every slow eigenvalue of a flooding sweep
has modulus $\rho$ and a message needs six sweeps to go round the loop; an in-place order that goes
round in one sweep converges at $\rho^6$ per sweep for the same 18 evaluations. Measured in
`bench/bp_schedules.jl` (vault: Composing Optimisers §4.2): 0.17 to 0.18 of flooding's message
evaluations to an error of $10^{-10}$ in four resistor settings; with `infer!` and `tol = 1e-11`
on the test's divider, 82 sweeps against flooding's 463 (`test/runtests.jl`, "a spanning-tree
schedule", which asserts fewer than a quarter and the same marginals).

What it does not do:
- **No guarantee.** Loopy BP stays loopy; the fixed points are flooding's, and whether it converges
  where flooding does not is untested.
- **Tree choice and alternating trees** were added by I3 (2026-11-18), described below.
  The default remains one breadth-first tree.
- **Mean-field nodes are not separated**: their messages are scheduled like any other.
  `hybrid_schedule` is the schedule for those graphs.
- **Plates**: `plate_schedule` is not built on it.


## Choice of tree and alternating trees (2026-11-18)

The heuristic is to put strong couplings inside a tree, so an inward/outward sweep propagates
those dependencies across the component before returning to the weaker off-tree connections.
This is a heuristic, not an optimality claim for convergence speed. A maximum-weight forest
maximises the sum of its edge weights; it does not minimise the spectral radius of a BP sweep.
For the Gaussian constraints measured here the proposed weight is the coupling precision `1/q`;
`q` is the opposite preference, included as a control.

Tree-based reparameterisation (Wainwright, Jaakkola & Willsky, 2003,
`docs/src/refs.bib`, `wainwright2003trp`) views tree updates as reorganising the same distribution's
local factors and propagating information exactly through selected trees. Here the implementation
only composes ordinary BP message updates in static tree orders. It does not introduce tree
appearance probabilities, fractional counting numbers, or a tree-reweighted objective.

`_spanning_forests` returns structural edge masks, independently of message legality. Tree 1 is
the original breadth-first forest without weights; otherwise Kruskal with a path-compressed
union-find selects a maximum-weight forest. An edge inherits its factor's weight. Equal weights
prefer lower edge indices. Later forests order edges by increasing earlier use count, decreasing
weight, then increasing index. `:cover` adds forests until every edge is covered: the first unused
edge in each Kruskal order joins distinct singleton sets and is accepted, so progress is guaranteed.
On an acyclic graph the first forest already covers every edge.

Each forest is rooted at the resolved `root`, then the lowest-index unseen variable, then the
lowest-index unseen factor in the other components. `_tree_sweeps` accepts an `edge_in` predicate;
its default leaves the tree, plate, and hybrid callers unchanged. The returned sequential task
list starts with legal tasks outside every forest, followed by each forest's inward tasks of all
components and then its outward tasks. A shared task appears once per forest. The existing
`sweep!` maximum over every evaluation is unchanged; a later small update cannot hide an earlier
large residual. K trees cost approximately K tree sweeps; fewer sweeps need not mean fewer message
evaluations or less wall time.

**Measured** (`julia --project=bench bench/bp_trees.jl`, 2026-11-18; output kept in
`~/.cache/lenticulum/i3/bp_trees.out`). 10 × 10 Gaussian grids, prior variance 100, each grid
constraint with its own noise `q`: isotropic `q = 1`; "horizontal" `q = 0.01` on horizontal edges
and 1 on vertical; "vertical" the transpose; "random" `log10 q ~ U[-2, 1]` (seed 20261118).
`infer!` to a residual of `1e-11`, at most 20 000 sweeps. Message evaluations = sweeps × the
schedule's task count (a task in two trees counts twice); "precision" is `weights = 1/q`
(priors weight 1), "q" the minimum-weight control `weights = q`, "2" is `ntrees = 2`. On these
grids `ntrees = 2` already covers every edge, so `:cover` gave the same rows.

| message evaluations (sweeps) | isotropic | horizontal | vertical | random |
|---|---:|---:|---:|---:|
| flooding | 6 472 200 (7 035) | > 18.4 M (not conv.) | > 18.4 M (not conv.) | > 18.4 M (not conv.) |
| BFS | 1 017 520 (1 106) | 5 882 480 (6 394) | 2 186 840 (2 377) | 4 063 640 (4 417) |
| precision | = BFS | 2 139 000 (2 325) | = BFS | 2 369 920 (2 576) |
| q (control) | = BFS | = BFS | 6 376 520 (6 931) | 5 761 960 (6 263) |
| BFS, 2 trees | 1 631 216 (1 076) | 10 980 388 (7 243) | 2 994 100 (1 975) | 5 289 324 (3 489) |
| precision, 2 trees | = BFS, 2 trees | 2 862 208 (1 888) | = BFS, 2 trees | 3 191 180 (2 105) |
| flooding + A(3) | 686 320 (746) | 4 278 000 (4 650) | 6 060 960 (6 588) | 2 150 040 (2 337) |
| BFS + A(3) | 107 640 (117) | 792 120 (861) | 161 920 (176) | 196 880 (214) |
| precision + A(3) | = BFS + A(3) | **147 200 (160)** | = BFS + A(3) | **67 160 (73)** |
| q + A(3) | = BFS + A(3) | = BFS + A(3) | 1 128 840 (1 227) | 327 520 (356) |
| precision, 2 trees + A(3) | 213 756 (141) | 191 016 (126) | 174 340 (115) | 86 412 (57) |

Every converged run's largest mean error against the dense solve is below `6e-10`. "= BFS" rows
are the same forest, not a coincidence of counts: with equal weights (isotropic) Kruskal's
lowest-index tie-break picks the breadth-first tree on this grid, and on the vertical grid the
corner-rooted BFS tree already holds every strong (vertical) edge, so it is a maximum-weight tree
(checked by comparing `_spanning_forests` masks, 2026-11-18). On the horizontal grid BFS holds
the weak direction (tree weight 10 180 against the maximum 18 199) and is 2.8 times (plain) and
5.4 times (with Anderson) slower than the precision tree; the minimum-weight control is never better than BFS.

So: **weight the tree by coupling strength** (1.7–5.4 times fewer evaluations where BFS picks badly,
never worse here); **alternating trees did not pay**: they cut sweeps by up to 20 % (with BFS on the horizontal grid they added sweeps) but cost
1.65 times the evaluations per sweep, so in evaluations they lose to one tree in every setting, with and without
Anderson. The 20 × 20 random grid was started and stopped (flooding alone needed 250 s for 20 000
sweeps); it is not measured. Wall times are in the output file but ran beside test suites, so
they are not recorded here.

Verification: `test/runtests.jl` ("weighted and alternating spanning forests") checks relay graphs,
weights and their errors, rooting, disconnected components, coverage and the task count.
`lib/Lenticulum.jl/test/i3_spanning_trees.jl` checks one sweep on random Gaussian trees (exact
means and covariances), one sweep of a weighted tree on loopy graphs whose off-tree constraints
carry no information (exact; this test fails if the weighted tree's order is replaced by the
BFS order), and `infer!` on diagonally dominant loopy graphs with every option and with Anderson
(exact means).

Difficulties: connectivity must be tracked on the structure before legal-task pruning (a forest
rebuilt from the surviving messages loses edges whose tasks are both illegal). Weight ordering
compares directly instead of negating, so unsigned weights keep the descending order.

Default weights (I25): Mycelium does not know a factor's coupling strength; Lenticulum's
`coupling_weights(g, ps)` computes one for its linear-Gaussian factors (`lib/Lenticulum.jl/src/coupling.md`).
What this does not do: no
adaptation of trees during inference; no tree-appearance probabilities or tree-reweighted
objective (the fixed points are ordinary BP's; TRP is cited as motivation, no theorem used); no
convergence guarantee; plates and mean-field nodes as before (`plate_schedule`,
`hybrid_schedule` do not take weights). Untested: graphs where BP oscillates, non-Gaussian graphs.
