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

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
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
