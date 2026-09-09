# schedules.jl — implementation note

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
| `forward_backward_schedule` | `isdag` | topological sweep + reverse over bidirectional edges |

`tree_schedule` roots at a leaf, BFS to record which endpoint of each edge is the *child*, then
emits child→parent messages in reverse BFS order and parent→child in BFS order. That ordering is
what makes each message computable exactly once from already-final inputs.

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

### 2. `forward_backward_schedule`'s backward list is always empty

`isdag` excludes any graph with a bidirectional edge, and the backward sweep only covers
bidirectional edges. So the two conditions are mutually exclusive and `backward` is always `[]`.

That is **correct**, and it is the sharpest available statement of the Lux/Lenticulum boundary:
a strictly unidirectional DAG has no backward *belief* flow, only cotangent flow. But it does
mean the `backward` field is dead code in every reachable case, which looks like a bug on
reading. Kept, with the test asserting it is empty, because the assertion documents the boundary
better than a comment would.

### 3. `ResidualSchedule` has no execution path

The type and its documentation exist; the priority loop does not. It would be useless without a
working [[messages|`belief_distance`]], which returns `Inf` for most pairs — a residual
scheduler that cannot compare messages degrades to an arbitrary order, which is what
`flooding_schedule` already provides more honestly.

Deliberately left as a documented type rather than a fake implementation.

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
