# Schedules

> In Lux there is one order and the wiring implies it, so scheduling is invisible. On a
> general graph it is the entire problem.

## Two atomic operations

| task | what it does | legal iff |
|---|---|---|
| `:to_factor` on edge $e$ | pool the *other* messages at the variable and hand the result to the factor | the edge **absorbs** |
| `:to_variable` on edge $e$ | resolve a polarity with $e$'s channel `Unobserved()`, assemble, invert | the edge **emits** |

Everything else is an order over these.

## The schedule zoo

| schedule | applies to | guarantee |
|---|---|---|
| `TreeSchedule` | trees | **exact in two sweeps** |
| `ForwardBackwardSchedule` | DAGs | exact; equals a Lux forward pass |
| `SequentialSchedule` | anything | none; in-place, propagates fast |
| `FloodingSchedule` | anything | none; order-independent |
| `ResidualSchedule` | loopy graphs | none, but converges more often |

### `TreeSchedule` — the only one with a guarantee

Root the tree, sweep inward (children strictly before parents), then outward. Each edge carries
one message per sweep. After the outward sweep every `marginal` is the **true** marginal and
the free energy is the **true** free energy.

This is the classical result, and it is why [[Factor Graphs]] insists that `istree` — a
property of the *undirected* graph — is the one that matters for correctness.

### Pruning: directed edges shorten the sweeps

A tree schedule wants $2|E|$ messages. It does not always get them: an `Emitting` edge cannot
carry a variable → factor message and an `Absorbing` edge cannot carry a factor → variable one.
So the sweeps are filtered, and `TreeSchedule.pruned` counts what the directions forbade.

In the supervised example of [[Factor Graphs]] — two `Emitting` data edges and two `Absorbing`
loss edges — exactly **4 of the 12** messages are pruned.

> [!warning] Pruning weakens exactness, deliberately
> A `LossFactor` on absorbing edges never tells its input variable anything, because in
> *belief* terms a sink is genuinely uninformative about what it consumes. What a loss tells
> its input is a **cotangent**, and cotangents travel by the
> [[Composition of Gradients|`AbstractGradientCoupling`]] machinery, not by this scheduler.
>
> So `pruned > 0` does not mean the schedule is broken; it means part of the graph is
> explicit rather than implicit. It is surfaced as a field rather than swallowed so that a
> genuinely mis-wired graph — an implicit factor accidentally given a unidirectional edge —
> is visible instead of quietly under-informed.

**This was found by running the tests, not by design.** The first version generated all $2|E|$
tasks and threw a `PolarityError` at the loss node.

### `ForwardBackwardSchedule` — and why its backward list is empty

Topological belief sweep, then a reverse sweep over bidirectional edges. But `isdag` excludes
any graph with a bidirectional edge ([[Factor Graphs]]), so in practice:

$$\texttt{isdag(g)} \;\Longrightarrow\; \texttt{isempty(sched.backward)}$$

**That is correct, and it is the sharpest available statement of the Lux/Lenticulum boundary.**
A strictly unidirectional DAG has *no backward belief flow*; its backward pass carries
cotangents. `isempty(sched.backward)` is therefore a one-line diagnostic meaning *"this graph
is explicit; there is nothing to infer"*.

### `FloodingSchedule` versus `SequentialSchedule`

- **Sequential** executes in place: later tasks see earlier results *in the same sweep*, so
  information crosses a whole path per sweep. Fast, but the answer depends on the order, and on
  a graph with no natural order that dependence is arbitrary.
- **Flooding** is double-buffered: every task reads the previous sweep, all results commit
  together. Information crosses one edge per sweep. Slower, but order-independent.

The trade is not about speed alone. On a loopy graph the sequential order *is* a modelling
choice you did not know you were making.

### `ResidualSchedule`

Send whichever pending message would change the most (Elidan, McGraw & Koller's *residual
belief propagation*). Converges on many loopy graphs where flooding oscillates, because it
prioritises the parts that have not settled.

Needs a usable `belief_distance`. With the current belief types most pairs return `Inf`, so it
degrades to "some order" — recorded rather than hidden. `belief_distance` returning `Inf`
rather than `0` on unknown pairs is deliberate: **a false "converged" is much worse than a
wasted sweep.**

## The schedule is not neutral

Two things depend on it that look like they should not:

1. **Which answer you get on a loopy graph.** Different schedules reach different fixed points,
   or none. See [[Loopy Message Passing]].
2. **The free-energy decomposition.** AutoBayes Theorem 23's recursion refers to "downstream",
   which on a graph is a property of the message order, not of the wiring. The Bethe form has
   no order in it at all — see [[Bethe Free Energy]].

> [!note] The parallel structure is declared and unexploited
> `FloodingSchedule` is double-buffered, so every message in a sweep is independent — yet
> `sweep!` runs them in a sequential loop. And `tree_schedule`'s critical path is twice the
> tree depth, so on a **chain** (a SLAM trajectory) there is no parallelism at all — for which
> there is a known ``O(\log N)`` associative-scan reformulation. See
> [[Parallelism and Compilation]] §3.

Related: [[Parallelism and Compilation]], [[Factor Graphs]], [[Messages are Inversions]], [[Loopy Message Passing]], [[schedules]]
