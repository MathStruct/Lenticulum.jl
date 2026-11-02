#implementation

> Plates: a template replicated over $n$ observations in one unrolled graph, with tied parameters,
> a record of the replicas, and schedules that run the same template task for every replica in
> one call. Equal to the loop of $n$ graphs in marginals, free energy and gradients
> (`test/plates.jl`); 2.1 to 2.5 times faster than that loop for factor analysis
> (`bench/plates.jl`), still 430 to 740 times slower than the same E-step written by hand.

> Sources: original to this repository (design: [[Extending Mycelium]] §2). code: `plates.jl`, `graph.jl` (`parameter_key`, `parameter_keys`, `tied_factors`), `passing.jl` (`factor_messages`), `free_energy.jl` (`free_energy_gradient`, `parameter_step`), `../../../src/constraint.jl` (the batched `factor_messages`); tests: `../../../test/plates.jl`; benchmark: `../../../bench/plates.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

Implements: [[L-SPEC-006 Plates]].

## Parameter keys and tying

Each `FactorNode` has a **parameter key** (`params`, `parameter_key(node)`): the entry of `ps` and
`st` it reads. It is the node's name unless `factor!(...; params = key)` says otherwise, so a graph
without ties behaves exactly as before. Nodes with one key are tied:

- `LuxCore.initialparameters(rng, g)` has one entry per key (`parameter_keys(g)`), initialised by
  the first node that reads it;
- every message, energy and gradient path reads `ps[key]` and `st[key]` (`step!`,
  `factor_free_energies`, the replay and implicit adjoints, the sampler);
- the default gradient driver evaluates each node's fixed belief (LP-020) or its mean-field
  marginals outside AD, then differentiates the **sum** of the tied nodes' scalars in one AD call
  per key (`_tied_scalar`). The coupled drivers differentiate the replay with respect to the whole
  of `ps`, so tied entries sum there without any change;
- `parameter_step` has one optimiser state per key.

`build` checks that tied nodes have the same factor type and the same `freeze` (both decide which
coordinates are active), and that a key is not the name of a node reading another key, which
would make `ps[name]` ambiguous. Tied nodes may differ in their other fields: replicas of a
`DataFactor` hold different data, and a `LinearConstraintFactor` may hold a different noise
covariance (its batched messages check).

Tied nodes share their **state** entry too, and must be stateless: `factor_free_energies` (and so
the free energy, `parameter_step` and every gradient driver that reports state) throws when a
tied key has a non-empty state, since which node's update were kept would depend on the order of
evaluation. Tied factors must also have parameter trees of one shape, since the shared entry is
initialised by the first node; that is not checked at build time (a mismatch usually fails as a
dimension error at the first message).

## Plates

`plate!(template, b, name, n)` calls `template(p, i)` for each replica on a `PlateBuilder`, which
adds nodes named `name[i]` (`replica_name`) and resolves a name the replica did not declare to a
node of the enclosing graph, shared by every replica. Factors are tied by default
(`factor!(p, ...; tie = false)` gives each replica its own key, `replica_name(params, i)`). The
replicas must declare the same nodes and edges in the same order, with the same channels,
directions, couplings, detach flags and shared nodes; the check compares a signature per edge.
The graph keeps a `Plate` record: node and edge ids as matrices, one row per template position
and one column per replica. Plates do not nest.

The graph is **unrolled**. Every replica is an ordinary set of nodes, so nothing else in Mycelium
needed to know about plates: inference, the free energy, sampling and every gradient driver are
those of any graph. A shared parameter leaves the replicas as separate components; a shared
variable couples them (`test/plates.jl` has both).

## Schedules and grouped execution

A `PlateSchedule` is a list of groups, each the same template task in every replica (or one task
outside the plate). `plate_schedule(g, replica_tasks)` lifts a schedule written for replica 1;
`plate_schedule(g)` gives each component its tree schedule and lifts replica 1's, which needs the
replicas to be separate components.

A sweep gives the messages of `SequentialSchedule(tasks(sched))`. Within a group:

- **variable → factor** messages read only factor → variable messages, which no member writes, so
  they are computed in one loop without `step!`;
- **factor → variable** messages of a belief-propagation node are computed in one call of
  `factor_messages(factors, target, polarity, inputs, priors, ps, st)`, with the polarity resolved
  once, when the members are tied, have the same available channels, and target **different**
  variables. A shared target would see the earlier members' messages in its `prior` argument in
  the sequential order, so such a group runs member by member, as do mean-field nodes;
- `factor_messages` defaults to `factor_message` for each replica in turn. `LinearConstraintFactor`
  overrides it: when the replicas share their noise and every replica's inputs have the same
  covariances and flat directions (data and a shared prior), the metric, its Cholesky factor and
  the rank test are computed once, and each replica costs one matrix-vector product
  (`constraint.md`).

Checked in `test/plates.jl`: two grouped sweeps give the sequential schedule's messages to
$10^{-12}$ on every edge, in both directions.

## Speed

`bench/plates.jl`, the E-step of factor analysis ($d = 3$, $k = 2$), minimum of five runs:

| $N$ | graph per point, built each time | graph per point, prebuilt | plate, sequential | plate, grouped | by hand |
|---|---|---|---|---|---|
| 100 | 2.86 ms | 1.55 ms | 1.86 ms | 0.64 ms | 0.0015 ms |
| 300 | 8.67 ms | 4.56 ms | 7.72 ms | 1.85 ms | 0.0032 ms |
| 1000 | 29.48 ms | 14.77 ms | 31.11 ms | 7.02 ms | 0.0095 ms |

The plate's posterior means equal the hand-written E-step to $8 \cdot 10^{-16}$. The grouped plate is
2.1 to 2.5 times faster than a loop over prebuilt graphs and 4.2 to 4.7 times faster than building a
graph per point; it is 430 to 740 times slower than the closed form. The plate on a sequential
schedule is no faster than the loop (and slower at $N = 1000$; not investigated).

What is left is Mycelium's per-message overhead, not linear algebra. A profile of the sequential
plate sweep (300 points) put about a quarter of the time in `resolve_polarity`, which builds a
`NamedTuple` type at run time, a quarter in the factor's message (about a quarter of that in the rank
test's SVD), and most of the rest in dynamic dispatch on the untyped node and edge vectors and in
`available_channels`. Grouping removes the per-replica polarity and rank work; the dispatch
remains. The target of [[Extending Mycelium]] §2, within a small factor of hand-written code, is
not met.

## What it does not do

- No batched data structure: messages are still one belief per edge. A factor that wants real
  vectorisation implements `factor_messages` and stacks inside it.
- No nested plates, no plate over a sub-plate, no plates of different sizes sharing a template.
- No batched gradient beyond summing the tied scalars in one AD call; the beliefs are still
  evaluated per node.
- Mean-field groups and shared-target groups run member by member.
- Tied factors must be stateless.

## Difficulties

1. **The first version passed the first replica's factor to every replica.** Tying shares the
   parameters, not the factor object, and replicas of a `DataFactor` hold different data. The
   benchmark showed it at once: the plate's free energy was off by 301 nats against the
   sequential schedule. `factor_messages` now takes the vector of the replicas' own factors.
2. **Independence within a group is not only about what is written.** The variable → factor and
   factor → variable writes of a group never overlap, but a factor's message also reads the
   `prior` at its target (the excluded marginal), which includes the earlier members' messages
   when the target is shared. Hence the distinct-target condition.
3. **`adjoint = :implicit` on the factor-analysis plate disagreed with every other driver.** It does
   on a single graph too, with the default couplings: it was recorded as
   [[LP-021 The coupled gradient drivers cut diagonal edges at fixed messages]], not a plate
   problem.
