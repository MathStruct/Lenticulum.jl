#implementation

> Plates: a template replicated over $n$ observations in one unrolled graph, with tied parameters,
> a record of the replicas, and schedules that run the same template task for every replica in
> one call. Equal to the loop of $n$ graphs in marginals, free energy and gradients
> (`test/plates.jl`). With the messages held as blocks (`blocks.md`, since 2026-11-03) the
> factor-analysis E-step is about 9 times the hand-written one at $N = 1000$, down from 240
> (`bench/plates.jl`).

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

`bench/plates.jl`, the E-step of factor analysis ($d = 3$, $k = 2$): one fresh store and one sweep,
minimum of five runs, the posterior means checked against the closed form outside the timings:

| $N$ | graph per point, built each time | graph per point, prebuilt | plate, sequential | plate, grouped | by hand |
|---|---|---|---|---|---|
| 100 | 2.80 ms | 1.48 ms | 1.41 ms | 0.24 ms | 0.0014 ms |
| 300 | 8.39 ms | 4.49 ms | 4.31 ms | 0.71 ms | 0.0038 ms |
| 1000 | 28.30 ms | 15.04 ms | 14.22 ms | 2.39 ms | 0.0099 ms |

The plate's posterior means equal the hand-written E-step to $8 \cdot 10^{-16}$. The grouped plate is
about 6.3 times faster than a loop over prebuilt graphs and 12 times faster than building a graph
per point; it is 170 to 240 times slower than the closed form. The second part of the benchmark
times large plates: building the graph and the scalar free energy after one sweep.

| $N$ | build | free energy, first call | later call |
|---|---|---|---|
| 100 | 0.00 s | 1.311 s | 1.0 ms |
| 1000 | 0.04 s | 0.010 s | 10.3 ms |
| 3000 | 0.05 s | 0.032 s | 33.9 ms |

The first call at $N = 100$ is the method's compilation; later sizes reuse it.

### What changed after the review (2026-11-02)

A review of this file (outside the repository) timed each group and found the costs that were
left. Measured before the changes, on commit 345e03f3, with the same models:

- **The graph's type grew with the graph.** `FactorGraph{V,F}` carried the tuples of all node names
  as type parameters (vestigial, `graph.md` §2), and `scalar_free_energy` went through a
  `NamedTuple` with one field per factor. Every new size recompiled: the first free-energy call
  took 4.1, 6.8 and 14.6 s at $N$ = 100, 1000 and 3000, later calls 13.5, 48 and 311 ms. The type
  parameters are gone, the node lists are vectors rather than tuples, and the scalar free energy
  sums a vector of terms (`_factor_terms`); `factor_free_energies` still returns its keyed
  `GradedEnergy`.
- **Building was quadratic**: `connect!` scanned every edge for a duplicate port (2.2 s at
  $N = 3000$). The builder keeps a set of connected ports.
- **The batched groups redid per replica what the plate fixes once.** In the likelihood group
  (4.3 ms of 7 at $N = 1000$), building each replica's input `NamedTuple` by searching its edges
  took 3.0 ms, the legality check 1.3 ms and the unused priors 1.2 ms. `plate_schedule` now stores
  a `_GroupPlan` per lifted factor → variable group (each member's factor, its edges by channel,
  one channel layout and direction pattern, distinct targets, all checked when the schedule is
  built); the sweep reads availability through those edges, resolves and checks the polarity on
  the first member only, and passes the priors as a lazy vector that is evaluated only if read.
  Inside `LinearConstraintFactor`'s method, the per-replica input lists are built over a
  compile-time tuple of channels (the filtered comprehension cost half the group), the means are
  stacked into one matrix product, and the precision is shared.
- **`GaussianPrior` converted its covariance for every replica.** It now has a batched method
  that converts once when the replicas share mean and covariance (`gaussian.md`).
- **The benchmark timed the readout too** (a `Symbol` and a dictionary lookup per replica);
  it now times the sweep only.

One claim of the review was wrong: that `_step_group!` built a `Vector{Any}` of factors and so
never reached the batched `LinearConstraintFactor` method. Julia's comprehensions take the type
of the elements, the vector was concrete, and the method was dispatched; `test/plates.jl` now
checks which method each planned group dispatches to.

What was left was spread out: the variable → factor groups (one `excluded_marginal` each), the
commits, the input gathering, and the factor methods, all through untyped `Vector{Any}` message
storage. Closing the remaining factor of about 200 needed a batched message representation:
[[L-ADR-0010 Plates get batched message blocks]], implemented in `blocks.jl` (`blocks.md`). The
tables above are those of before; the grouped plate now takes 0.09 ms at $N = 1000$, and the
free energy 1.1 ms (the table in `blocks.md`).

## What it does not do

- Batched messages only as `blocks.md` describes: Gaussian blocks with a shared precision, points,
  and per-replica `VectorBlock`s for everything else.
- No nested plates, no plate over a sub-plate, no plates of different sizes sharing a template.
- The batched gradient (one item per plate factor row, `blocks.md`) needs the row's messages as
  blocks and a factor type with `factor_beliefs`; other rows sum the tied scalars per node in one
  AD call.
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

## Hybrid templates (2026-11-03)

`plate_schedule(g)` uses `_hybrid_tasks` for a component with mean-field nodes (only its
belief-propagation clusters must be trees), lifts replica 1's, and marks the schedule `hybrid`, so
`propagate!` converges on marginals. `test/plates.jl` checks it against the hand-lifted schedule of
the same plate.

## Expectation-propagation groups (2026-11-03)

`_group_plan` now plans a group of tied `ExpectationPropagationRule()` nodes as well, when the
edges of the targets' variables form block rows in one order (`_variable_rows`, shared with
`_cavity_plan`); the plan's `ep` field holds those rows. Such a group runs only on blocks
(`_step_blocks!`: the message block, the cavity block, `ep_block`) and otherwise edge by edge
through `step!`, never through the vector-batched `factor_messages`, which would send the
belief-propagation messages unprojected. Details and timings in `blocks.md`.
