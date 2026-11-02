#design

> **Status: draft (2026-11-02, Claude), implemented as described.** Related: [[Extending Mycelium]] §2, [[L-SPEC-003 Exposed Parameters and Energy Channels]], [[L-SPEC-005 The Hybrid Free Energy]], [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]], [[LP-021 The coupled gradient drivers cut diagonal edges at fixed messages]]. Code: `lib/Mycelium.jl/src/plates.jl`, `graph.jl` (parameter keys), `passing.jl` (`factor_messages`), `free_energy.jl`, `src/constraint.jl`.

> Sources: original to this repository. code: `lib/Mycelium.jl/src/plates.jl`, `lib/Mycelium.jl/src/graph.jl`, `lib/Mycelium.jl/src/passing.jl`, `lib/Mycelium.jl/src/free_energy.jl`, `src/constraint.jl`; tests: `test/plates.jl`; benchmark: `bench/plates.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

One graph for many observations: a template replicated $n$ times with shared parameters, so that
learning through a graph does not mean building a graph per data point. It covers parameter
tying in general, the plate builder, and schedules that compute the replicas together. It does
not add a batched message representation.

## The contract

**§1. Parameter keys.** Every factor node reads the entry `parameter_key(node)` of `ps` and `st`,
its name unless `factor!(...; params = key)` ties it to other nodes. Tied nodes:

- have one entry in `LuxCore.initialparameters(rng, g)`, whose keys are `parameter_keys(g)`;
- must have the same factor type and `freeze` (`build` throws otherwise), and may differ in other
  fields (data, noise);
- share their state entry, which must be empty (`factor_free_energies` throws otherwise);
- must have parameter trees of one shape (not checked at build time);
- receive the **sum** of their contributions to every gradient: `free_energy_gradient` (all
  drivers) returns one entry per key, and `parameter_step` steps one vector per key.

A graph without ties is unchanged: its keys are its factor names.

**§2. Plates.** `plate!(template, b, name, n)` calls `template(p, i)` for $i = 1, \dots, n$. Inside it,
`variable!`, `factor!` and `connect!` on `p` add replica $i$'s nodes, named `replica_name(name, i)`;
a name the replica did not declare refers to a node of the enclosing graph, shared by all
replicas. Plate factors are tied by default (`tie = false` keys them per replica). Every replica
must declare the same nodes and edges in the same order, with the same channels, directions,
couplings, detach flags and shared nodes; otherwise `plate!` throws. The graph is unrolled, and
`plates(g)` returns the record of each plate (node and edge ids per template position and
replica).

**§3. Plate schedules.** `plate_schedule(g, replica_tasks)` lifts a task list written for replica 1:
a task on a replica-1 edge becomes the group of that template task in every replica; a task
outside the plate stays alone; a task on another replica's edge throws. `plate_schedule(g)` uses
each component's tree schedule and needs the replicas to be separate components. A sweep of a
`PlateSchedule` gives the messages of `SequentialSchedule(tasks(sched))`.

**§4. Batched messages.** Within a group, factor → variable messages of tied belief-propagation
replicas with the same available channels and distinct target variables are computed by one
call of `factor_messages(factors, target, polarity, inputs, priors, ps, st)`, which must return
what `factor_message` returns for each replica. The default loops; `LinearConstraintFactor`
shares the factorisation and the rank test when its replicas' noise and input covariances agree.

## Checked by

`test/plates.jl`:

- the builder: names, the record, keys, untied replicas; refusals for unequal replicas, $n = 0$,
  duplicate plates, references into another replica, tied factors of different types, and a key
  that is another node's name;
- factor analysis over 6 points: each replica's marginal equals its own graph's to $10^{-12}$; the
  free energy equals the sum over the separate graphs and the closed-form $-\log p(X)$ to
  $10^{-10}$; the gradient of the tied entry equals the sum of the separate gradients to $10^{-10}$
  with ForwardDiff and Zygote ($10^{-6}$ with reference differences) and the closed-form gradient
  in $W$ and $c$ to $10^{-8}$; `parameter_step` moves the tied entry by the summed gradient;
- two grouped sweeps give the sequential schedule's messages to $10^{-12}$; the batched
  `LinearConstraintFactor` messages equal `factor_message` for inputs of different covariances;
- a shared latent mean over 5 replicas (a star): the posterior mean, the free energy and its
  derivative in the shared offset against closed forms; a lifted replica schedule with a shared
  target equals the sequential one; `plate_schedule(g)` refuses the coupled graph;
- a plate of 4 hybrid graphs (the sum node on `MeanField()`): free energy and gradient equal the
  sum over separate graphs, and the gradient equals central differences of the converged free
  energy to $10^{-6}$.

`bench/plates.jl` measures the speed (`plates.md`): the grouped plate is 2.1–2.5 times faster
than a loop of prebuilt graphs and 430–740 times slower than the hand-written E-step.

## Open

- **Speed.** [[Extending Mycelium]] §2 asked for a small factor of hand-written code; the gap is
  Mycelium's per-message overhead (polarity resolution, dynamic dispatch), not the factors.
- No nested plates, no batched message type, no stateful tied factors.
- `adjoint = :implicit` disagrees with the default driver on graphs with diagonal couplings,
  plates or not ([[LP-021 The coupled gradient drivers cut diagonal edges at fixed messages]]).
