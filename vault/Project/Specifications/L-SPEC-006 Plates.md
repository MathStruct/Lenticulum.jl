#design

> **Status: draft (2026-11-02, Claude), implemented as described; §5–§6 added 2026-11-03 with [[L-ADR-0010 Plates get batched message blocks]], §7 the same day with [[L-SPEC-008 Expectation Propagation and Mixture Reduction]], and §8, planned first, implemented later that day.** Related: [[Extending Mycelium]] §2, [[L-SPEC-003 Exposed Parameters and Energy Channels]], [[L-SPEC-005 The Hybrid Free Energy]], [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]], [[LP-021 The coupled gradient drivers cut diagonal edges at fixed messages]]. Code: `lib/Mycelium.jl/src/plates.jl`, `blocks.jl`, `graph.jl` (parameter keys), `passing.jl` (`factor_messages`), `free_energy.jl`, `src/constraint.jl`, `src/gaussian.jl`, `src/switching.jl`.

> Sources: original to this repository. code: `lib/Mycelium.jl/src/plates.jl`, `lib/Mycelium.jl/src/graph.jl`, `lib/Mycelium.jl/src/passing.jl`, `lib/Mycelium.jl/src/free_energy.jl`, `src/constraint.jl`; tests: `test/plates.jl`; benchmarks: `bench/plates.jl`, `bench/mixture_plates.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

One graph for many observations: a template replicated $n$ times with shared parameters, so that
learning through a graph does not mean building a graph per data point. It covers parameter
tying in general, the plate builder, schedules that compute the replicas together, and, since
2026-11-03, message blocks: a batched message representation for plate edges, with batched
energies.

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
call (the structure of a lifted group, its edges by channel and its checks, is found once when the
schedule is built; the polarity is resolved and checked on the first member, which assumes that
tied factors of one type in one template position support the same polarities) of `factor_messages(factors, target, polarity, inputs, priors, ps, st)`, which must return
what `factor_message` returns for each replica. The default loops; `LinearConstraintFactor`
shares the factorisation and the rank test when its replicas' noise and input covariances agree.

**§5. Message blocks.** `MessageStore(g)` gives each template edge of each plate a block row (its
edges, replica $i$ in column $i$), in each direction. A row holds either one `BeliefBlock` or one
message per edge. Indexing an edge of a row held as a block returns that edge's column as a
`Message`; assigning to one first turns the whole row back into per-edge messages. A store built
from vectors has no rows. A `PlateSchedule` group whose edges are block rows writes its result as
one block: a factor → variable group through `factor_messages(factors, target, polarity,
inputs::NamedTuple, priors, ps, st)`, whose inputs are blocks and whose result is a block (or a
vector of beliefs), and which must give, column for column, what `factor_message` gives; a
variable → factor group as the `combine` of the other rows at each member's variable. A group
takes the per-edge path when a variable is pinned, a row is partly filled or partly
uninformative, damping is on, or a target edge is detached. The default `factor_messages` on
blocks goes replica by replica; `LinearConstraintFactor`, `GaussianPrior` and `DataFactor` have
block methods.

**§6. Batched energies.** A plate factor row whose nodes are belief-propagation nodes on one key,
with their edges in block rows (none detached) and their incoming messages held as blocks, is
evaluated by `local_free_energies(factors, msgs, ps, st)` (each replica's `local_free_energy`) in
the free energy, and by `factor_beliefs` (outside AD) and `belief_free_energies` (differentiable)
in `free_energy_gradient`, one AD item per row. `nothing` from `factor_beliefs` sends the row to
the per-node path. The defaults loop over replicas; `LinearConstraintFactor` and `GaussianPrior`
implement them for Gaussian blocks of shared precision. A plate variable row's Bethe terms are
the entropies of its marginal block.

**§7. Categorical blocks and switching plates.** A `CategoricalBlock` holds the normalised
log-probabilities of $n$ categoricals over one support as a $K \times n$ matrix; `belief_block` builds it
from categoricals of one length and labels. `SwitchingConstraintFactor` has a block method to its
outcome (inputs as for `LinearConstraintFactor`'s), which returns a `CategoricalBlock`, and
implements the three energy hooks of §6 through `FrozenSwitchingBeliefs`. Energy rows accept
expectation-propagation nodes as well, when every node of the row has the same rule
([[L-SPEC-008 Expectation Propagation and Mixture Reduction]]).

**§8. Mixture blocks and EP plates** (planned 2026-11-03, implemented the same day; the plan's
fourth step changed, see below). A `MixtureBlock(η, Λ, logw, improper)` holds $n$ mixtures of $K$
Gaussians with one precision for every component: $\eta$ is $d \times K \times n$, the log-weights $K \times n$.
`belief_block` builds it from mixtures whose components share one precision object and flag.

1. `SwitchingConstraintFactor`'s block method to a continuous channel returns one, when its
   inputs are as in §7, the outcome's input is trivial, absent or a categorical block, $K > 1$, and
   every outcome is possible in every replica; otherwise the replicas go one by one.
2. `combine_blocks` of a mixture block and a Gaussian block with one proper shared precision, in
   either order, gives the columns of `combine`, batched; other pairs go column by column.
3. A group of tied `ExpectationPropagationRule()` nodes is planned when its targets' cavity edges
   are block rows (`_GroupPlan.ep`). On blocks the group sends `ep_block(message, cavity)`: for a
   mixture block and a shared-precision Gaussian cavity, the sites as one `GaussianBlock` with a
   precision per replica; otherwise `ep_message` column by column. When a column's update would
   be skipped, a variable is pinned, or an input has a negative direction, the group goes edge by
   edge. An EP group never takes the vector-batched path.
4. `LinearConstraintFactor`'s block method takes inputs with a precision per replica (EP sites,
   or any proper Gaussian blocks, beside points) in one loop with preallocated buffers
   (`_linear_block_each`): $R_i = Q + \sum_n A_n S_{n,i} A_n^\top$ and the message of `_linear_message` without
   flat inputs; inputs with flat directions still go replica by replica. (Planned as a follow-up
   on 2026-11-03 and implemented the same day.)

Measured (`bench/mixture_plates.jl`, `blocks.md`): at $N = 1000$ the mixture messages back to the
points take 0.22 ms instead of 5.0 ms (the plan's conjecture was a few tenths of a millisecond), and
one EP sweep of a latent mixture 0.85 ms instead of 44 ms edge by edge, and one sweep of two layers
(EP sites feeding a linear factor) 1.7 ms against 67 ms edge by edge. The per-replica loop of step 4
brought that from 2.3 ms: the per-replica path it replaces was already nearly as fast, and the
allocations of small matrix products were what cost (`blocks.md`).

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

- each lifted factor → variable group has a plan and dispatches to the batched method of its
  factor type, on edge inputs and on blocks; a plate of 500 replicas has the closed-form free
  energy to $10^{-12}$ (relative);
- after a sweep of the factor-analysis plate all six live rows are blocks (four Gaussian with a
  shared precision, two of points); writing one edge of a row turns it back into per-edge
  messages with the same values;
- the energies on blocks against the same messages per edge: the scalar free energy to
  $10^{-13}$ (relative), the graded terms to $10^{-12}$, the gradient to $10^{-10}$ with ForwardDiff and
  Zygote ($10^{-6}$ with reference differences); the gradient of `:lik` is one batched item.

- a plate of switching factors over 40 observed points, and one with a latent mixture on
  expectation-propagation nodes: messages equal the sequential schedule's to $10^{-10}$; after a
  sweep the outcome messages form a `CategoricalBlock`; the free energy on blocks against the same
  messages per edge to $10^{-12}$ (relative), the graded terms to $10^{-12}$, the gradient to
  $10^{-10}$ with ForwardDiff and Zygote ($10^{-6}$ with reference differences), one batched item;
  the categorical block's product, distance and entropies column for column.

- mixture blocks: a plate sending its mixtures back to observed points and one pooling a Gaussian
  prior with the mixture at a latent variable give the sequential schedule's messages to $10^{-10}$
  with mixture blocks in both directions; the block product equals the per-column `combine` to
  $10^{-12}$ (proper and improper components, both orders); the EP plate runs on blocks (a site
  block with per-replica precisions) and matches the sequential schedule after 30 sweeps;
- EP sites into a linear factor (two layers): the grouped schedule gives the sequential one's
  messages to $10^{-10}$ after 30 sweeps, the linear factor's row is a block with per-replica
  precisions, and its block method equals the per-replica messages to $10^{-12}$.

`bench/mixture_plates.jl` (`blocks.md`, `switching.md` §10): the mixture E-step at $N = 1000$ takes
0.10 ms on blocks, 9.7 ms edge by edge and 0.044 ms by hand; $F$ 1.1 ms against 5.1 ms per edge,
$\nabla F$ 2.9 ms against 29 ms.

`bench/plates.jl` measures the speed (`plates.md`, `blocks.md`): on blocks the grouped plate is
about 9 times the hand-written E-step at $N = 1000$ (0.09 ms against 0.0102 ms; 2.39 ms before), the
free energy 1.1 ms and its gradient 2.1 ms (10.7 and 57 ms with the same messages per edge);
building is linear in $N$.

## Open

- **Speed.** [[Extending Mycelium]] §2 asked for a small factor of hand-written code; on blocks the
  plate is about 9 to 13 times slower than the closed form, the mixture E-step about 2 to 3.5 times.
  Inputs with flat directions and a precision per replica go replica by replica.
- No nested plates, no stateful tied factors.
- [[LP-021 The coupled gradient drivers cut diagonal edges at fixed messages]], found here, is
  resolved for factors whose edges are all cut; factors with mixed couplings are open.
