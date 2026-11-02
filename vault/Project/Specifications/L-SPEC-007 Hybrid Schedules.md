#design

> **Status: draft (2026-11-03, Claude), implemented as described.** Related: [[Extending Mycelium]] §3, [[L-SPEC-005 The Hybrid Free Energy]], [[L-SPEC-006 Plates]], [[L-ADR-0009 Factor nodes carry a message rule]]. Code: `lib/Mycelium.jl/src/schedules.jl` (`hybrid_schedule`, `HybridSchedule`), `passing.jl` (`propagate!`'s `criterion`), `messages.jl` (`_moment_distance`), `plates.jl` (`plate_schedule(g)` for hybrid templates).

> Sources: Riegler, Kirkelund, Manchon, Badiu & Fleury, *Merging Belief Propagation and the Mean Field Approximation: A Free Energy Approach*, IEEE Trans. Inf. Theory 59(1), 2013 (arXiv:1112.0467v3), §III, the message passing (21)–(22) and Theorem 2. code: `lib/Mycelium.jl/src/schedules.jl`, `lib/Mycelium.jl/src/passing.jl`, `lib/Mycelium.jl/src/messages.jl`, `lib/Mycelium.jl/src/plates.jl`; tests: `test/switching_constraint.jl`, `test/mean_field_free_energy.jl`, `test/plates.jl`
>
> Bibliography: [[Bibliography#^riegler2013merging|Riegler et al. 2013]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

A schedule for graphs with `MeanField()` nodes, built from the rules instead of by hand, and a
convergence test that holds for mixtures of very narrow components. Graphs whose
belief-propagation part has loops are out of scope.

## The contract

**§1. The schedule.** `hybrid_schedule(g)` returns a `HybridSchedule`. The **clusters** are the
connected components of the graph with its mean-field nodes removed, and each must be a tree
(otherwise it throws). A sweep is

1. every cluster swept inward and outward (its tree schedule, edges to mean-field nodes excluded),
   with the mean-field messages already at its variables held fixed;
2. each mean-field node in graph order: its message to each variable it can emit to, each followed
   by that variable's cluster swept again.

On a fresh store the first pass of 1 starts each cluster from its own factors (priors, data), the
starting point of E7. Without mean-field nodes the schedule is the tree schedule of each component.

**§2. Why this order.** Within a cluster, two sweeps give the exact beliefs given the mean-field
messages. A mean-field message followed by its cluster's sweep therefore replaces that cluster's
beliefs by their exact update, so when a mean-field node's other variables lie in other clusters
each step is block coordinate descent on the merged free energy of L-SPEC-005 and the free energy
does not increase. When two variables of one mean-field node lie in one cluster this is not
claimed.

**§3. Convergence on marginals.** `propagate!(...; criterion)` accepts `:messages` (the largest
message residual, the default for other schedules) or `:marginals` (the default for a
`HybridSchedule` and for a plate schedule lifted from one): the largest change of a variable's
marginal between sweeps, compared by moments (means, covariances, and the weights of a mixture or a
categorical). Narrow mixture components have a rounding floor in natural parameters of the order of
their precision times the machine epsilon; their moments do not.

**§4. Plates.** `plate_schedule(g)` uses the hybrid schedule for a component with mean-field nodes,
and lifts replica 1's.

## Checked by

- `test/switching_constraint.jl`, the two-block switching hybrid at 8 points: converged at
  `tol = 1e-10` on marginals (the hand-written schedule used `1e-6` on messages); outcome
  probabilities within $10^{-6}$ of the structured mean-field oracle and $10^{-9}$ of the hand-written
  schedule; the merged free energy non-increasing over 30 sweeps to $10^{-10}$ with components of width
  $10^{-3}$; a cluster with a loop refused.
- `test/mean_field_free_energy.jl`: naive mean field reaches the closed-form free energy to $10^{-10}$.
- `test/plates.jl`: the default schedule of a plate of hybrid graphs is hybrid and reaches the
  hand-lifted schedule's free energy to $10^{-10}$.

## Open

- With components of width $10^{-8}$ the free energy rises by up to about $10^{-7}$ after convergence,
  which is rounding in the entropies of narrow mixtures (it disappears at width $10^{-3}$); measured
  while writing this, not asserted by a test.
- Clusters with loops, and mean-field nodes with several variables in one cluster (§2).
- The order of mean-field nodes is the graph's; no residual-driven ordering.
