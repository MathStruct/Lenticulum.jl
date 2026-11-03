#design

> **Status: draft (2026-11-03, Claude), implemented as described.** Related: [[Extending Mycelium]] §4, [[L-ADR-0009 Factor nodes carry a message rule]], [[L-SPEC-005 The Hybrid Free Energy]], [[L-SPEC-006 Plates]] §7–§8. Code: `lib/Mycelium.jl/src/graph.jl` (the rules), `passing.jl` (`_rule_message`, `ep_message`, `mixture_message`, `reduce_message`, `_tree_exact`), `gaussian.jl` (`ep_project`, `ep_divide`, products of improper components), `free_energy.jl`; Lenticulum `src/switching.jl` (`mixture_message`, the switching blocks), `src/beliefs.jl` (`reduce_message`), `src/constraint.jl`.

> Sources: Minka, *Expectation Propagation for approximate Bayesian inference*, UAI 2001 (arXiv:1301.2294), the algorithm and the clutter problem; Runnalls, *Kullback-Leibler Approach to Gaussian Mixture Reduction*, IEEE Trans. Aerosp. Electron. Syst. 43(3), 2007, the merge cost. code: `lib/Mycelium.jl/src/passing.jl`, `lib/Mycelium.jl/src/gaussian.jl`, `lib/Mycelium.jl/src/free_energy.jl`, `src/switching.jl`, `src/beliefs.jl`; tests: `test/mixture_rules.jl`, `test/plates.jl`; benchmark: `bench/mixture_plates.jl`
>
> Bibliography: [[Bibliography#^minka2001ep|Minka 2001]] · [[Bibliography#^runnalls2007kl|Runnalls 2007]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

## Scope

Two message rules beside `BeliefPropagation()` and `MeanField()`, chosen per node with
`factor!(b, name, f; rule)`, for factors whose belief-propagation message leaves the target's family
(a mixture of Gaussians). Expectation propagation keeps every message in the family; mixture
reduction keeps mixtures within a budget. Cluster nodes (§5 of the design note) are out of scope.

## The contract

**§1. `ExpectationPropagationRule()`.** The node computes the belief-propagation message $m_t$ from
its cavities, as a `BeliefPropagation()` node does. If $m_t$ is not a `MixtureBelief` it is sent.
Otherwise, with $\pi_t$ the target's cavity, it sends `ep_message(m_t, π_t)` $= \operatorname{proj}[\pi_t m_t] / \pi_t$:
`ep_project` matches moments (a Gaussian mixture: total mean and covariance), and `ep_divide`
subtracts canonical parameters, declaring the site improper unless its precision is positive
definite. An uninformative cavity leaves $m_t$ as the tilted belief; a point cavity sends $m_t$
unchanged. The update is **skipped**, the edge keeping its message, when the cavity is a Gaussian
that is not proper, when the tilted belief has a component that is not proper, or when an input
to the factor has a negative direction. The first skip of a run warns. The name avoids
Lenticulum's `ExpectationPropagation`, the inversion method of `LikelihoodFactor`.

**§2. `MixtureReduction(M)`.** The node computes `mixture_message(factor, target, polarity,
inputs, prior, ps, st)` when an input is a `MixtureBelief` (else `factor_message`) and sends
`reduce_message(message, M)`. `mixture_message` is the exact message with mixture inputs expanded
into their components; a factor without a method refuses. `reduce_message` returns a belief that is
not a mixture, or has at most `M` components, unchanged; Lenticulum reduces a Gaussian mixture with
Runnalls' merge and returns the component when one is left; a mixture with an improper component
over the budget is refused. `MixtureReduction(0)` is refused.

**§3. Implementations.** Lenticulum implements `mixture_message` for `SwitchingConstraintFactor` (to
a continuous channel and to the outcome) and `LinearConstraintFactor`, with mixture inputs of
proper Gaussian components and the other inputs as `factor_message` takes them, flat directions
included (`switching.md` §9).

**§4. Schedules.** One undamped `TreeSchedule` sweep is reported exact only when every node is a
belief-propagation or mixture-reduction node; a graph with an expectation-propagation or
mean-field node is iterated. Plate sweeps group belief-propagation and (on message blocks only)
expectation-propagation nodes, [[L-SPEC-006 Plates]] §8; mixture-reduction nodes go edge by edge.

**§5. Free energy.** An expectation-propagation node is counted as a belief-propagation node: its
`local_free_energy` at its incoming messages, and its edges in the counting numbers. The default
free energy and gradient drivers accept it; the drivers that differentiate through inference and
the residual-moment recorder refuse it. Every energy path refuses a mixture-reduction node.

**§6. Products of improper components.** `combine_with_overlap` of two improper Gaussians (two
likelihood components) is their exact product; normalised with its mass in $\log Z$ when proper,
with $\log Z = 0$ otherwise.

## Checked by

`test/mixture_rules.jl`:

- the clutter problem ($D = 1$, $w = 0.5$, variances 100 and 10, data from the model): belief
  propagation's $2^N$-component marginal against enumeration to $10^{-12}$; the rule at the fixed
  point of a hand-written Minka loop to $10^{-10}$, on a tree and on a sequential schedule; at $N = 12$
  within $6 \cdot 10^{-4}$ of the exact mean and 2% of the variance; at $N = 4$ the same oscillation as the hand loop;
- the rule's pieces: a message in its family passes, an improper cavity skips, a point cavity
  passes, the site times the cavity has the tilted moments, an indefinite site is declared
  improper, a Gaussian graph gives belief propagation's marginals;
- a location-mixture likelihood on EP nodes: the gradient at fixed messages against central
  differences of the re-converged free energy, $10^{-7}$ relative; the implicit adjoint refused;
- `mixture_message` against pointwise oracles to $10^{-12}$ (both factors, both kinds of target),
  with a flat input against a prior of variance $10^8$ to $10^{-6}$, equal to `factor_message` without
  mixture inputs, and its refusals;
- a switching linear chain ($T = 5$, $K = 2$) against enumeration: errors 0.16, 0.011,
  $9.6 \cdot 10^{-4}$, $1.0 \cdot 10^{-4}$ and $1.6 \cdot 10^{-15}$ for budgets 1, 2, 4, 8, 16; belief propagation equal to budget 2;
  the free energy refused; `reduce_message`'s cases;
- the product of two improper components against its closed form.

`test/plates.jl` checks the switching plates of L-SPEC-006 §7.

## Open

- Mixture-reduction nodes in plates run edge by edge (their component counts differ between
  replicas); EP plates feeding a linear factor go replica by replica there (L-SPEC-006 §8.4).
- No free energy for reduction nodes, and none for belief-propagation graphs with Gaussian-mixture
  marginals (no closed-form entropy).
- Expanded mixture inputs must have proper components.
- The projection is onto Gaussians only (and categoricals pass as they are); other families
  would add an `ep_project` method.
