#design

> **Status: draft (2026-11-02, Claude), implemented as described.** Related: [[Extending Mycelium]] §1, [[L-ADR-0009 Factor nodes carry a message rule]], [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]], [[Bethe Free Energy]]. Code: `lib/Mycelium.jl/src/free_energy.jl` (`counting_number`, `bp_degree`, `marginals_into`, `factor_free_energies`, `free_energy_gradient`), `passing.jl` (`mean_field_energy`), `lib/Lenticulum.jl/src/constraint.jl` and `lib/Lenticulum.jl/src/switching.jl` (the methods).

> Sources: Riegler, Kirkelund, Manchon, Badiu & Fleury, *Merging Belief Propagation and the Mean Field Approximation: A Free Energy Approach*, IEEE Trans. Inf. Theory 59(1), 2013 (arXiv:1112.0467v3): §III, the regions and counting numbers (16), the free energy (17), Theorem 2 (stationary points are the fixed points of the message passing (21)–(22)). code: `lib/Mycelium.jl/src/free_energy.jl`, `lib/Mycelium.jl/src/passing.jl`, `lib/Lenticulum.jl/src/constraint.jl`, `lib/Lenticulum.jl/src/switching.jl`; tests: `lib/Lenticulum.jl/test/mean_field_free_energy.jl`, `lib/Mycelium.jl/test/message_rules.jl`
>
> Bibliography: [[Bibliography#^riegler2013merging|Riegler et al. 2013]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

The free energy of a graph whose factor nodes use belief propagation or mean field, and its
parameter gradient. It replaces the refusal of [[L-ADR-0009 Factor nodes carry a message rule]] for
`MeanField()` nodes. Other rules (expectation propagation, reduction; [[Extending Mycelium]] §4)
are not covered and are still refused.

## The contract

**§1. The free energy.** With $A_{BP}$ and $A_{MF}$ the nodes on each rule, and $N_{BP}(i)$ the
belief-propagation neighbours of variable $i$, the graph free energy is eq. (17) of Riegler et al.:

$$
F_{BP,MF} = \sum_{a \in A_{BP}} \sum_{x_a} b_a \ln\frac{b_a}{f_a}
 - \sum_{a \in A_{MF}} \mathbb E_{\prod_{i \in N(a)} b_i}[\ln f_a]
 - \sum_i \bigl(|N_{BP}(i)| - 1\bigr) \sum_{x_i} b_i \ln b_i .
$$

- A belief-propagation node contributes `local_free_energy` at its incoming (cavity) messages, as
  before.
- A mean-field node contributes `mean_field_energy(factor, marginals, ps, st)`, the average energy
  under the product of its variables' **marginals** (`marginals_into`), with no entropy: its
  `negentropy` component is zero.
- `counting_number(g, v) = 1 - bp_degree(g, v)`. With every node on belief propagation this is
  the Bethe counting number, and nothing changes.

**§2. The hook.** `Mycelium.mean_field_energy` has no generic method; a factor with a
`mean_field_message` should add one, graded like its `local_free_energy`. Lenticulum provides it
for `LinearConstraintFactor` and `SwitchingConstraintFactor`: the factor free energy at the product
of the marginals as a frozen belief (means stacked, covariances block diagonal, entropy dropped). A
Gaussian-mixture marginal enters by its moments, which is exact because both energies are quadratic
in each continuous channel. A marginal that is not a point or a proper distribution is refused.

**§3. The gradient.** `free_energy_gradient` (and so `parameter_step`) differentiates each
belief-propagation node's `belief_free_energy` at its frozen belief (LP-020) and each mean-field
node's `mean_field_energy` at the fixed marginals. By Theorem 2 a fixed point of the hybrid
message passing is a stationary point of §1, so this is the derivative of the converged free
energy (the envelope theorem).

**§4. What still refuses.** Non-diagonal couplings and `adjoint = :implicit` (they differentiate
through inference, written for belief propagation), and the sample buffer's residual moments
(read from cavity messages), throw for a graph with a mean-field node. Any rule other than the two
throws everywhere.

## Checked by

`lib/Lenticulum.jl/test/mean_field_free_energy.jl`:

- counting numbers count belief-propagation neighbours only;
- naive mean field over $(u, v)$ with $y = u + v + c + \varepsilon$: the converged free energy equals
  the closed form at the hand-iterated fixed point to $10^{-10}$; it exceeds $-\log p(y)$ (by 0.31
  here), while the same graph on belief propagation is exact; each coordinate update decreases it;
  the mean-field node's negentropy is zero;
- the gradient at the fixed point matches central differences of the converged free energy to
  $10^{-6}$, with reference differences, ForwardDiff and Zygote;
- both average energies against Monte Carlo ($2 \cdot 10^5$ samples, within 5 standard errors),
  including a categorical outcome for the switching factor;
- a mean-field sum of two overlapping switches (each switch on belief propagation): the free
  energy is above the exact $-\log p(x)$ by 0.0233, against a KL of 0.0215 from the converged
  factorised posterior over the outcomes to the exact one (the rest is the switches' width
  $\eta = 10^{-3}$); the gradient matches central differences to $10^{-5}$ with all three backends;
  `adjoint = :implicit` refuses.

The all-belief-propagation case is checked by every existing free-energy test, unchanged.

## Open

- A mean-field node whose neighbours have **improper** marginals: refused, not integrated.
- Mixture marginals at a variable with two or more belief-propagation neighbours still have no
  entropy in closed form ([[Extending Mycelium]], the cluster nodes of §5).
- Monotone descent along a schedule is checked for naive mean field only; for the hybrid it holds
  for coordinate updates of the mean-field marginals, not for every belief-propagation message.
