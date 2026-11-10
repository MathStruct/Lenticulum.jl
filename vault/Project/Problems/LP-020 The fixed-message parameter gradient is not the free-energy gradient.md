#open-problem

> **Status: resolved (2026-11-01), by option 1 below.** Found while building Mycelium's sampler; measured and now checked in `lib/Lenticulum.jl/test/lp020.jl`. Related: [[free_energy]], [[gradient_couplings]], [[sampling]], [[Extending Mycelium]], [[LP-002 The Jacobian side is not implemented]].

> Sources: original repository analysis; code: `lib/Mycelium.jl/src/free_energy.jl` (`free_energy_gradient`, `parameter_step`), `lib/Mycelium.jl/src/factors.jl` (`factor_belief`, `belief_free_energy`), `lib/Lenticulum.jl/src/gaussian.jl` (`FrozenGaussianBelief`), `parked/AlgebraicFactors.jl/src/toric.jl` (`ToricBelief`), `parked/AlgebraicFactors.jl/src/polynomial.jl` (`PolynomialBelief`), `lib/Mycelium.jl/src/sampling.jl` (`sample_gradient`); tests: `lib/Lenticulum.jl/test/lp020.jl`, `parked/AlgebraicFactors.jl/test/lp020.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

**Problem.** With every edge on the default diagonal coupling, `free_energy_gradient` (and so
`parameter_step`) differentiates each factor's local free energy with its **incoming messages**
held fixed. The local free energy recomputes the factor's belief $b_c \propto f_\theta \prod_i \mu_i$ from those
messages and the parameters, so the belief moves with $\theta$. Writing
$F_c(\theta; \mu) = -\log Z_c(\theta; \mu) + \mathbb E_{b_c}[\log \prod_i \mu_i]$:

- the gradient of the graph's free energy at a belief-propagation fixed point is
  $\sum_c \partial_\theta(-\log Z_c) = \sum_c \mathbb E_{b_c}[-\partial_\theta \log f_c]$ (the envelope theorem: the Bethe free
  energy is stationary in the beliefs, so they are held fixed, not the messages);
- the fixed-message derivative adds $\sum_c \partial_\theta\, \mathbb E_{b_c}[\log \prod_i \mu_i]$, which is not zero in general.

Measured in `lib/Lenticulum.jl/test/lp020.jl`, on the smallest example: $u \sim \mathcal N(0, 1)$, $z = a u + c + \varepsilon$,
$\varepsilon \sim \mathcal N(0, 0.3)$, $z = 1.7$ observed. On this tree the Bethe free energy equals $-\log p(z)$ exactly. At
$a = 0.8$, $c = 0.4$:

| | $\partial/\partial A_u$ ($A_u = -a$) | $\partial/\partial c$ |
|---|---|---|
| exact gradient of $-\log p(z)$ | 0.679 | −1.383 |
| through inference (`sample_gradient`) | 0.679 | −1.383 |
| fixed messages (`free_energy_gradient`) | −0.146 | −0.441 |

The discrepancy changes the **fixed point**, not only the step. At the maximum-likelihood
$a = 1.179$ the fixed-message derivative in $A_u$ is −0.574, not 0. Alternating inference with fixed-message
steps in $a$ stops at $a = 0.754$. (In $c$ it happens to reach the maximum-likelihood value.) So
`parameter_step` with diagonal couplings does not, in general, train to maximum likelihood, even on
a tree with exact inference.

**Options.**

1. **Differentiate at fixed beliefs.** A factor hook for the expected energy of a *given* belief,
   $\mathbb E_b[-\log f_\theta]$, differentiated with $b$ frozen. That is the envelope gradient, exact at a fixed
   point, and as cheap as now. It needs one method per factor family. Closed form for the Gaussian
   and switching factors (mean and covariance terms); an expectation of the pointwise energy in
   general.
2. **Differentiate $-\log Z_c(\theta; \mu)$ locally.** $F_c$ minus the cross-entropy term $\mathbb E_{b_c}[\log \mu_i]$. That is
   generic over factors, but needs that expectation for every belief type (closed form for Gaussian
   and categorical messages).
3. **Differentiate through inference.** Already available, as `ExactCoupling` with a schedule replay
   (`gradient_couplings.md`) or as `sample_gradient`'s re-run of inference per sample. Exact on trees,
   but costs an inference per gradient evaluation (per coordinate with reference differences).

**Resolution (2026-11-01).** Option 1, decided by Daniel ("fix LP-20"). Two hooks in Mycelium:
`factor_belief(factor, msgs, ps, st)` returns the factor's belief, evaluated once outside AD, and
`belief_free_energy(factor, b, ps, st)` is $\mathbb E_b[-\log f_\theta] - H(b)$ at that fixed belief. The graph
gradient differentiates the latter. Lenticulum implements them for `GaussianFactor`,
`GaussianPrior`, `LinearConstraintFactor`, `SwitchingConstraintFactor` and the collapsed Gaussian
composite, through one frozen joint Gaussian of each factor's non-point channels
(`FrozenGaussianBelief`). Theory and limits: [[free_energy]], section "Fixed beliefs, not fixed
messages".

Checked in `lib/Lenticulum.jl/test/lp020.jl`:
- on the example above the gradient is now $(0.679, -1.383)$, and alternating inference with it
  converges to the maximum-likelihood $a = 1.179$;
- on four trees (this one; a Gaussian factor and prior with learnable noise; the composite; a
  switching constraint, including its log-weights) the graph gradient matches central differences
  of the converged Bethe free energy to $10^{-6}$, with reference differences, ForwardDiff and Zygote.

**What remains.**
- Factors without `factor_belief` (the neural and diffusion factors) keep the fixed-message
  derivative. That is exact with point messages, and they warn once when their messages are not
  points. `ToricFactor` and the degree-one `PolynomialFactor` gained the hooks on 2026-11-02, with
  the same check against central differences (`parked/AlgebraicFactors.jl/test/lp020.jl`).
  `PriorFactor` needs none, since its energy does not read the parameters.
- `sample_gradient` now uses the fixed-belief gradient per sample by default (one inference per
  sample); differentiating through inference stays as `through_inference = true`, and the two agree
  to $10^{-6}$ in `lib/Lenticulum.jl/test/switching_constraint.jl` ([[sampling]]).
