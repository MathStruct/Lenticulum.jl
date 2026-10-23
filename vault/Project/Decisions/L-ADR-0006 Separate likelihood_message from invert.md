#design

> **Status: proposed (2026-10-23, `review-argon`).** Related: [[Messages are Inversions]] §5, [[bayesian_lens]], [[passing]], [[gaussian]] §4, [[L-SPEC-001 The Factor Interface]] §2, §5.

## Context

In AutoBayes (Definition 10), the Bayesian inversion $c^\dagger_\pi(y)$ of a lens $(c, c^\dagger)$ returns the **posterior** belief over the unobserved channel given the prior $\pi$ and observation $y$: $c^\dagger_\pi(y) \propto \pi \cdot \mu_{c\to\text{target}}(y)$. In belief propagation on a factor graph, however, the factor-to-variable message $\mu_{c\to i}$ must be the **likelihood** contribution (with the cavity prior $\pi$ divided out, using $\pi$ at most as an expansion point for linearisation or Laplace approximation); otherwise a variable of degree $d$ counts its prior $d$ times.

Previously, `Mycelium.factor_message`'s generic fallback called `LenticulumCore.invert(lens, prior, inputs, ps, st)`, while individual factors (`ImplicitLayers`, `Adversarial`, `VariationalDiffusion`) returned likelihoods from `invert` and `Lenticulum`'s `GaussianFactor` returned posteriors from `invert` and bypassed `invert` in `factor_message`.

## Options

1. Have `invert` return the likelihood message everywhere, departing from AutoBayes Definition 10.
2. Have `invert` return the posterior everywhere and divide out the prior in `Mycelium.factor_message` via a belief division operation.
3. Add `LenticulumCore.likelihood_message(lens, π, inputs, ps, st) -> (belief, st)` for the cavity-free BP likelihood message (defaulting to `invert(lens, π, inputs, ps, st)` for lenses whose `invert` already returns a likelihood or point estimate), define `invert` on Gaussian/Laplace factors as `combine(π, likelihood_message(...))`, and have `Mycelium.factor_message` call `likelihood_message`.

## Decision

Option 3. Belief division is ill-defined when the prior or message is improper or non-Gaussian (`SampleBelief`, `DiracBelief`), whereas forward multiplication (`Mycelium.combine(π, msg)`) is always well-defined. Separating `likelihood_message` (BP factor-to-variable likelihood) from `invert` (AutoBayes posterior) allows both contracts to hold without division or double-counting.

## Consequences

- `LenticulumCore.invert(lens, π, inputs, ps, st)` returns the posterior $c^\dagger_\pi(y) = \pi \odot \mu_{c\to t}$ on `GaussianFactor`, `LinearConstraintFactor`, `NonlinearFactor`, and `LikelihoodFactor`.
- `LenticulumCore.likelihood_message(lens, π, inputs, ps, st)` returns the likelihood message $\mu_{c\to t}$, using $\pi$ only where needed as an expansion point (`NonlinearFactor`, `LikelihoodFactor`, `DiffusionFactor`).
- `Mycelium.factor_message`'s generic fallback calls `LenticulumCore.likelihood_message(lens, prior, inputs, ps, st)`.
