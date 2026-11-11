#implementation

> `HardConstraintFactor(dims, m; noise = 0)` (`hard_constraint.jl`): the linear relation
> $\sum_i A_i x_i = c$ imposed exactly, with [[extended_gaussian|extended Gaussian]] messages. What
> `LinearConstraintFactor(…; noise = 1e-8)` approximates. Messages only: no free energy.

> Sources: Stein & Samuelson, *A Category for unifying Gaussian Probability and Nondeterminism*, CALCO 2023 (Gaussian relations: a linear relation composed with Gaussian noise). code: `hard_constraint.jl`; the belief algebra in `lib/LenticulumCore.jl/src/extended_gaussian.jl`. tests: `test/hard_constraint.jl`.
>
> Bibliography: [[Bibliography#^stein2023gaussian|Stein & Samuelson 2023]]
>
> Theory (CT-ML wiki): [Gaussian Relations](https://mathstruct.org/CategoryTheory-ML-Wiki/Gaussian-Relations)

## Theory

The factor is the relation $R = \{x : \sum_i A_i x_i = c\}$, optionally blurred by a noise
$\varepsilon \sim \mathcal N(0, Q)$ on its residual. Its message to channel $t$, given the other
channels' messages $\mu_i$, is the likelihood of $x_t$:

$$
\mu_{f \to t}(x_t) \;=\; \int \mathbb 1\Bigl[A_t x_t = c - \textstyle\sum_{i \ne t} A_i x_i + \varepsilon\Bigr] \prod_{i \ne t} \mu_i(x_i)\, \mathcal N(\varepsilon; 0, Q)\; dx\, d\varepsilon .
$$

Written with extended Gaussians it is two exact operations. The joint of the inputs (independent,
so block-diagonal) is pushed forward to the residual $r = c - \sum_{i \ne t} A_i x_i + \varepsilon$, an
extended Gaussian on $\mathbb R^m$. The message is then the **preimage** of $r$ under $A_t$: the
likelihood $x_t \mapsto p_r(A_t x_t)$,

$$
\mu_{f \to t} \;=\; \operatorname{preimage}\Bigl(\operatorname{pushforward}\bigl(\textstyle\bigotimes_{i \ne t}\mu_i,\; -[A_i]_{i\ne t},\; c,\; Q\bigr),\; A_t\Bigr).
$$

Both are exact for flat, hard and Gaussian directions (`extended_gaussian.md`), so nothing about
the inputs needs special cases. An uninformative input makes $r$ flat along its $A_i$'s range,
the $\tau \to \infty$ limit of `constraint.md` §4.1. A clamped input makes $r$ hard there.
$\operatorname{null}(A_t)$ is flat in the message, as is anything the residual cannot see. With
$Q \succ 0$ the message is the one `LinearConstraintFactor` computes; with $Q = 0$ and point inputs
it is the hard relation itself.

**Why no free energy.** The Bethe free energy of a factor is $\mathbb E_{b_f}[-\log f] - H(b_f)$.
For a relation without noise, $f$ is an indicator: $-\log f$ is $0$ on $R$ and $\infty$ off it, and
$b_f$ lives on $R$, a set of measure zero in the joint space, so $H(b_f)$ is $-\infty$ with respect to
Lebesgue measure there. The two diverge together and their difference has a limit only with a
convention for the measure on $R$; [[Bethe Free Energy]] fixes one for clamped variables
($H(\delta) = 0$) and none for relations. Choosing it is the open row of the [[Feature Backlog]]
("the entropy and free-energy contribution of an extended Gaussian").

## Implementation

- The parameters and polarities are `LinearConstraintFactor`'s: `(A, c)`, any one channel
  `Unobserved()`, the others `Observed()` or `Latent()`. `noise` defaults to $0$ and may be any
  positive semidefinite `m × m` matrix (soft along its range, hard elsewhere).
- Inputs become extended Gaussians by `LenticulumCore.extended`: a `GaussianBelief` (also improper), a
  `DiracBelief`, an `ExtendedGaussian`, or nothing (`TrivialBelief()`, no message: flat). A
  mixture is refused rather than collapsed silently.
- The message is always an `ExtendedGaussian`, also when it could be a `GaussianBelief`, so its type
  does not depend on the data. Mycelium's `combine`, `belief_distance` and `damp` take it
  (`lib/Mycelium.jl/src/gaussian.md`, "Extended Gaussians"), so the factor runs under any schedule.
- `islearnable` is `false`, and `local_free_energy` throws an `ArgumentError` (see above).

## Measured

`test/hard_constraint.jl`:

| check | instances | agreement |
|---|---|---|
| with a positive definite noise, the messages of `LinearConstraintFactor`, inputs points, proper and improper Gaussians or uninformative, every target | 60 random factors (2–3 channels of dimension 1–3, 1–3 equations), 149 messages | $10^{-7}$ in $\mu$, $\Sigma$ and the flat projector |
| without noise, prior × message against the Gaussian conditioned on the equality, $\mu - \Sigma A^\top (A\Sigma A^\top)^+(A\mu - c)$ | 60 random factors with fewer equations than unknowns | $10^{-8}$ |
| a graph of priors and one hard factor, `infer!` with a tree schedule and with damped flooding (damping 0.5), against the same conditioned Gaussian | 15 random graphs | $10^{-7}$; every run converges to `tol = 1e-8` |
| the factor contract (`check_factor_conformance`, direct message, with the documented gaps for energies) | $x + y = 1$, $x$ clamped | the site and posterior exactly $y = 0.75$; independent of the prior; an input on the target is excluded |

## What it does not do

- **Free energy and learning** (above): `fit_em`, `free_energy_gradient` and `scalar_free_energy`
  refuse a graph that contains it.
- **Batched plates**: the factor has no `factor_messages` block method; a plate of hard
  constraints runs per replica.
- **Mean-field messages** (`MeanField()` nodes): not implemented.
- **Mixture inputs**: refused; `LinearConstraintFactor` collapses them to their moments.

## Difficulties

The first comparison with `LinearConstraintFactor` failed in 23 of 149 messages, all with every
other input uninformative. The pushforward is then flat in every direction, and projecting $\Sigma$
with $I - DD^\top$ left a residue of $10^{-30}$, which the relative tolerance took for a real variance:
the message came out hard where it should be flat. The extended Gaussian now projects through an
orthonormal basis of the complement of $D$, which is empty in that case, so $\Sigma$ is exactly zero.

The damped flooding run of one instance (an ill-conditioned $A$) stops at a residual of
$3.8 \cdot 10^{-10}$, a rounding floor of the absolute convergence bound (LP-024's open row), so the test
asks for $10^{-8}$; its marginals agree to $10^{-7}$.
