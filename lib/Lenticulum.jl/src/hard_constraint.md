#implementation

> `HardConstraintFactor(dims, m; noise = 0)` (`hard_constraint.jl`): the linear relation
> $\sum_i A_i x_i = c$ imposed exactly, with [[extended_gaussian|extended Gaussian]] messages. What
> `LinearConstraintFactor(…; noise = 1e-8)` approximates. Its free energy is the finite limit of the
> noisy relation's as the noise vanishes.

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

## Free energy (2026-11-11)

Without noise the factor is $f(x) = \delta(Gx - c)$, the limit of the Gaussian relation
$\mathcal N(Gx;\, c,\, \varepsilon I)$ as $\varepsilon \to 0$; $G = [A_i]_i$ stacks the channels' coefficients.
Its Bethe term $F_f = \mathbb E_{b_f}[-\log f] - H(b_f)$ diverges, energy and entropy alike, but the
term of the noisy relation has a finite limit, and that limit is the term used.

First the inputs are reduced. A point $x_j = v_j$ moves into the constant, $c \mapsto c - A_j v_j$, as
`LinearConstraintFactor` does. An extended-Gaussian input with hard directions $H_j$ is written
$x_j = x_{0j} + U_j s_j$ along the rest ($U_j$ an orthonormal basis of $H_j^\perp$), which moves
$A_j x_{0j}$ into the constant and leaves the block $A_j U_j$. This gives $\tilde G s = \tilde c$ on
$k$ coordinates. The factor belief in $s$ is $b_\varepsilon \propto \mathcal N(\tilde G s; \tilde c, \varepsilon I)\,\prod_j \mathcal N^{-1}(s_j; \eta_j, \Lambda_j)$.
If $\tilde G$ has full row rank $m$, then as $\varepsilon \to 0$:

- the residual covariance tends to $\varepsilon I$, so $\mathbb E[\tfrac1{2\varepsilon} r^\top r] \to \tfrac m2$;
- the term $\tfrac12\log\det(2\pi\varepsilon I) = \tfrac m2 \log 2\pi\varepsilon$ grows without bound;
- in coordinates along the solution set $R = \{\tilde G s = \tilde c\}$ (orthonormal, $z$) and across it, the
  entropy splits as $H_R(b_f) + \tfrac m2 \log(2\pi e \varepsilon) - \tfrac12 \log\det(\tilde G\tilde G^\top)$.
  Across $R$ the belief has covariance $\varepsilon\,(\tilde G V)^{-1}(\tilde G V)^{-\top}$, with $V$ an
  orthonormal basis of $\tilde G$'s row space and $|\det \tilde G V| = \det(\tilde G\tilde G^\top)^{1/2}$.

The divergent parts cancel ($\tfrac m2 + \tfrac m2 \log 2\pi\varepsilon - \tfrac m2\log 2\pi e\varepsilon = 0$):

$$
F_f \;=\; \tfrac12 \log\det\bigl(\tilde G \tilde G^\top\bigr) \;-\; H_R(b_f),
\qquad H_R(b_f) = \tfrac{k-m}{2}(1 + \log 2\pi) - \tfrac12 \log\det\bigl(N^\top \Lambda N\bigr),
$$

with $N$ an orthonormal basis of $\operatorname{null}(\tilde G)$ and $\Lambda$ the inputs' block-diagonal
precision. $H_R$ is the entropy relative to Lebesgue measure on $R$, and a point has $H_R = 0$.

For the sum over the graph to be $-\log Z$ on a tree, the other terms must measure entropy the same
way: relative to Lebesgue measure on each belief's own support. A variable that the relation fixes
in some directions has `variable_entropy` = `support_entropy` of its extended Gaussian
(LenticulumCore, `extended_gaussian.md`, "Entropy"). A `GaussianFactor` or `GaussianPrior` that
receives a hard message substitutes its hard directions out, as above
(`lib/Lenticulum.jl/src/gaussian.md`, "Free energy with hard inputs"). A clamped variable keeps
$H(\delta) = 0$, the case of every direction hard.

The term is infinite, and refused, in three cases:

- $\tilde G$ has rank below $m$: redundant equations, or equations on a direction an input already
  fixes exactly ($\delta(0)$, two hard constraints on one direction);
- $N^\top\Lambda N$ is singular: the factor belief is flat along $R$, an improper posterior;
- the noise is singular but nonzero. Splitting it into hard and soft parts is not implemented.

With a positive definite noise the term is `LinearConstraintFactor`'s.

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
- `islearnable` is `false`. `local_free_energy` is the term above (`fit` 0, `complexity`
  $\tfrac12\log\det\tilde G\tilde G^\top$, `negentropy` $-H_R$), from the cavity messages
  (`uses_cavity`). The reduction is `_reduced_relation` in `constraint.jl`, shared with
  `LinearConstraintFactor`.

## Measured

`test/hard_constraint.jl`:

| check | instances | agreement |
|---|---|---|
| with a positive definite noise, the messages of `LinearConstraintFactor`, inputs points, proper and improper Gaussians or uninformative, every target | 60 random factors (2–3 channels of dimension 1–3, 1–3 equations), 149 messages | $10^{-7}$ in $\mu$, $\Sigma$ and the flat projector |
| without noise, prior × message against the Gaussian conditioned on the equality, $\mu - \Sigma A^\top (A\Sigma A^\top)^+(A\mu - c)$ | 60 random factors with fewer equations than unknowns | $10^{-8}$ |
| a graph of priors and one hard factor, `infer!` with a tree schedule and with damped flooding (damping 0.5), against the same conditioned Gaussian | 15 random graphs | $10^{-7}$; every run converges to `tol = 1e-8` |
| the factor contract (`check_factor_conformance`, direct message, the free energy; gaps for the pointwise and game energies) | $x + y = 1$, $x$ clamped | the site and posterior exactly $y = 0.75$; independent of the prior; an input on the target is excluded; free energy 0 (a point, $\tilde G = [1]$) |

`test/hard_free_energy.jl`:

| check | instances | agreement |
|---|---|---|
| the Bethe free energy of a tree (`scalar_free_energy` after `infer!`) against $-\log Z = -\log \mathcal N(c;\, C\mu,\, C\Sigma C^\top)$ of the dense model; marginals against the conditioned Gaussian | 40 random trees: prior → `GaussianFactor` → $u$ → `GaussianFactor` → $w$, a prior on $v$, a hard relation on $(u, v)$ with up to $\dim u + \dim v$ equations; in 25 of them $u$ ends with hard directions, in 19 $v$ (so hard messages reach both `GaussianFactor`s and the prior on $v$) | within the test's $10^{-7}(1 + \lvert\log Z\rvert)$ |
| the term without noise against the noisy relation's term at $\varepsilon = 10^{-30}$, from its formula in 512-bit arithmetic, on the same reduced relation | 80 draws of 2–3 channels (points, proper Gaussians, extended Gaussians with hard directions); the 64 with a free coordinate | largest relative difference $1.3 \cdot 10^{-15}$ (the test asks for $10^{-9}$) |
| refusals: redundant equations on a clamped input, a belief flat along $R$, a singular nonzero noise; with a positive definite noise, `LinearConstraintFactor`'s value | 4 | exact |

## What it does not do

- **Learning**: `islearnable` is `false`. `free_energy_gradient` and `fit_em` with the factor in the
  graph are not tested; the SVD in its term has no ForwardDiff method.
- **Free energy with a singular nonzero noise**: refused (above).
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
