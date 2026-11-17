#implementation

> `HardConstraintFactor(dims, m; noise = 0)` (`hard_constraint.jl`): the linear relation
> $\sum_i A_i x_i = c$ imposed exactly, with [[extended_gaussian|extended Gaussian]] messages. What
> `LinearConstraintFactor(…; noise = 1e-8)` approximates. Its free energy is the finite limit of the
> noisy relation's as the noise vanishes.

> Sources: Stein & Samuelson, *A Category for unifying Gaussian Probability and Nondeterminism*, CALCO 2023 (Gaussian relations: a linear relation composed with Gaussian noise). code: `hard_constraint.jl`; the belief algebra in `lib/LenticulumCore.jl/src/extended_gaussian.jl`. tests: `test/hard_constraint.jl`, `test/hard_plates.jl` (plates and mean-field nodes, P6).
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

The term is infinite, and refused, in two cases:

- $\tilde G$ has rank below $m$: redundant equations, or equations on a direction an input already
  fixes exactly ($\delta(0)$, two hard constraints on one direction);
- $N^\top\Lambda N$ is singular: the factor belief is flat along $R$, an improper posterior.

With a positive definite noise the term is `LinearConstraintFactor`'s.

## Free energy with a singular noise (P5, 2026-11-17)

A noise $Q \succeq 0$ of rank $0 < r < m$ is written $Q = Q_s D Q_s^\top$, with $Q_s$ ($m \times r$) an orthonormal
basis of its range, $D$ its positive eigenvalues, and $Q_h$ ($m \times (m - r)$) an orthonormal basis of its
null space; an eigenvalue counts as zero below LenticulumCore's `_eig_tol` (the tolerance the
messages use; `isposdef` would accept a rank-deficient $BB^\top$, whose zero eigenvalues come out
as $10^{-16}$). The relation is the limit of the noise $Q_\varepsilon = Q + \varepsilon Q_h Q_h^\top$, and since
$[Q_s\ Q_h]$ is orthogonal its density factors exactly:

$$
\mathcal N(\tilde G s;\, \tilde c,\, Q_\varepsilon) \;=\; \mathcal N(G_s s;\, c_s,\, D)\;\mathcal N(G_h s;\, c_h,\, \varepsilon I),
\qquad G_s = Q_s^\top \tilde G,\ c_s = Q_s^\top \tilde c,\ G_h = Q_h^\top \tilde G,\ c_h = Q_h^\top \tilde c .
$$

The soft factor does not depend on $\varepsilon$, so it joins the inputs: the precision $\Lambda' = \Lambda + G_s^\top D^{-1} G_s$ and
information $\eta' = \eta + G_s^\top D^{-1} c_s$. The exact factor is the noise-free case above with $(G_h, c_h)$
for $(\tilde G, \tilde c)$ and $\Lambda'$ for $\Lambda$, and the soft factor adds its average energy under the
factor belief on $R = \{G_h s = c_h\}$:

$$
F_f \;=\; \tfrac12\log\det(G_h G_h^\top) \;-\; H_R(b_f) \;+\; \tfrac12\,\mathbb E_{b_f}\bigl[r^\top D^{-1} r\bigr] \;+\; \tfrac12\log\det(2\pi D),
\qquad r = G_s s - c_s,
$$

with $H_R$ from $N^\top \Lambda' N$ ($N$ a basis of $\operatorname{null}(G_h)$). Under $b_f$, $s = s_0 + N z$ with $s_0$ any
solution of $G_h s_0 = c_h$ and $z \sim \mathcal N\bigl((N^\top\Lambda'N)^{-1}N^\top(\eta' - \Lambda' s_0),\ (N^\top\Lambda'N)^{-1}\bigr)$; the
mean of $s$ does not depend on the choice of $s_0$. With $r = 0$ soft equations this is the noise-free term, and
with $r = m$ it is `LinearConstraintFactor`'s. The rank condition is on $G_h$ alone: the soft equations may
outnumber the free coordinates.

Graded as `fit` $= \tfrac12\mathbb E[r^\top D^{-1} r]$, `complexity` $= \tfrac12\log\det G_hG_h^\top + \tfrac12\log\det 2\pi D$,
`negentropy` $= -H_R$.

**Derivatives.** $Q$ is a constant of the factor (`Float64`), so its split is taken once, on values,
and carries no tangent. $G_h = Q_h^\top \tilde G$ carries the inputs' tangents, and its SVD takes P4's
constant-rank terms as before; $s_0 = G_h^\top (G_h G_h^\top)^{-1} c_h$ and the soft part are plain arithmetic,
so ForwardDiff and Zygote go through them. `islearnable` stays `false`, so the row's "learnable" means
the derivatives of the other factors' parameters through this term (`test/hard_gradient.jl`).

## Implementation

- The parameters and polarities are `LinearConstraintFactor`'s: `(A, c)`, any one channel
  `Unobserved()`, the others `Observed()` or `Latent()`. `noise` defaults to $0$ and may be any
  positive semidefinite `m × m` matrix (soft along its range, hard elsewhere).
- Inputs become extended Gaussians by `LenticulumCore.extended`: a `GaussianBelief` (also improper), a
  `DiracBelief`, an `ExtendedGaussian`, or nothing (`TrivialBelief()`, no message: flat). A
  mixture (of Gaussians or of extended Gaussians) is collapsed to its total moments
  (`moment_match`), as `LinearConstraintFactor` collapses one: exact for the collapsed input, an
  assumed-density approximation for the mixture (P27, 2026-11-17; refused before).
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
| refusals: redundant equations on a clamped input, a belief flat along $R$, redundant exact equations of a singular noise; with a positive definite noise, `LinearConstraintFactor`'s value | 4 | exact |
| P5, singular noise: the tree above with a noise of rank $0 < r < m$ (up to $\dim u + \dim v + 1$ equations): marginals against the conditioned Gaussian with $C\Sigma C^\top + Q$, the Bethe free energy against $-\log Z$ | 40 random trees, $u$ with hard directions in 17 | within $10^{-7}$ (marginals) and $10^{-7}(1 + \lvert\log Z\rvert)$ |
| P5: the term against the noisy relation's with noise $Q_sDQ_s^\top + \varepsilon Q_hQ_h^\top$, $\varepsilon = 10^{-30}$, 512-bit | 80 draws as above, the 70 with a free coordinate | largest relative difference $3.8 \cdot 10^{-15}$ (the test asks for $10^{-9}$) |
| P5 by hand: $x \sim \mathcal N(0, I_2)$, $x = \varepsilon$ with $\varepsilon \sim \mathcal N(0, \operatorname{diag}(1, 0))$ | 1 | marginal $\operatorname{diag}(0.5, 0)$; $F = \tfrac12\log 4\pi + \tfrac12\log 2\pi$ |

`test/hard_gradient.jl`, P5: the coupled replay through a relation with a singular noise against
ForwardDiff of the dense $-\log Z$, on 12 random trees: reference differences within $10^{-5}$,
ForwardDiff and Zygote within $10^{-7}$ relative.

## Plates (P6)

A plate of tied hard relations (`factor_messages`, one call for all replicas) shares the work when
the replicas have one noise and one set of dimensions, and each channel's inputs share their
covariance and flat directions across the replicas: points, no message, `TrivialBelief()`,
Gaussians of one precision (also improper, flat along its null space), extended Gaussians of one
`Σ` and `D`. That is the case of data under a shared prior. Every decomposition in
`factor_message` (the hard and flat directions of the pushforward $r$, its pseudo-inverse precision
$\Lambda_r$, the SVD of $C = H^\top A_t$ and the precision on its null space) then depends on the
covariances only, so replica 1 computes them, and each replica's message has replica 1's $\Sigma$ and
$D$ with the mean

$$
\mu_i \;=\; \Pi_o\Bigl[\,P H^\top + G\bigl(A_t^\top \Lambda_r - \Lambda_p P H^\top\bigr)\Bigr]\,\Pi_r\Bigl(c - \textstyle\sum_{n \ne t} A_n m_{n,i}\Bigr),
$$

where $m_{n,i}$ is replica $i$'s input mean on channel $n$, $\Pi_r$ projects onto the complement of the
pushforward's flat directions, $H$ is its hard directions, $P = C^+$, $N$ a basis of $\operatorname{null}(C)$,
$\Lambda_p = A_t^\top\Lambda_r A_t$, $G = N (N^\top \Lambda_p N)^+ N^\top$, and $\Pi_o$ projects onto the complement of
the message's flat directions. This is `_from_potential`'s $x = x_0 + N S_z N^\top(\eta - \Lambda_p x_0)$, with
$x_0 = P H^\top \mu_r$ and $\eta = A_t^\top \Lambda_r \mu_r$, written as one matrix for all replicas. Whether a replica's
hard constraints have a common solution depends on its mean, so it is checked per replica, with
`_from_potential`'s residual and tolerance: a contradiction in replica $i$ throws as it would alone.
The messages are built with a trusted `ExtendedGaussian` constructor (the parts are already a
normalised representative), so a replica costs a few matrix-vector products and no decomposition.

Anything else takes `factor_message` per replica: inputs that differ in structure, any number that is
not a `Float64` (ForwardDiff duals), and a Zygote reverse pass (`_within_gradient()` is `true`
there through an `rrule`), since the shared path takes its decompositions on values without the
constant-rank derivative terms of LenticulumCore's. The per-replica loop is a comprehension, as
Zygote cannot differentiate writes into a vector (Mycelium's default loop writes; P6 found that
Zygote through it fails).

From message blocks the result is a `VectorBlock` of the messages: there is no block type for
extended Gaussians, so the variable's side combines them replica by replica.

**Measured** (`test/hard_plates.jl`, 300 random instances: 2 or 3 channels of dimension 1 to 3, every
kind of input, zero, positive definite and singular noise, 2 to 5 replicas; 278 went through the
shared path, 22 were refused by the per-replica path and by the batch alike): the largest relative
deviation from `factor_message` over mean, covariance and flat projector is $9.5 \cdot 10^{-12}$. The
graph-level oracle is `test/extended_plates.jl` (a plate of `GaussianFactor`s under hard relations
against the dense conditioned Gaussian), which now goes through this method: counted once with
an instrumented copy, 261 of its group steps took the shared path and 20 the per-replica one. Timing (one machine,
2026-11-17, a relation on $(u, v, t)$ of dimensions $(3, 2, 3)$, $m = 3$, Gaussian $u$ of one precision and
point $v$, the message to $t$; a scratch script, not in the repository, so a rough figure): 0.045 ms
against 0.27 ms for 10 replicas, 0.081 against 2.7 ms for 100, 0.44 against 27 ms for 1000.

Review: [[LP-044 Findings of the review of P6]] (the block fallback under Zygote, fixed: it built the
block with `belief_block`, which Zygote cannot take; argument lengths now checked).

## Mean-field nodes

With a positive definite noise the factor is `LinearConstraintFactor` with that noise, and its
mean-field message and average energy are that factor's (LP-044; tested equal on random factors). With
zero noise, or a singular one (exact along its null space), they are refused with the reason. The mean-field message $\exp \mathbb E_q[\log f]$ of the
relation is the limit, as $\varepsilon \to 0$, of `LinearConstraintFactor`'s with noise $\varepsilon I$:
$\mathcal N^{-1}(A_t^\top \bar d/\varepsilon, A_t^\top A_t/\varepsilon)$ with $\bar d = c - \sum_{i \ne t} A_i \mathbb E_q[x_i]$, hard along
$\operatorname{range}(A_t^\top)$ at $A_t x_t = \bar d$. The marginal $q(x_t) \propto p(x_t)\,\mu_{f\to t}(x_t)$ then
sits there exactly, whatever the prior says along those directions. Two scalar variables with
$u - v = 0$ and priors $\mathcal N(0, 1)$, $\mathcal N(2, 1)$: the update is $\mathbb E u \leftarrow \mathbb E v$,
$\mathbb E v \leftarrow \mathbb E u$, so every $u = v$ is a fixed point and the iteration stays where it starts, while
the posterior has mean 1. With noise $\varepsilon$ the update is $\mathbb E u \leftarrow \mathbb E v/(1 + \varepsilon)$ (and alike), a
contraction by $(1 + \varepsilon)^{-2}$ per sweep that vanishes as $\varepsilon \to 0$. The average energy
$\mathbb E_q[-\log f]$ is infinite unless each $A_i x_i$ is deterministic under $q$; its finite part is not the
Bethe one either, as the divergent terms (the factor's $\tfrac m2\log\varepsilon$, the variables'
$\tfrac{\operatorname{rank} A_i}{2}\log\varepsilon$) do not cancel unless $\sum_i \operatorname{rank} A_i = m$. Belief propagation, exact on trees, is the
rule for an exact relation; a `MeanField()` node wants a positive definite noise.

## What it does not do

- **Learning**: `islearnable` is `false`: its own `A` and `c` get no gradient. Other factors in
  its graph do: the coupled replay (`free_energy_gradient(...; schedule, ad)`) goes through its
  messages and its free energy under ForwardDiff and Zygote (P4: the SVD of its term is of values,
  with $\tfrac12\log\det GG^\top = \sum\log\sigma_i + \operatorname{tr}(G^+\Delta)$ and $N = N_0 - G^+\Delta N_0$ at constant
  rank, LenticulumCore `extended_gaussian.md`, "Derivatives"); `test/hard_gradient.jl` checks it
  against the gradient of the dense $-\log Z$ on random trees. `fit_em` with the factor in the graph
  is not tested.
- **A nearly singular noise** (an eigenvalue above `_eig_tol` but tiny, say a relative $10^{-11}$)
  counts as positive definite and goes through `LinearConstraintFactor`'s formulas, messages
  included, which invert it: 6 of 20 random trees with such a noise were off by up to $1.6\cdot10^{-6}$
  in $F$ and $4\cdot10^{-5}$ in the marginals ([[LP-047 Findings of the review of P5]], finding 4; a
  review script, not a test). A formulation in the residual space would avoid the inverse.
- **Plates whose inputs differ in structure**, dual numbers and Zygote take the per-replica path
  ("Plates"); there is no block type for extended Gaussians.
- **Mean-field messages** (`MeanField()` nodes) of an exact relation: refused ("Mean-field nodes").
- **Mixture inputs** exactly: they are collapsed to their moments (P27), not expanded into one
  message per component.

## Difficulties

The first comparison with `LinearConstraintFactor` failed in 23 of 149 messages, all with every
other input uninformative. The pushforward is then flat in every direction, and projecting $\Sigma$
with $I - DD^\top$ left a residue of $10^{-30}$, which the relative tolerance took for a real variance:
the message came out hard where it should be flat. The extended Gaussian now projects through an
orthonormal basis of the complement of $D$, which is empty in that case, so $\Sigma$ is exactly zero.

The damped flooding run of one instance (an ill-conditioned $A$) stops at a residual of
$3.8 \cdot 10^{-10}$, a rounding floor of the absolute convergence bound (LP-024's open row), so the test
asks for $10^{-8}$; its marginals agree to $10^{-7}$.

P5's first tree test failed in one instance: a noise $BB^\top$ of rank 1 had a zero eigenvalue
computed as about $10^{-16}$, `isposdef` accepted it, and the free energy took
`LinearConstraintFactor`'s path while the messages (which split at `_eig_tol`) had treated that
direction as exact. The free energy now decides positive definiteness by the same tolerance. The
512-bit oracle met the same rounding the other way: $Q + \varepsilon Q_hQ_h^\top$ at $\varepsilon = 10^{-30}$ was
indefinite, as $Q$'s null eigenvalues in `Float64` are $\pm 10^{-17}$; the oracle rebuilds the noise
from its split.
