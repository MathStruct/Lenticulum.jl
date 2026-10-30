#implementation

> The linear-Gaussian factor with its weight `A` and/or bias `b` exposed as channels: with a
> point input, the factor is linear-Gaussian in the exposed blocks, so its messages and Bethe
> term are exact closed forms.

> Sources: code: `exposure.jl`; the generic wrapper in Mycelium's `exposure.jl`; tests in `test/exposure.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

Implements L-SPEC-003 §1.2 ([[L-SPEC-003 Exposed Parameters and Energy Channels]]) for
`GaussianFactor`.

## Theory

With the input clamped at $x_0$ the residual is

$$
r = y - A x_0 - b = C_y\, y + C_A \operatorname{vec}A + C_b\, b,
\qquad C_y = I,\quad C_A = -(x_0^\top \otimes I),\quad C_b = -I,
$$

using $\operatorname{vec}(A x_0) = (x_0^\top \otimes I)\operatorname{vec}A$. The factor
$\mathcal N(r; 0, Q)$ is therefore a linear-Gaussian relation among the blocks that carry
beliefs, and every message is the usual one. For a target block $t$, the other blocks'
means $m_k$ and covariances $S_k$ (a point has $S_k = 0$; an unexposed block is the point in
`ps`) give

$$
s = \sum_{k \ne t} C_k m_k, \qquad V = Q + \sum_{k \ne t} C_k S_k C_k^\top, \qquad
\mu_{c \to t} = \mathcal N^{-1}\bigl(-C_t^\top V^{-1} s,\; C_t^\top V^{-1} C_t\bigr).
$$

To the output this is the posterior predictive $\mathcal N(\bar A x_0 + \bar b,\ C_A S_A C_A^\top + S_b + Q)$; to $\operatorname{vec}A$ it has rank `nout` when $x_0 \ne 0$ (rank 0
when $x_0 = 0$), so it is improper unless `nin = 1`, and a prior on the weight makes the
belief proper. An incoming belief on a block may itself be improper as long as the directions
$C_k$ sees are proper ($C_k\Lambda^+\Lambda = C_k$): then $C_k z_k$ has mean $C_k\Lambda^+\eta$ and
covariance $C_k\Lambda^+C_k^\top$; otherwise the message throws. If another block has no
information and a full-row-rank coefficient ($I$, or $x_0^\top \otimes I$ with $x_0 \ne 0$), the
residual is unconstrained and the message is exactly uninformative.

The Bethe term uses the joint factor belief over the stacked non-point blocks $z$,
$b_c(z) \propto \mathcal N(Gz + c; 0, Q) \prod_k \mu_{k \to c}(z_k)$, with precision
$G^\top Q^{-1} G + \operatorname{blockdiag}(\Lambda_k)$ and information
$\oplus_k \eta_k - G^\top Q^{-1} c$, and reports the same `(fit, complexity, negentropy)`
grading as `local_free_energy` (shared through `_graded_free_energy`).

## What it does not do

- A belief on $x$ together with one on $A$: $Ax$ is bilinear, the joint is not Gaussian, and
  both the messages to $x$ and the free energy throw rather than approximate (VMP or EP would
  be the next step).
- Exposing the noise covariance: `ps.noise` exists only with `learn_noise`, and a belief on a
  covariance is not Gaussian (a Wishart belief is not in LenticulumCore).

## Measured

`test/exposure.jl` (2026-10-30): Bayesian linear regression with seven tied observations
matches the closed-form posterior, predictive and $-\log p(y)$; the random-slopes hierarchy
built from thirty exposed regression factors matches the dense reference of
`examples/hierarchical_models.jl` (means rtol $10^{-9}$, $-\log p(y)$ rtol $10^{-10}$).
