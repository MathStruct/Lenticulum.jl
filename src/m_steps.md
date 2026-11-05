#implementation

> Closed-form M-steps for `LinearConstraintFactor`, `SwitchingConstraintFactor` and
> `RegimeFactor` (`Mycelium.m_step`): weighted least squares in the learned coefficient blocks
> given the fixed ones, the outcome masses as weights, and per-outcome noise for `RegimeFactor`.

> Sources: code: `m_steps.jl`; Mycelium `lib/Mycelium.jl/src/em.jl` (the statistics and `em_step`); tests: `test/em.jl`
>
> Theory (CT-ML wiki): [[Switching Constraint Factors]]

## 1. The minimiser

With $x$ the factor's continuous channels and $\varphi = [x;\, e_k]$, the residual is $r = W \varphi$, $W = [A,\ -B]$, and

$$
\mathbb E_b[-\log f] = \tfrac12 \operatorname{tr}\bigl(Q^{-1} W S_\varphi W^\top\bigr) + \tfrac12 \sum_k n_k \log\det 2\pi Q_k - \sum_k n_k \log w_k,
\qquad
S_\varphi = \begin{pmatrix} \sum_k S_k & M \\ M^\top & \operatorname{diag}(n) \end{pmatrix},
$$

$M = [m_1, \dots, m_K]$. Split the columns of $W$ into learned ($L$) and fixed ($F$: a declared output,
frozen paths). The minimiser over $W_L$ is the same for every $Q$:

$$
W_L = -W_F\, S_{FL}\, S_{LL}^{+}, \qquad \text{plus } W_L^{\text{old}} (I - S_{LL} S_{LL}^{+}),
$$

the second term keeping the old values in the directions the statistics do not see (an outcome
of mass zero, a channel that never varies). The weights are $w_k = n_k / \sum_j n_j$. `RegimeFactor` solves
per outcome ($\varphi = [x; 1]$ with that outcome's moments) and, with learned noise, sets
$Q_k = W_k S_{\varphi,k} W_k^\top / n_k$ and stores its Cholesky factor (`L`, log-diagonal).

**With nothing fixed the minimiser is $W = 0$** (LP-022): a relation normalised in its residual
puts every point on itself. `m_step` refuses it with an `ArgumentError` naming LP-022. Declare an
`output` or freeze a coefficient.

## 2. Checked by

`test/em.jl`:
- **A mixture of regressions** with one noise level per line: the graph's EM follows hand-written
  EM (weighted least squares and residual variance per line) to $10^{-10}$ after 1, 3 and 8 steps, and
  $F$ never increases. Frozen weights stay.
- **A mixture of factor analysers** in a cluster per point (three regimes, $x \in \mathbb R^2$ latent): EM
  lowers $F$ by more than 100 and stops where the free-energy gradient (ForwardDiff) in $A$, $b$ and
  the noise is below $10^{-4}$.
- **Switches over a frozen sum:** stationary to $10^{-5}$, the frozen paths unchanged, SQUAREM
  reaching plain EM's fixed point.
- **The refusal** for a sum with every coefficient learned.

## 3. Gaussian factors (2026-11-05)

`GaussianFactor` ($y = Ax + b + \varepsilon$) and `GaussianPrior` have cluster forms now: the residual $y - Ax - b$ with the
coefficient of $y$ fixed at the identity, and the current noise covariance (`noise_covariance`,
the eigenvalue floor included). Learned noise is allowed in a cluster node, a noise prior is not
(its density is not part of a cluster's free energy). Their M-steps:

- `GaussianFactor`: least squares in $A$ and $b$ by the same block solve (the $y$ block fixed), and,
  with learned noise and no prior, the residual covariance $W S_\varphi W^\top / n$. It is floored as
  `_noise_metrics` floors it (covariance eigenvalues at least `eigenvalue_floor`) and stored in the
  factor's own coordinates (log-precisions, or the precision's Cholesky factor).
- `GaussianPrior`: its mean is fixed. With learned noise and no prior its covariance is
  $\mathbb E[(x - \mu)(x - \mu)^\top]$.
- With a noise prior the noise's M-step is a MAP problem that is not done. The noise keeps its
  value, and $A$, $b$ are still updated, since their minimiser does not depend on $Q$.

`test/em.jl`, "Gaussian factors":
- **Regression, fully observed:** one M-step gives least squares to $10^{-10}$ and the residual
  covariance to $10^{-8}$.
- **Factor analysis** ($x$ latent, diagonal noise): the clustered free energy equals the
  unclustered one to $10^{-10}$, EM does not raise it, and it stops where the gradient in $A$, $b$ and
  the log-precisions is below $10^{-5}$.
- **A prior's learned covariance** equals the sample covariance about its mean.
- **The refusal** of a noise prior in a cluster.

## 4. What it does not do

- `LinearConstraintFactor` and `SwitchingConstraintFactor` have fixed noise, so it is not
  updated.
- No MAP M-step for noise priors (Wishart, log-normal); no M-step for `NonlinearFactor`,
  `LikelihoodFactor` (not learnable) or the other families.
