#implementation

> Closed-form noise predictors: the optimal $\varepsilon^\ast$ of a Gaussian mixture, with its exact
> input Jacobian and parameter VJP. A ring of narrow components is a circle **relation with an
> exact score**, which makes implicit inference and its backward pass checkable against
> arithmetic.

> Sources: code: `analytic.jl`
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

## 1. The formulas

For $p_0 = \frac1J\sum_j\mathcal N(\mu_j, s^2I)$ under the VP-SDE, $p_t$ is the mixture with means
$\alpha_t\mu_j$ and variance $v_t = \alpha_t^2s^2+\sigma_t^2$. With $u_j = (x-\alpha_t\mu_j)/v_t$, responsibilities
$\gamma_j \propto \exp(-\lVert x-\alpha_t\mu_j\rVert^2/2v_t)$ and $\bar u = \sum_j\gamma_ju_j$:

$$
\varepsilon^\ast = \sigma_t\,\bar u,\qquad
\partial_x\varepsilon^\ast = \sigma_t\Bigl[\tfrac{1}{v_t}I - \textstyle\sum_j\gamma_j u_j(u_j-\bar u)^\top\Bigr],\qquad
(\partial_{\mu_j}\varepsilon^\ast)^\top w = \sigma_t\alpha_t\gamma_j\bigl[-\tfrac{w}{v_t} + u_j\,(u_j-\bar u)^\top w\bigr].
$$

The input Jacobian is $\sigma_t$ times ($1/v_t$ minus a covariance of the $u_j$), so it is
**symmetric**, as $-\sigma_t\nabla^2\log p_t$ must be.

## 2. Implementation difficulties

- **The sign of the covariance term.** The first prototype had $+\operatorname{Cov}$; finite differences caught it (error 0.58). The responsibilities' gradient is $\gamma_j(\bar u - u_j)$, not $\gamma_j(u_j - \bar u)$.
- **Log-sum-exp.** Responsibilities are computed from shifted log-weights, because at small $t$ the exponents are of order $1/v_t\sim10^4$.
- **Equal weights, isotropic components.** Enough for relations along curves; general weights and covariances would add parameters but no new ideas.

## 3. Why it lives in `src/` and not in `test/`

It is an *oracle*, the same role `GaussianFactor` plays in the parent package. It is also a
usable model: a mixture whose means are trained by [[implicit]]'s adjoint is an implicit learner
with an exact score, which [[Backpropagation through Implicit Inference]] §7 uses to learn a
parabola from a circle.

Related: [[implicit]], [[predictor]], [[Implicit Diffusion Learners]]
