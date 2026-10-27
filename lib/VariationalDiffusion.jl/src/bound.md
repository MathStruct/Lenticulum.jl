#implementation #derivation

> A separately typed negative ELBO for a normalised finite VP diffusion chain.
> Analytic Gaussian expectations give an upper bound; sampled expectations return an
> explicitly different estimate type. The graph's denoising surrogate is unchanged.

> Sources: original finite Gaussian-chain derivation, checked in `../test/bound.jl`; code: `bound.jl`, `factor.jl`.
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

## Normalised model and variational law (2026-10-27, LP-009)

The caller supplies strictly increasing `times` in $(0,1]$. Set
$a_i=\alpha_i/\alpha_{i-1}$, $\beta_i=1-a_i^2$, $\alpha_0=1$ and
$\sigma_i^2=1-\alpha_i^2$. These define a finite model, including its decoder:

$$
q(x_i\mid x_{i-1})=N(a_i x_{i-1},\beta_i I),\quad
p(x_T)=N(0,I),\quad
p_\theta(x_{i-1}\mid x_i)=N(m_i(x_i),\beta_i I),
$$

$$
m_i(z)=\frac{z-\beta_i\varepsilon_\theta(z,t_i)/\sigma_i}{a_i}.
$$

For full joint evidence $x=x_0$, the variational law is $q(x_{1:T}\mid x)$.
The negative ELBO includes the terminal KL, every reverse-transition KL, and the
Gaussian decoder negative log likelihood, with all constants. Jensen gives

$$
B(x)=-\mathrm{ELBO}(x)=-\log p_\theta(x)
 +\mathrm{KL}(q(x_{1:T}\mid x)\Vert p_\theta(x_{1:T}\mid x))
 \ge -\log p_\theta(x).
$$

For $i>1$, the forward posterior has mean
$\alpha_{i-1}\beta_i x/\sigma_i^2+a_i\sigma_{i-1}^2z/\sigma_i^2$ and
variance $\beta_i\sigma_{i-1}^2/\sigma_i^2$. Its Gaussian KL leaves an expectation
only over $z\sim N(\alpha_i x,\sigma_i^2 I)$. For a single-component
`GaussianMixtureEps`, $m_i$ is affine, so its squared residual expectation is exactly
its squared mean plus a covariance trace. No numerical quadrature is called exact.

`free_energy_upper_bound(f,x,ps,st;times)` returns `FreeEnergyUpperBound(value)`.
It supports that analytic single-Gaussian predictor with the default input adapter,
a shared time, matching model/predictor schedules and no device transfer. `estimate_free_energy_upper_bound` supports
other predictors using independent Gaussian marginal draws at each time and returns
`FreeEnergyBoundEstimate(value,standard_error,nsamples)`. Its expectation is a bound;
an individual estimate may fall below the exact negative log evidence. Both APIs
require inference-mode state and preserve the supplied state tree, including ProxDM's
separate predictor tree. Neither wrapper is a `Real` or accepted by `scalarise`.

## Checked scope and difficulties

`../test/bound.jl` checks the stationary $N(0,I)$ case: its analytic score is $-z$,
all reverse transitions preserve that Gaussian, and the bound equals its closed-form
negative log density. Displacing the analytic score's Gaussian mean gives an affine
reverse chain whose marginal is calculated independently; the bound stays above that
model's exact negative log density and the gap decreases as the displacement vanishes.
Monte Carlo estimates are compared with the analytic expectation using their standard
errors; unsupported schedules, times, dimensions and exact expectations are rejected.

This API evaluates full joint point evidence only. It does not provide partial-channel
evidence, a continuous-time likelihood, a bound on `(clamp,score)`, or the entropy of the
prox posterior. `free_energy` and `entropy` on DiffusionFactor continue to throw with
these reasons. The common factor checker still applies only to the existing surrogate
message path; its game exemption remains justified. A finite Gaussian decoder and a
terminal distribution are essential: the denoising loss alone omits normalisers and is
not an ELBO. `times` changes the model as well as the variational law.
