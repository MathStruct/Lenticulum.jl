# schedule.jl — implementation note

> The VP-SDE of [Song et al. 2021](https://arxiv.org/abs/2011.13456), presented through its
> perturbation kernel rather than its SDE.

## 1. What is implemented

$$
dx = -\tfrac12\beta(t)x\,dt + \sqrt{\beta(t)}\,dw,
\qquad
\beta(t) = \beta_{\min} + t(\beta_{\max}-\beta_{\min}),
\qquad t\in[0,1]
$$

with closed-form marginals

$$
p_{0t}(x_t\mid x_0) = \mathcal{N}(x_t;\ \alpha_t x_0,\ \sigma_t^2 I),
\qquad
\alpha_t = e^{-B(t)/2},
\qquad
\sigma_t^2 = 1-e^{-B(t)},
\qquad
B(t)=\int_0^t\beta
$$

Because $\beta$ is affine, $B(t) = \beta_{\min}t + \tfrac12(\beta_{\max}-\beta_{\min})t^2$ in
closed form. **That is the only reason the VP-SDE is convenient here**: a schedule needing
quadrature for $B$ would put a numerical integral inside every message.

The defining identity is $\alpha_t^2 + \sigma_t^2 = 1$ — *variance preserving*: unit-variance
data has unit variance at every $t$. It is asserted in the test suite at eight values of $t$,
because it is the one property the whole file exists to have.

## 2. The interface is the kernel, not the SDE

[`AbstractNoiseSchedule`](@ref) requires only `alpha(s,t)` and `sigma(s,t)`. `drift` and
`diffusion` are provided and **nothing in this package calls them** — RED-Diff replaces
sampling by optimisation, so the SDE is never integrated.

This is a deliberate narrowing and it has a cost: a schedule whose marginal is not Gaussian
does not fit. Cold diffusion, discrete/categorical diffusion and flow matching with a
non-Gaussian path all fall outside. What *does* fit trivially is sub-VP and VE — both are
another pair of $(\alpha_t,\sigma_t)$ formulas — and neither is implemented.

## 3. Three weightings, one integrand

`denoising_loss` returns the **unweighted** $\|\varepsilon_\theta - \varepsilon\|^2$. That is
not laziness: the weighting is where the objective's identity lives.

| $\omega(t)$ | gives |
|---|---|
| $\sigma_t^2/\alpha_t^2 = 1/\mathrm{SNR}_t^2$ | the ELBO / exact likelihood bound |
| $1$ | DDPM's "simple" loss — the one everybody trains |
| $\lambda/\mathrm{SNR}_t = \lambda\sigma_t/\alpha_t$ | **RED-Diff's regulariser** |

So the same network, the same integrand, three different meanings. `reddiff.md` §3 is about
what the third choice does.

## 4. Implementation difficulties

### 4.1 `tmin` is a floor everyone has and nobody documents

At $t=0$, $\sigma_t = 0$: the score $-\varepsilon_\theta/\sigma_t$ divides by zero and
RED-Diff's weight $\lambda\sigma_t/\alpha_t$ vanishes. `VPSDE` therefore carries
`tmin = 1e-3` and `sample_time` draws from $[t_{\min},1]$.

This is a **modelling choice smuggled in as a numerical guard**. It changes the value of every
integral in the package — including [`calibrate_lambda`](@ref), whose answer depends on
`tmin` through the lower limit. Two implementations with different floors compute different
posteriors and neither is wrong. Recorded because the constant is invisible in the maths and
load-bearing in the code.

### 4.2 `sigma` must not be `sqrt(1 - alpha^2)`

Near $t=0$, $\alpha_t = 1-\varepsilon$ and `sqrt(1 - alpha^2)` evaluates
$\sqrt{1-(1-\varepsilon)^2}$, losing roughly half the significant digits to cancellation. The
code uses `sqrt(-expm1(-B(t)))` instead, which is accurate to full precision. Asserted in the
test suite at $t = 10^{-8}$ against the small-$B$ asymptote $\sigma \approx \sqrt{B}$.

### 4.3 Times are scalars, so a batch is not batched

`alpha(s, t)` takes a scalar `t`, and `perturb` broadcasts it over `x`. A real training loop
draws a *different* $t$ per batch element. Nothing here forbids it — `t` could be an array
and the broadcasts would work — but nothing tests it either, and `sample_time` returns one
scalar. This package is written for the inference-time inner loop, where one $t$ per step is
what RED-Diff's Algorithm 1 asks for.

### 4.4 No discretisation, hence no sampler

There is no ancestral sampler, no Euler–Maruyama, no probability-flow ODE, and therefore **no
way to draw from the prior**. That is consistent — RED-Diff never samples — but it means the
package cannot be sanity-checked by generating from the model, only by the analytic oracle in
the test suite.

Related: [[The VP-SDE]], [[predictor]], [[reddiff]]
