#implementation

> The VP-SDE of [Song et al. 2021](https://arxiv.org/abs/2011.13456), presented through its
> perturbation kernel rather than its SDE.

> Sources: code: `schedule.jl`

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

`AbstractNoiseSchedule` requires only `alpha(s,t)` and `sigma(s,t)`. `drift` and
`diffusion` are provided and **nothing in this package calls them** — RED-Diff replaces
sampling by optimisation, so the SDE is never integrated.

This is a deliberate narrowing and it has a cost: a schedule whose marginal is not Gaussian
does not fit. Cold diffusion, discrete/categorical diffusion and flow matching with a
non-Gaussian path all fall outside. What *does* fit trivially is sub-VP and VE — both are
another pair of $(\alpha_t,\sigma_t)$ formulas — and neither is implemented.

## 3. Time measure and the likelihood weighting

Use the power convention $\mathrm{SNR}_t=\alpha_t^2/\sigma_t^2$ throughout this note.
The historical `snr(s,t)` function returns the amplitude ratio $\alpha_t/\sigma_t$,
that is, $\sqrt{\mathrm{SNR}_t}$; its API is unchanged.

For a small VP step, let $a=\alpha_t/\alpha_{t-dt}$ and $b=1-a^2=\beta(t)dt+o(dt)$.
A unit change in the predicted noise changes the reverse mean by $b/(a\sigma_t)$.
The Gaussian KL's quadratic term, with reverse variance $b$, is therefore
$b/(2a^2\sigma_t^2)$ times squared noise error. Dividing by $dt$ gives

$$
\omega_{\mathrm{ELBO}}(t)=-\tfrac12\frac{d}{dt}\log\mathrm{SNR}_t
=\beta(t)/(2\sigma_t^2).
$$

Indeed $d\log\alpha_t^2/dt=-\beta(t)$ and
$d\log\sigma_t^2/dt=\beta(t)\alpha_t^2/\sigma_t^2$, whose difference is
$-\beta(t)/\sigma_t^2$. This is the coefficient in the continuous likelihood-bound
integral $\int_{t_{\min}}^1\omega_{\mathrm{ELBO}}(t)\,\mathbb E\|\hat\varepsilon-\varepsilon\|^2dt$;
endpoint/reconstruction and terminal terms must also be included for a bound.
Written as an expectation under $t\sim U[t_{\min},1]$, the coefficient is multiplied
by $1-t_{\min}$. `test/lp018.jl`, “LP-018.10 ELBO epsilon weight from Gaussian KL”,
checks this limit against independent finite-step Gaussian KLs.

| quantity | meaning |
|---|---|
| $\omega_{\mathrm{ELBO}}(t)$ | coefficient of squared noise error under $dt$ |
| $1$ | unweighted squared noise error, DDPM's simple training loss |
| $\lambda_t=\lambda\sigma_t/\alpha_t=\lambda/\sqrt{\mathrm{SNR}_t}$ | RED-Diff's stop-gradient residual field weight under uniform time sampling |

The last row weights a field, not the squared-error likelihood integrand. The missing
denoiser Jacobian and different time measure prevent identifying it with the ELBO coefficient
or comparing them by squaring weights; see [[reddiff]] §3.

## 4. Implementation difficulties

### 4.1 `tmin` is a floor everyone has and nobody documents

At $t=0$, $\sigma_t = 0$: the score $-\varepsilon_\theta/\sigma_t$ divides by zero and
RED-Diff's weight $\lambda\sigma_t/\alpha_t$ vanishes. `VPSDE` therefore carries
`tmin = 1e-3` and `sample_time` draws from $[t_{\min},1]$.

This is a **modelling choice smuggled in as a numerical guard**. It changes the value of every
integral in the package — including `calibrate_lambda`, whose answer depends on
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

## LP-018 convention correction (2026-10-29)

The likelihood coefficient and time measure in §3 replace the incorrect inverse-SNR
claim. `snr` keeps its historical amplitude-ratio return value; only the documentation
now distinguishes it from power SNR. The finite-step Gaussian KL regression in
`test/lp018.jl` checks the derivation without treating RED-Diff's field as that bound.
