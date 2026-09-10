# The VP-SDE

> Song et al. 2021, [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) §3.4 — the forward
> corruption process, and the three quantities a trained $\varepsilon_\theta$ hands you for
> free.
>
> Implemented in `lib/VariationalDiffusion.jl/src/schedule.jl`; see [[schedule]].

## 1. One SDE, and why this one

$$
dx = -\tfrac12\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw,
\qquad
\beta(t) = \beta_{\min}+t(\beta_{\max}-\beta_{\min}),
\qquad
t\in[0,1]
$$

Song's paper unifies the two families that existed separately before it:

| SDE | drift | ends at | discrete ancestor |
|---|---|---|---|
| **VP** (variance preserving) | $-\tfrac12\beta x$ | $\mathcal{N}(0,I)$ | DDPM |
| VE (variance exploding) | $0$ | $\mathcal{N}(0,\sigma_{\max}^2I)$ | SMLD / NCSN |
| sub-VP | $-\tfrac12\beta x$, smaller noise | $\mathcal{N}(0,I)$ | — |

VP is the one implemented, because it is DDPM's continuous limit and therefore what almost
every pretrained checkpoint speaks.

## 2. The perturbation kernel is the whole interface

Solving the SDE gives a closed-form Gaussian marginal:

$$
p_{0t}(x_t\mid x_0) = \mathcal{N}\bigl(x_t;\ \alpha_t x_0,\ \sigma_t^2 I\bigr),
\qquad
\alpha_t = e^{-B(t)/2},
\qquad
\sigma_t^2 = 1 - e^{-B(t)},
\qquad
B(t) = \int_0^t\!\beta
$$

and because $\beta$ is affine, $B(t) = \beta_{\min}t + \tfrac12(\beta_{\max}-\beta_{\min})t^2$
**in closed form** — the only reason this schedule is convenient in an inner loop.

The defining identity:

$$\boxed{\alpha_t^2 + \sigma_t^2 = 1}$$

*Variance preserving*: unit-variance data has unit variance at every $t$. More generally
$\mathrm{Var}(x_t) = \alpha_t^2 v_0 + \sigma_t^2$, which equals $1$ iff $v_0 = 1$ — so the
name is a statement about **normalised** data, and it is why every diffusion pipeline
normalises to unit variance first. Feed it $v_0 = 4$ and the process is not variance
preserving at all; it merely converges to $\mathcal{N}(0,I)$ anyway.

> [!note] The implementation exposes the kernel, not the SDE
> `AbstractNoiseSchedule` requires only `alpha` and `sigma`. `drift` and `diffusion` exist for
> completeness and **nothing calls them**, because RED-Diff replaces sampling with
> optimisation. A schedule with a non-Gaussian marginal — cold diffusion, discrete diffusion —
> does not fit this interface at all.

## 3. Three quantities from one network

A network trained on $\|\varepsilon_\theta(\alpha_tx_0+\sigma_t\varepsilon,t)-\varepsilon\|^2$
gives you three things, of which it only learned one:

$$
\underbrace{\varepsilon_\theta(x,t)}_{\text{learned}}
\qquad
\underbrace{s_\theta(x,t) = -\frac{\varepsilon_\theta(x,t)}{\sigma_t}}_{\text{the score}}
\qquad
\underbrace{\hat x_0(x,t) = \frac{x-\sigma_t\varepsilon_\theta(x,t)}{\alpha_t}}_{\text{Tweedie's denoiser}}
$$

**The score identity is exact**, not approximate: for the perturbation kernel,

$$\nabla_{x_t}\log p_{0t}(x_t\mid x_0) = -\frac{x_t-\alpha_t x_0}{\sigma_t^2} = -\frac{\varepsilon}{\sigma_t}$$

so predicting the noise and estimating the score are the same task in different units. This
is why $\varepsilon$-parametrisation won: it is the score, scaled to unit variance at every
$t$, which is far better conditioned to regress.

**Tweedie's formula** is what makes a diffusion model a prior rather than a sampler:
$\hat x_0 = \mathbb{E}[x_0\mid x_t]$, the MMSE denoiser. For Gaussian data this is checkable
in closed form, and the test suite checks it:

$$
x_0\sim\mathcal{N}(0,v_0)
\;\Longrightarrow\;
\mathbb{E}[x_0\mid x_t] = \frac{\alpha_t v_0}{\alpha_t^2v_0+\sigma_t^2}\,x_t
$$

and `denoise` reproduces it to floating point. That agreement is the single most useful
sanity check available on this whole family, because it is the only place a diffusion model
has a closed form to be checked against.

## 4. The weighting is where the objective's identity lives

The integrand $\|\varepsilon_\theta-\varepsilon\|^2$ is always the same. What $\omega(t)$
multiplies it by decides what you are computing:

| $\omega(t)$ | objective |
|---|---|
| $\sigma_t^2/\alpha_t^2 = 1/\mathrm{SNR}_t^2$ | the ELBO — an actual likelihood bound |
| $1$ | DDPM's "simple loss" — what people train |
| $\lambda\sigma_t/\alpha_t = \lambda/\mathrm{SNR}_t$ | **RED-Diff's regulariser** |

That the *trained* objective ($\omega=1$) is not the *likelihood* objective is well known and
is why diffusion models are good samplers and mediocre density estimators. That the
*inference-time* regulariser is a third weighting again is the subject of
[[RED-Diff as a Statistical Game]] §4 — and it turns out to determine the strength of the
implied prior.

## 5. The floor nobody documents

At $t=0$: $\sigma_t=0$, so the score $-\varepsilon_\theta/\sigma_t$ divides by zero and
RED-Diff's weight $\lambda\sigma_t/\alpha_t$ vanishes. Every implementation therefore samples
$t$ from $[t_{\min},1]$ with $t_{\min}\approx10^{-3}$, and almost none say so.

> [!warning] It is a modelling choice disguised as a numerical guard
> $t_{\min}$ appears as the lower limit of every integral in the package — including the λ
> calibration of [[RED-Diff as a Statistical Game]] §4, whose answer depends on it. Two
> implementations with different floors compute different posteriors, and neither is wrong.

A second numerical point, small but real: $\sigma_t$ must be computed as
$\sqrt{-\mathrm{expm1}(-B(t))}$, not $\sqrt{1-\alpha_t^2}$. The latter evaluates
$\sqrt{1-(1-\epsilon)^2}$ near $t=0$ and loses half its significant digits to cancellation.

## 6. What is not here

- **No sampler.** No ancestral sampling, no Euler–Maruyama, no probability-flow ODE. Which is
  consistent — RED-Diff never samples — but means the model cannot be checked by generating
  from it.
- **No VE or sub-VP**, though both are just another pair of $(\alpha_t,\sigma_t)$ formulas.
- **No training loop.** `lib/VariationalDiffusion.jl` consumes a trained
  $\varepsilon_\theta$; training it needs AD and this package deliberately has none
  ([[reddiff]] §2).
- **No per-element time.** `alpha(s,t)` takes a scalar; a real training batch draws a
  different $t$ per element. Fine for RED-Diff's inner loop, wrong for training.

Related: [[The Diffusion Family]], [[RED-Diff as a Statistical Game]], [[schedule]],
[[predictor]], [[Implicit Learners]]
