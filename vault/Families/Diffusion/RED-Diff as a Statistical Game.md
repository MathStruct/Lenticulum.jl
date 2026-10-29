#model

> Implemented in `lib/VariationalDiffusion.jl/src/reddiff.jl`; see [[reddiff]].

> Sources: Mardani et al. 2023, [arXiv:2305.04391](https://arxiv.org/abs/2305.04391), read as AutoBayes Definition 20 — and the sharpest thing this vault has to say about the method: **its regularisation weight λ is derivable, not merely tunable, and one scalar λ cannot fit a correlated prior
>
> Bibliography: [[Bibliography#^mardani2024reddiff|Mardani et al. 2024]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## 1. Don't sample the reverse SDE — optimise

The standard way to condition a diffusion model on data $y$ is to run the reverse SDE with a
guidance term (DPS, ΠGDM). RED-Diff refuses: it posits a variational posterior
$q(z\mid z_0) = \mathcal{N}(\mu,\sigma^2I)$, minimises $\mathrm{KL}(q\,\|\,p(z\mid z_0))$, and
turns inference into an optimisation over $\mu$.

> [!note] Notation
> RED-Diff writes $x_0$ for the clean signal and $y$ for the measurement; [[ImplicitREDDiff]]
> writes $x$ for the state and $x_0$ for the clamp. This vault writes $z \in Z$ for the joint
> state and $z_0$ for the evidence (clamp values and anchors), as fixed in
> [[Channels and Polarity]] §"Notation".

Specialised to [[ImplicitREDDiff]]'s linear clamp, the objective is

$$
E(z; z_0) \;=\;
\underbrace{\mathbb{E}_{t,\varepsilon}\bigl[\omega(t)\|\varepsilon_\theta(\alpha_t z+\sigma_t\varepsilon,t)-\varepsilon\|^2\bigr]}_{\text{the learned prior}}
\;+\;
\underbrace{\tfrac12\|P(z-z_0)\|^2}_{\text{data consistency}}
$$

which is exactly the energy in the prompt and in [[ImplicitREDDiff]]. So the note's
sketch was right, and this is the formal reading of it.

## 2. The energy/entropy split, and it is not arbitrary

[[Factors are Parameterized Statistical Games]] (Definition 20) wants a factor to supply an energy $\mathbf{l}^c$ and an
entropy $\mathbf{H}^c$. The two terms above sort themselves:

| term | is | why |
|---|---|---|
| $\tfrac12\|P(z-z_0)\|^2$ | the **energy** $\mathbf{l}^c$ | pointwise; a function of the *data point* |
| $\mathbb{E}_{t,\varepsilon}[\omega\|\varepsilon_\theta-\varepsilon\|^2]$ | the **entropy** $\mathbf{H}^c$ | a function of the *learned distribution*, not of the datum |

This was the vault's first reading, and the code still splits the terms this way.
[[The Implicit Diffusion Factor as a Statistical Game]] §2 revises it: the score-matching term
is the **energy of the prior game** (a cross-entropy, by Remark 24's "priors are games too"),
and the entropy slot belongs to the inversion's output. The split itself is unchanged; only
its labels differ. `LenticulumCore.GradedEnergySpace` expresses it as
`(clamp = ..., score = ...)` with no new type.

> [!warning] But the "entropy" is not an entropy
> $q$ is a point mass, so $H(q) = -\infty$. The score term *stands in for* the entropy
> because it plays the entropy's structural role (it depends on the learned distribution and
> it regularises the inversion), not because it equals one. So a `DiffusionFactor` and a
> `GaussianFactor` in the same graph produce a Bethe total that is not $-\log p(y)$ for
> anything. See [[The Diffusion Factor]] §5.

> [!note] Superseded reading (2026-10-02)
> [[The Implicit Diffusion Factor as a Statistical Game]] §2 files the score-matching term as the
> **energy of a prior game** (Remark 24), not as the entropy: it is a cross-entropy
> $\mathbb E_q[-\log p_\theta]$, and the entropy slot holds $H(q)$. The table above is kept for the record.

## 3. Proposition 2: the stop-gradient is the method

$$
\nabla_z\,\mathrm{reg}(z)
=\mathbb{E}_{t,\varepsilon}\bigl[\lambda_t(\,\underbrace{\varepsilon_\theta(x_t,t)}_{\text{stop-grad}}-\varepsilon)\bigr],
\qquad
\lambda_t=\frac{\lambda}{\sqrt{\mathrm{SNR}_t}}=\frac{\lambda\sigma_t}{\alpha_t}
$$

Two separate things happen, and conflating them is easy:

- the **reparametrisation is kept** — $x_t = \alpha_t z + \sigma_t\varepsilon$ is
  differentiable in $x$, and its $\alpha_t$ is absorbed into $\lambda_t$;
- the **denoiser Jacobian is dropped** — the true gradient carries
  $(\partial\varepsilon_\theta/\partial x_t)^\top$ in front of the residual, and RED-Diff
  replaces it with the identity.

So RED-Diff is not an unbiased estimator of the regulariser's gradient; it is a
*preconditioned* one. In the vocabulary of [[abstract_types]]'s `AbstractGradientCoupling`,
the sampling path is `PathwiseCoupling` and the network Jacobian is stopped — which is **not
any of the four constructors**, and is worth recording as a fifth case rather than forced into
`DiagonalCoupling`. This is a comparison with planned semantics, not a graph annotation
used by RED-Diff: `PathwiseCoupling` and the other non-default annotations now fail at
`connect!`, `Edge` and `ComposedFactor` construction (LP-011 option 2). RED-Diff's own
closed-form regulariser is separate from [[L-SPEC-002 Stop Gradient]].

> [!important] This is why the implementation needs no AD
> One forward pass of $\varepsilon_\theta$ per Monte-Carlo draw; the clamp's gradient is
> $P^2(z-z_0)$ in closed form. `lib/VariationalDiffusion.jl` therefore depends on `LuxCore`,
> `Random` and `LinearAlgebra` — and not on `Lux`, `Zygote` or `Enzyme`. The stop-gradient is
> usually sold as a memory saving; here it is the difference between a package with an AD
> dependency and one without. Training *through* inference does need network derivatives; they
> come from a backend the user chooses, through a package extension ([[backends]]).

## 4. λ is derivable, and this is the finding

Both halves below are asserted in `lib/VariationalDiffusion.jl/test/runtests.jl`.

### 4.1 For isotropic Gaussian data there is exactly one correct λ

Gaussian data is the one case where $\varepsilon_\theta$ has a closed form
([[The VP-SDE]] §3), so the expectation can be done by hand. With
$z\sim p_0=\mathcal{N}(0,v_0I)$ and $D_t=\alpha_t^2v_0+\sigma_t^2$:

$$
\mathbb{E}_\varepsilon\bigl[\varepsilon_\theta(\alpha_tz+\sigma_t\varepsilon,t)-\varepsilon\bigr]
=\frac{\sigma_t\alpha_t}{D_t}z
\quad\Longrightarrow\quad
\nabla_z\mathrm{reg}(z)=\kappa z,
\quad
\kappa=\frac{\lambda}{1-t_{\min}}\!\int_{t_{\min}}^{1}\!\frac{\sigma_t^2}{D_t}dt
$$

**RED-Diff's regulariser *is* a Gaussian prior of precision $\kappa$.** The true prior
gradient is $z/v_0$, so the regulariser is correct iff $\kappa = 1/v_0$, which pins λ
uniquely. With that λ, the fixed point $\kappa z+\rho^2(z-z_0)=0$ gives

$$z^\star=\frac{\rho^2z_0}{1/v_0+\rho^2}$$

which is **exactly** the Gaussian posterior mean. For the default schedule and $v_0=1$,
$\lambda^\star\approx1.379826$ — while the paper's tuned value is $0.25$.

> Tuning λ is not adjusting "how much regularisation feels right". It is **choosing how
> strong the learned prior is.** A wrong λ biases every posterior by a computable factor.

### 4.2 For correlated data no single λ works

Take $p_0=\mathcal{N}(0,\Sigma)$. Both $\kappa$ and $\Sigma^{-1}$ are functions of $\Sigma$,
hence simultaneously diagonalisable, so per eigenvalue $s$ the requirement is
$\lambda\,s\,I(s)=1$ where $I(s)=\frac{1}{1-t_{\min}}\int_{t_{\min}}^1\sigma_t^2/(\alpha_t^2s+\sigma_t^2)dt$. That needs
$s\,I(s)$ constant in $s$. It is not:

| $s$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| $s\,I(s)$ | 0.206878 | 0.389614 | 0.724729 | 1.331492 | 2.417511 |

A factor of ~12 across a 16× spread. Calibrating at $s=1$ therefore

- **over**-regularises high-variance directions ($s=4$: about $3.3\times$ too strong),
- **under**-regularises low-variance ones ($s=0.25$: about $0.29\times$).

Suppressing exactly the directions that carry the most signal variance is an
**over-smoothing bias** — and over-smoothed, mode-seeking reconstructions are precisely what
RED-Diff is criticised for empirically. The theory predicts the known artefact, from a
five-line calculation, on the one data family where it can be checked.

## 5. The field weight and the likelihood bound

RED-Diff's $\lambda_t=\lambda\sigma_t/\alpha_t=\lambda/\sqrt{\mathrm{SNR}_t}$
grows steeply with noise, but is finite at $t=1$ for the default VP schedule:
$\alpha_1=e^{-5.025}>0$ and $\lambda_1=\lambda\sqrt{e^{10.05}-1}$.
Divergence requires a schedule whose terminal signal vanishes.

With the power convention $\mathrm{SNR}_t=\alpha_t^2/\sigma_t^2$, the continuous
ELBO squared-noise-error coefficient under $dt$ is
$-\tfrac12(d/dt)\log\mathrm{SNR}_t=\beta(t)/(2\sigma_t^2)$ ([[schedule]] §3).
Under $t\sim U[t_{\min},1]$ its coefficient includes $1-t_{\min}$.
RED-Diff instead weights a stop-gradient residual field under that uniform distribution;
it drops the denoiser Jacobian. Comparing these two quantities by squaring weights does
not establish a likelihood bound or explain mode seeking. Its point variational family and
tuned surrogate target reconstruction rather than posterior calibration.
The Gaussian calibration below measures the resulting prior strength directly.

## 6. What this is not

- **Not a posterior.** $q$ is a point mass; there are no error bars. The general
  $\sigma>0$ case is derived in the paper's §3 and dropped in its experiments, and dropped
  here too — it is the most valuable missing piece ([[The Diffusion Factor]] §5.2).
- **Not exact at its own fixed point** — for the regulariser it starts from. For an exact score its field *is* the exact gradient of a smoothed log-density ([[Implicit Diffusion Learners]] §3), so it is exact for a different, computable prior.
- *(original wording)* **Not exact at its own fixed point.** Unlike the algebraic and equilibrium families, which
  are approximate because they *stop early*, this one converges to the wrong point by
  construction. [[Inversions and Bayesian Lenses]] permits inexact inversions and the free energy is supposed
  to measure the cost — but here the cost is a Jacobian nobody computes, so it is not
  measured either.
- **Not trainable here** — by RED-Diff itself. Training *through* inference is now possible by the adjoint: [[Backpropagation through Implicit Inference]].

Related: [[The Diffusion Family]], [[The VP-SDE]], [[The Diffusion Factor]],
[[ProxDM and Proximal Alternatives]], [[Factors are Parameterized Statistical Games]], [[Implicit Learners]],
[[ImplicitREDDiff]], [[Bethe Free Energy]]

## LP-018 calibration measurements (2026-10-29)

Time is uniform on `[tmin,1]`; both the shrinkage and `I(s)` above include
`1/(1-tmin)`. `examples/lp018_calibration.jl` reproduces the table (independent
20,000-point midpoint quadrature), the default calibrated value `1.379825576009811`
(previously `1.3812067827926036`), and the six-seed Gaussian posterior measurements.
At clamp precisions 1 and 2, the maximum relative errors are 2.1833% and 1.4030%;
the errors of the six-seed means are 0.9147% and 0.5028%. Settings match the existing
Gaussian posterior regression: 1,500 steps, 64 samples, learning rate 0.1, no Adam.
The corresponding terminal weight at unit lambda is `152.16697028394637`.
