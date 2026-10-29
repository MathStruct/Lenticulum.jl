#implementation

> Samples of a query, not only its modes: the reverse diffusion with the hard-clamped
> coordinates replaced at every step. Root finding returns the most typical answer of each
> basin; `implicit_sample` returns draws from $p(z_F \mid z_C)$, and so an independent check of
> the branch weights that `implicit_mixture` estimates from Laplace masses.

> Sources: code: `sampler.jl`; Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 (the ancestral step); Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021 (imputation by replacement); Lugmayr et al., *RePaint*, CVPR 2022 (resampling); Kingma et al., *Variational Diffusion Models*, NeurIPS 2021 (log-SNR)
>
> Bibliography: [[Bibliography#^ho2020ddpm|Ho et al. 2020]] · [[Bibliography#^song2021sde|Song et al. 2021]] · [[Bibliography#^lugmayr2022repaint|Lugmayr et al. 2022]] · [[Bibliography#^kingma2021vdm|Kingma et al. 2021]] · [[Bibliography#^ho2022cfg|Ho & Salimans 2022]]
>
> Theory: [[Implicit Diffusion Learners]] · [[Inference Signatures]] · [[Mixture Belief]]

## 1. The sampler

From $x_T \sim \mathcal N(0, I)$, each step $t \to u < t$ draws from the Gaussian
$q(x_u \mid x_t, \hat x_0)$ with Tweedie's $\hat x_0 = (x_t - \sigma_t\hat\varepsilon)/\alpha_t$ (DDPM's
ancestral step, written for the VP-SDE's $\alpha_t, \sigma_t$), then overwrites the clamped
coordinates by $\alpha_u z_{0,C} + \sigma_u\xi$, the clamp's own forward marginal. The last step
returns $\hat x_0$ with the clamps set exactly. With `resample = r` each step is followed by
re-noising $u \to t$ and repeating, $r$ times (RePaint), so that the free coordinates catch up
with the clamp.

`ρ` uses the polarity encoding of the solvers: `Inf` is clamped, `0` is free. Soft evidence
(finite positive `ρ`) is refused: sampling a posterior under a Gaussian likelihood needs
guidance (DPS and relatives), which is approximate, and a silent approximation here would
undermine the sampler's use as a check.

## 2. Design decisions

- **Uses the predictor, not the nodes.** The field's fixed nodes are a quadrature for root
  finding; a sampler needs the whole noise range.
- **A grid uniform in log-SNR.** A grid uniform in $t$ spends almost no steps at low noise, where
  the density has its fine structure: with 100 steps on the unit-circle mixture (data width
  0.05) the samples' radial spread was 0.034; with the log-SNR grid it is 0.047.
  The implementation uses $\log(\alpha_t/\sigma_t)$, half the power log-SNR of
  [[schedule]] §3. A constant factor leaves the uniform grid unchanged (LP-018.10).
- **Batched like the field.** With `m.batched` all chains go through the network in one call per
  step; otherwise one call per chain. Both give identical draws for the same `rng`.

## 3. Measured on the circle

The mixture of 128 components on the unit circle (width 0.05), upper half three times heavier:

| | upper branch | $\lvert y\rvert$ |
|---|---|---|
| exact, given $x = 0.3$ | 0.75 | $0.952 \pm 0.052$ |
| `implicit_sample`, 100 steps, no resampling | 0.74 | $0.927 \pm 0.065$ |
| `implicit_sample`, 100 steps, `resample = 3` | 0.75 | $0.951 \pm 0.048$ |
| `implicit_mixture` (Laplace weights) | 0.78 | modes at $\pm 0.954$ |

Plain replacement gets the branch frequencies right but pulls the samples towards the centre: the
free coordinate's early history does not know where the clamp is. Three resamplings fix it, at
four times the cost. Unconditionally (nothing clamped) the samples have radius
$0.999 \pm 0.047$, against $1.001 \pm 0.050$ for the data.

## 4. Corrector steps and reweighting

`corrector = :langevin | :mala` adds Markov chain steps at every noise level, and `smc = true`
weights and resamples the chains as a sequential Monte Carlo sampler over the levels. For a
single model the reverse process is already consistent, and these options only refine it.
They exist for products of relations, whose summed scores are not the scores of the noised
product ([[product]] §5). With `smc` the clamped coordinates follow $\alpha_u z_{0,C}$ without noise;
on the circle above, clamped at $x = 0.3$, it gives the upper branch 0.75 without RePaint's
resampling (a test).

## 5. Held clamps and guidance

For a predictor with noise blocks ([[predictor]] §6), `clamp_level = tmin` holds every fully
clamped block at that level for the whole reverse diffusion, instead of replacing it by its
forward marginal. The network sees the clamp clean, as in training with independent block
times, and the free coordinates follow the conditional score. Blocks that are only partly
clamped (half an image) are still replaced. `guidance = w` mixes in the unconditional
prediction, $(1 + w)\,\hat\varepsilon(x, z_C) - w\,\hat\varepsilon(x, \varnothing)$, with the held blocks at level 1
where they are noise (classifier-free guidance, Ho & Salimans 2022).

On two components at $(\pm1, \pm1)$ of width 0.1, $x$ clamped at 1, 2000 draws (exact
conditional $\mathcal N(1, 0.1^2)$):

| | mean of $y$ | std of $y$ |
|---|---|---|
| replacement | 0.981 | 0.142 |
| held clean | 1.005 | 0.098 |
| held, guidance 2 | 1.010 | 0.094 |

Replacement lags behind the clamp and comes out too wide; holding it gives the conditional. For
an exact model guidance only sharpens; it is meant for a learned one whose conditional is too
weak (MNIST, [[MNIST in Both Directions]] §6). Held clamps do not combine with correctors or
SMC.

Related: [[implicit]], [[product]], [[proxdm]], [[MNIST in Both Directions]]
