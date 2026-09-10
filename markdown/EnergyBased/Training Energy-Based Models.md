# Training Energy-Based Models

> Song & Kingma, *How to Train Your Energy-Based Models*
> ([arXiv:2101.03288](https://arxiv.org/abs/2101.03288)), and Du & Mordatch, *Implicit
> Generation and Modeling with Energy-Based Models*
> ([arXiv:1903.08689](https://arxiv.org/abs/1903.08689)).
>
> [[Energy-Based Learning]] §3 says Lenticulum has energies and no loss functionals. This note
> is the modern answer to what a loss functional should be — and the finding is that **two of
> the three standard answers are already implemented in this repository, under other names.**

## 1. The problem

An EBM defines $p_\theta(x) \propto e^{-E_\theta(x)}$ with $Z_\theta$ unknown. Maximum
likelihood needs

$$\nabla_\theta \log p_\theta(x) \;=\; -\nabla_\theta E_\theta(x) \;+\; \mathbb{E}_{x'\sim p_\theta}\bigl[\nabla_\theta E_\theta(x')\bigr]$$

— push down on the data, push **up** on the model's own samples. That second term is
[[Energy-Based Learning]] §3's contrastive term, made concrete: it is an expectation under the
model, and sampling from the model is the whole difficulty.

Song & Kingma organise the escape routes into three.

## 2. The three families, and where each already lives here

| family | avoids $Z$ by | already in this repo as |
|---|---|---|
| **MLE with MCMC** | sampling the model (Langevin, contrastive divergence) | — nothing |
| **Score matching** | matching $\nabla_x\log p$, which kills $Z$ | **`VariationalDiffusion.jl`** |
| **Noise-contrastive estimation** | classifying data against known noise | **`Adversarial.RatioFactor`** |

The middle two rows are the finding.

### 2.1 Score matching is the diffusion package

$\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$, and the gradient is taken in $x$, so
$Z_\theta$ — a constant in $x$ — differentiates away. That is why score matching needs no
normaliser and no sampling.

**Denoising score matching is the diffusion objective.** `VariationalDiffusion`'s
`predictor.md` §2 records the identity

$$s_\theta(x,t) \;=\; -\frac{\varepsilon_\theta(x,t)}{\sigma_t} \;\approx\; \nabla_x\log p_t(x)$$

which is exactly Song & Kingma's score-matching estimator with the noise scale as the
smoothing parameter. So the package built for [[The Diffusion Family]] is, without ever
saying so, **an energy-based model trained by score matching** — and `ImplicitREDDiff`'s
regulariser is that trained energy being reused at inference time.

The connection is not decorative. It explains a fact `reddiff.md` §4 found empirically and
could not account for: the calibration constant $\lambda^\star$, derived there so that
RED-Diff's implied prior matches a Gaussian's, is doing the job the missing normaliser would
have done. A score model knows $\nabla_x\log p$ and *not* $\log p$; the constant of
integration is exactly what $\lambda$ is standing in for, and that is why one scalar cannot
fit a correlated prior ([[RED-Diff as a Statistical Game]]).

### 2.2 NCE is the ratio factor — and it recovers the normaliser

Noise-contrastive estimation trains a classifier to separate data from a **known** noise
distribution $q$. `Adversarial`'s `ratio.md` §1 has the identity already:

$$\operatorname{logit} D^\ast(x) \;=\; \log p(x) - \log q(x)$$

In a GAN, $q$ is the generator and is unknown, so you get a ratio and stop. In NCE, **$q$ is
chosen and known**, so

$$\log p(x) \;=\; \operatorname{logit} D^\ast(x) \;+\; \log q(x)$$

and you have the *normalised* log-density, partition function included. That is the whole
trick, and it changes what `RatioFactor` is worth:

> [!important] `RatioFactor` with a known noise distribution is `belief_logdensity`
> [[messages]] §1 has recorded since the beginning that the package's main gap is `combine`
> for particle beliefs, which needs `belief_logdensity`, which nothing implements.
>
> [[Implicit Generative Models]] §5 got as far as: a *ratio* is enough for importance
> reweighting. NCE goes one step further — with $q$ known you get the *density*, which is what
> `messages.md` actually asked for, and it unlocks the generic path rather than a special-cased
> reweighting.
>
> The code change is small: `RatioFactor` already computes the logit; it needs a field
> holding $\log q$ and a method adding the two. What it does **not** have is any way to train
> the classifier, which is the real work.

## 3. Du & Mordatch: composition is the factor graph

Du & Mordatch scale EBM training with Langevin dynamics plus a replay buffer, and report three
capabilities. Two of them are things this project gets structurally rather than as a
technique.

**Compositionality.** Independently trained EBMs compose by *adding energies* —
$E_{\text{and}} = E_1 + E_2$ — giving concept conjunction with no retraining. That is
presented as a notable property of EBMs.

It is what a factor graph *is*. $E = \sum_c E_c$ is [[Energy-Based Factor Graphs]] §1, it is
`GradedEnergy`'s $\oplus$, and it is `Mycelium.combine` in the log domain. Lenticulum does not
have to discover compositionality; the whole architecture is that operation.

The difference is what happens next: Du & Mordatch compose energies and then **sample** by
Langevin; Lenticulum composes energies and then **passes messages**. Same composition, two
inference algorithms — and for Dirac-valued factors Lenticulum's is min-sum, which is the
$T\to0$ limit of the Gibbs distribution Langevin is sampling.

**Inpainting and corrupt-image reconstruction.** Clamp some coordinates, minimise over the
rest. That is `ImplicitREDDiff`'s selection matrices $P_{in} + P_{out} + P_{latent} = \mathrm{Id}$
and `DiffusionFactor`'s `precision_vector` — [[The Diffusion Factor]]. Again: a capability
there, a polarity here.

**Langevin as the missing inference mode.** The one thing Du & Mordatch have that this project
does not is *sampling* from a composed energy. RED-Diff's prox is gradient descent on an
energy — Langevin without the noise, i.e. MAP rather than a sample. Adding the noise term is
a two-line change to `reddiff_solve` and would turn a point estimate into a sample, which is
the difference between $T=0$ and $T=1$ in [[Energy-Based Factor Graphs]] §3.

## 4. What is actually missing

Putting [[Energy-Based Learning]] §3 together with this note, the gap is specific rather than
diffuse:

1. **No loss functional.** The free energy is evaluated at the inferred configuration; nothing
   raises the energy elsewhere. Every training method above exists to supply that second term,
   and the framework has no slot to put one in.
2. **No sampling from a composed energy.** MCMC is the one family of §2 with no counterpart
   here, and it is the one Du & Mordatch show scales.
3. **No training at all, in fact.** `VariationalDiffusion` consumes a trained
   $\varepsilon_\theta$; `Adversarial` consumes a trained discriminator. Both packages
   deliberately have no AD dependency. The energy-based reading says what the training
   objective *would* be; nothing computes it.
4. **The normaliser is nowhere.** `logpartition` exists for `GaussianBelief` and for nothing
   else, which is exactly the situation an EBM is designed to tolerate — but it means
   `scalar_free_energy` returning $-\log p(y)$ ([[The Linear Gaussian Chain]] §4) is a
   Gaussian-only guarantee, and no note has said so plainly.

## Sources

- Song & Kingma, *How to Train Your Energy-Based Models*,
  [arXiv:2101.03288](https://arxiv.org/abs/2101.03288) — MLE with MCMC, score matching, NCE.
- Du & Mordatch, *Implicit Generation and Modeling with Energy-Based Models*,
  [arXiv:1903.08689](https://arxiv.org/pdf/1903.08689) — Langevin training at scale, the
  replay buffer, compositionality, inpainting.
- Gutmann & Hyvärinen, *Noise-Contrastive Estimation*, AISTATS 2010 — §2.2's identity, and
  the reason a known $q$ recovers the normaliser.
- Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching*, JMLR 2005 —
  the original of §2.1.
- Vincent, *A Connection Between Score Matching and Denoising Autoencoders*, 2011 — why the
  diffusion objective is score matching.

Related: [[Energy-Based Learning]], [[Energy-Based Factor Graphs]],
[[Implicit Generative Models]], [[The Diffusion Family]], [[RED-Diff as a Statistical Game]],
[[The Diffusion Factor]], [[messages]], [[ratio]], [[reddiff]]
