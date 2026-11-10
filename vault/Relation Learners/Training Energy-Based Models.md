#annotation #definition

> [[Energy-Based Learning]] §3 says Lenticulum has energies and no loss functionals. This note
> is the modern answer to what a loss functional should be: **two of the three standard answers
> are implemented in this repository, under other names**, and the third (maximum likelihood with
> MCMC) is the one Lenticulum never needed for training. It then lists the newer options that
> avoid MCMC, and which of them would add something here (§4), notes how score matching relates to
> the Ritz method for PDEs (§5), and says what is still missing (§6).

> **Updated 2026-11-05:** §2.1 now records `EnergyNetwork` (an energy-parametrised score,
> implemented since this note was first written); §3's "Langevin is missing" is out of date
> (`implicit_sample`'s correctors and SMC); §4 and §5 are new; §6 replaces the old list of gaps,
> three of which had been closed.

> Sources: Song & Kingma, *How to Train Your Energy-Based Models* ([arXiv:2101.03288](https://arxiv.org/abs/2101.03288)); Du & Mordatch, *Implicit Generation and Modeling with Energy-Based Models* ([arXiv:1903.08689](https://arxiv.org/abs/1903.08689)); Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching*, JMLR 6, 2005; Gutmann & Hyvärinen, *Noise-Contrastive Estimation*, AISTATS 2010; Vincent, Neural Computation 23(7), 2011; Song, Garg, Shi & Ermon, *Sliced Score Matching*, UAI 2019; Salimans & Ho, *Should EBMs model the energy or the score?*, ICLR 2021 EBM workshop; Gao et al., *Learning EBMs by Diffusion Recovery Likelihood*, 2020; Zhu, Xie, Wu & Gao, *Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood*, 2023; Gao et al., *Flow Contrastive Estimation*, 2019; Rhodes, Xu & Gutmann, *Telescoping Density-Ratio Estimation*, 2020; Schröder et al., *Energy Discrepancies*, NeurIPS 2023; Du et al., *Reduce, Reuse, Recycle*, ICML 2023; Balcerak et al., *Energy Matching*, NeurIPS 2025; Wang & Du, *Equilibrium Matching*, 2025; Gladstone et al., *Energy-Based Transformers*, 2025; Xu et al., *Energy-Based Diffusion Language Models*, ICLR 2025; Meng, Choi, Song & Ermon, *Concrete Score Matching*, NeurIPS 2022; E & Yu, *The Deep Ritz Method*, Commun. Math. Stat. 6(1), 2018. §4 summarises the papers' own claims from their abstracts; nothing there is measured in this repository. code: `lib/VariationalDiffusion.jl/src/predictor.jl` (`denoising_loss`), `energy_network.jl`, `sampler.jl`; `parked/Adversarial.jl/src/ratio.jl`
>
> Bibliography: [[Bibliography#^song2021train|Song & Kingma 2021]] · [[Bibliography#^du2019implicit|Du & Mordatch 2019]] · [[Bibliography#^hyvarinen2005score|Hyvärinen 2005]] · [[Bibliography#^gutmann2010nce|Gutmann & Hyvärinen 2010]] · [[Bibliography#^vincent2011dsm|Vincent 2011]] · [[Bibliography#^song2019sliced|Song et al. 2019]] · [[Bibliography#^salimans2021ebm|Salimans & Ho 2021]] · [[Bibliography#^gao2021recovery|Gao et al. 2020]] · [[Bibliography#^zhu2023cooperative|Zhu et al. 2023]] · [[Bibliography#^gao2019flow|Gao et al. 2019]] · [[Bibliography#^rhodes2020telescoping|Rhodes et al. 2020]] · [[Bibliography#^schroeder2023discrepancy|Schröder et al. 2023]] · [[Bibliography#^du2023reduce|Du et al. 2023]] · [[Bibliography#^balcerak2025matching|Balcerak et al. 2025]] · [[Bibliography#^wang2025equilibrium|Wang & Du 2025]] · [[Bibliography#^gladstone2025ebt|Gladstone et al. 2025]] · [[Bibliography#^xu2025edlm|Xu et al. 2025]] · [[Bibliography#^meng2022concrete|Meng et al. 2022]] · [[Bibliography#^e2018ritz|E & Yu 2018]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

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
| **MLE with MCMC** | sampling the model (Langevin, contrastive divergence) | not for training; MCMC is used for *sampling* (`implicit_sample`'s Langevin and MALA correctors, SMC) |
| **Score matching** | matching $\nabla_x\log p$, which kills $Z$ | **`VariationalDiffusion.jl`** (`denoising_loss`), with an energy-parametrised score (`EnergyNetwork`) |
| **Noise-contrastive estimation** | classifying data against known noise | **`Adversarial.RatioFactor`** |

The middle two rows are the finding. The MCMC row's cost, a chain per gradient step, is the one
Lenticulum never pays in training.

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

**The energy-parametrised score** (Salimans & Ho 2021; Du et al. 2023) closes most of that gap,
and it is implemented: `EnergyNetwork` outputs a scalar $E_\theta(x, t)$ and the noise prediction is
$\sigma_t \nabla_x E_\theta$ ([[energy_network]]). It trains by the same denoising loss, with no MCMC;
the price is a second derivative, about three times the time per step on the circle (41 s
against 15 s, [[energy_network]] §6). In return the field is exactly a gradient, the implicit
learner has an energy $U(z)$, and two answers to one query can be compared. What it still lacks is
the **absolute level**: $E_\theta(\cdot, t)$ is $-\log p_t$ only up to a constant in $x$, which may
differ per noise level and per model (§6).

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
> `RatioFactor` now emits a `LogPotentialBelief` for $\log r(x)$, and Mycelium can
> pool it with explicit particles by importance weighting ([[ratio]], LP-016).
> With known $\log q(x)$, adding it to the log ratio gives a density potential for
> $p$ rather than the ratio potential $p/q$. These are different messages: their
> products with a cavity have different meanings.
>
> Two particle clouds still have no product rule without a density or potential.
> RatioFactor supplies scoring and explicit posterior reweighting; it does not train
> the classifier. The identity above requires the optimal equal-prior discriminator.

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

**Langevin as an inference mode.** When this note was first written, sampling from a composed
energy was the one thing Du & Mordatch had that this project did not. It exists now:
`implicit_sample` samples a relation or a product of relations by the reverse diffusion, with
Langevin or MALA corrector steps at every noise level and optionally SMC weights
([[sampler]], [[product]] §5); a product needs the correctors, because the sum of the
factors' scores is not the score of the noised product. RED-Diff's prox is still gradient
descent on an energy, MAP rather than a sample: the difference between $T=0$ and $T=1$ in
[[Energy-Based Factor Graphs]] §3.

## 4. Newer options without MCMC, and which would add something here

Since Song & Kingma's survey the field has moved in three directions: MCMC made cheap
(conditional or warm-started chains), contrastive losses with better negatives, and energies
learned with diffusion or flow objectives. The table gives each method's training signal as its
paper states it, and what it would give Lenticulum.

| method | trains by | MCMC in training | what it would add here |
|---|---|---|---|
| **Sliced score matching** (Song et al. 2019) | Hyvärinen's objective with the Hessian trace replaced by random projections | no | score matching on the *clean* data, without a noise level: for energy factors that are not diffusion models |
| **Energy discrepancy** (Schröder et al. 2023) | compares the data's energy with a log-mean-exp of energies at points drawn around a perturbation of the data; approaches score matching for small perturbations and maximum likelihood for large ones (the paper's result) | no, and no score | **a loss for a general energy factor** (§6.1): only energy evaluations, so it applies where no gradient in $x$ is available and to discrete data |
| **Diffusion recovery likelihood** (Gao et al. 2020); cooperative version (Zhu et al. 2023) | an EBM per noise level, trained on the conditional $p(x_t \mid x_{t+\delta})$, which is close to Gaussian; the cooperative version starts the chains from a learned initialiser | a few steps | an absolute energy per level from a short chain; a fallback where denoising score matching is not enough |
| **Flow contrastive estimation** (Gao et al. 2019); **telescoping density ratios** (Rhodes et al. 2020) | NCE with a flow as the noise, trained jointly; or a chain of ratios through intermediate distributions between data and noise | no | the fix for NCE when data and noise are too easy to tell apart, which is the usual failure of `RatioFactor` with a fixed $q$ |
| **Energy Matching** (Balcerak et al. 2025) | one *time-independent* scalar potential: far from the data its gradient carries samples along optimal-transport paths, near the data an entropic term makes it a Boltzmann density; the abstract claims simulation-free training away from the data manifold (how the near-data phase is trained is not checked here) | partly, as we read it | **one energy for the implicit learner instead of one per level**, usable as a prior for inverse problems (the paper's use, and the clamp-and-solve query here) |
| **Equilibrium Matching** (Wang & Du 2025) | the gradient of an implicit, time-independent energy landscape, in place of a time-conditioned velocity; samples by gradient descent with adjustable steps | no | sampling *is* optimisation: Lenticulum's root finding and proximal solves apply directly, and one field replaces the sum over levels of `ImplicitDiffusion` |
| **Energy-Based Transformers** (Gladstone et al. 2025) | an energy over (input, candidate prediction); a prediction is gradient descent on the energy until convergence | no | the implicit learner's query at scale, including text: evidence that inference-as-minimisation scales; relevant to LenticulumLM |
| **Energy-based diffusion language model** (Xu et al. 2025) | a residual, sequence-level energy per diffusion step, from a pretrained autoregressive model or by NCE fine-tuning; sampling by parallel importance sampling | no | a sequence-level factor that corrects the per-position independence of parallel decoding (LenticulumLM) |
| **Concrete score matching** (Meng et al. 2022) | ratios $p(y)/p(x)$ over neighbouring discrete states, the discrete analogue of the score | no | the training loss for a discrete implicit learner (LenticulumLM's open choice) |

Two of these are the most promising here. **Energy discrepancy** gives a loss to the energy
factors that today have only collapse warnings. **Energy and Equilibrium Matching** would replace
the implicit learner's sum over noise levels by one time-independent field, whose roots are the
relation and whose descent is the sampler. Both are listed in [[Feature Backlog]].

## 5. Score matching is a Ritz method

The Ritz method solves a PDE by minimising its energy functional over a family of functions.
The Deep Ritz method (E & Yu 2018) takes the family to be a network and estimates the integral by
Monte Carlo over sample points: for $-\Delta u = f$,

$$
I[u] = \int \Bigl(\tfrac12 \lVert\nabla u\rVert^2 - f u\Bigr)\, dx .
$$

Hyvärinen's score matching has the same shape. The Fisher divergence
$\tfrac12\, \mathbb E_p \lVert \nabla_x E_\theta + \nabla_x \log p \rVert^2$ contains the unknown $\nabla_x \log p$; integration
by parts moves the derivative onto $E_\theta$ and leaves, up to a constant,

$$
J(\theta) = \mathbb E_{x \sim p}\Bigl[\tfrac12 \lVert\nabla_x E_\theta(x)\rVert^2 - \Delta_x E_\theta(x)\Bigr] .
$$

In both, a functional quadratic in a gradient is estimated by sampling, after the integration by
parts that a weak (variational) form needs; the data samples play the role of the quadrature
points. This is an analogy of form, not a theorem of this vault. It explains the cost:
$\Delta_x E_\theta$, a Hessian trace, is the expensive term, and sliced score matching (random
projections) and denoising score matching (a noise level instead of the trace) are the two
standard ways to avoid it. The diffusion objective used here is the second.

## 6. What is still missing

1. **A loss for a general energy factor.** Constraint and Gaussian factors are normalised in
   their data channels, diffusion factors train by the denoising loss, and every other learned
   energy (toric factors with `normalised = false`, a DEQ with its state observed) is at least
   warned about ([[Why Learned Relations Do Not Collapse]]). Nothing trains such a factor
   *correctly*: that needs a contrastive term. Energy discrepancy (§4) or NCE with a good noise
   would be that term.
2. **The absolute level of a learned energy.** `logpartition` still exists only for
   `GaussianBelief`, and an `EnergyNetwork`'s energy is $-\log p_t$ up to a constant per level and
   per model. That is enough for gradients, roots, MALA and comparing two answers to one query.
   It is not enough to compare models, structures or noise levels by their energies, or to make
   `scalar_free_energy` a log-evidence beyond the Gaussian case. NCE against a known noise, or a
   ratio estimate through intermediate distributions, would supply the constant.
3. **A time-independent energy.** The implicit learner sums over noise levels; one level-free
   field (§4, Energy and Equilibrium Matching) is untried.

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
- Salimans & Ho 2021; Du et al. 2023 — the energy-parametrised score, §2.1.
- Song et al. 2019; Schröder et al. 2023; Gao et al. 2020; Zhu et al. 2023; Gao et al. 2019;
  Rhodes et al. 2020; Balcerak et al. 2025; Wang & Du 2025; Gladstone et al. 2025; Xu et al. 2025;
  Meng et al. 2022 — the newer options, §4.
- E & Yu 2018 — the Deep Ritz method, §5.

Related: [[Energy-Based Learning]], [[Energy-Based Factor Graphs]],
[[Implicit Generative Models]], [[The Diffusion Family]], [[RED-Diff as a Statistical Game]],
[[The Diffusion Factor]], [[messages]], [[ratio]], [[reddiff]], [[energy_network]], [[sampler]],
[[The Regulariser and the Loss]]
