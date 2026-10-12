#application #later

> **One implicit diffusion model of (image, label) on MNIST, queried in both directions:**
> clamp the image and the free label is a classification; clamp the label and the free image is
> a digit; clamp half an image and both the rest and the label come out at once. The demo of
> "one relation, polarity chosen at query time" that everyone recognises. Not to be built yet:
> §4 lists the interfaces that have to exist first, most of them already items in
> [[Open Problems in Implicit Diffusion Learning]] Part II.

> Sources: original to this vault (experiment design); LeCun, Bottou, Bengio & Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proc. IEEE 1998 (MNIST); Grathwohl et al., *Your Classifier is Secretly an Energy Based Model…*, ICLR 2020 (JEM); Li et al., *Your Diffusion Model is Secretly a Zero-Shot Classifier*, ICCV 2023; Clark & Jaini, *Text-to-Image Diffusion Models are Zero-Shot Classifiers*, NeurIPS 2023; Bao et al., *One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale*, ICML 2023 (UniDiffuser); Lugmayr et al., *RePaint*, CVPR 2022; code: `implicit.jl`, `answers.jl`, `energy_network.jl`
>
> Bibliography: [[Bibliography#^lecun1998mnist|LeCun et al. 1998]] · [[Bibliography#^grathwohl2020jem|Grathwohl et al. 2020]] · [[Bibliography#^li2023diffclassifier|Li et al. 2023]] · [[Bibliography#^clark2023zeroshot|Clark & Jaini 2023]] · [[Bibliography#^bao2023unidiffuser|Bao et al. 2023]] · [[Bibliography#^lugmayr2022repaint|Lugmayr et al. 2022]]
>
> Theory: [[Implicit Diffusion Learners]] · [[Inference Signatures]] · [[Mixture Belief]] · [[Backward Signatures]]

## 1. The setup

The joint state is $z = (x, y)$ with the image $x \in \mathbb R^{784}$ and the label
$y \in \mathbb R^{10}$, a one-hot vector (possibly scaled, so that the label is not drowned by
784 pixel coordinates in the denoising loss). Train ordinary denoising on the concatenation, with
an `EnergyNetwork`, so that answers get weights (§3). The direction is chosen only at query time,
by the precisions $\rho$:

| query | clamped ($\rho = \infty$) | free | computed by |
|---|---|---|---|
| classify | $x$ | $y$ | `implicit_roots` / `implicit_mixture` on 10 coordinates |
| the most typical "3" | $y$ | $x$ | `implicit_infer`: a mode, §2 |
| draw a "3" | $y$ | $x$ | a clamped sampler (signature #6 of [[Inference Signatures]]): missing, §4 |
| inpaint and classify | half of $x$ | the other half, and $y$ | `implicit_infer` |

The last row is the one no ordinary classifier or generator does, and it costs nothing extra.

## 2. What each direction returns

**Classification** is cheap: a 10-dimensional solve with a 10×10 Jacobian block. The answers
are roots near the one-hot vectors; with an energy, `implicit_mixture`'s Laplace-mass weights
$w_k \propto e^{-E(z_k)}\det(\Sigma_k)^{1/2}$ are class probabilities, and their calibration is a
measurable result (reliability diagram against a softmax classifier). An ambiguous digit should
give two roots of comparable weight, which is the point of returning a mixture rather than an
argmax.

**Generation by root finding** returns a **mode** of the smoothed $p(x \mid y)$: one
deterministic prototype per basin, not a sample. Whether the modes look like digits or like
blurry class averages depends on the smoothing scale (the field's noise levels,
[[Implicit Diffusion Learners]] §5), and is an experiment of its own. Multi-start from noise finds
several modes (styles of "7"), but with no frequencies attached unless the weights are computed
in 784 dimensions.

**Sampling** needs the reverse SDE with the label coordinates held fixed, the replacement trick
of RePaint: the same network, no retraining, but the sampler path, not the implicit solver.

## 3. How this differs from the prior work

| | how it classifies | how it generates |
|---|---|---|
| JEM (Grathwohl et al. 2020) | the logits of a classifier read as an energy of $(x, y)$ | SGLD on that energy |
| diffusion classifiers (Li et al. 2023; Clark & Jaini 2023) | Bayes' rule over per-class denoising errors: **one Monte-Carlo pass per class** | the diffusion model itself |
| UniDiffuser (Bao et al. 2023) | one noise level per modality, every conditional from one network | the same |
| **this** | **root finding in label space**, with Laplace weights as probabilities | modes by root finding; samples by a clamped reverse SDE |

The claim worth testing is the cost line: a 10-dimensional solve against ten Monte-Carlo
estimates of a denoising loss. The accuracy will most likely sit below a plain classifier's
~99%. The selling point is one model for three queries, with uncertainty, not the benchmark.

## 4. Interfaces that have to exist first

1. **Partial Jacobians** — *done.* `prior_jacobian(…; coords = free)` computes only $J_{FF}$, and
   the pullback gets $J_{FC}^\top\lambda$ from one VJP (`prior_vjp`). On a 794-dimensional MLP with 10
   free coordinates a solver iteration went from about 26 s to 0.04 s ([[implicit]] §6).
2. **Batching and matrix-free solves** ([[Open Problems in Implicit Diffusion Learning]] I5): one
   network call per noise level (*done*, `batched = true`); many queries at once (10,000 test
   images); Newton–Krylov, and a Lanczos estimate of stability and of $\log\det \Sigma$ when the image is free.
3. **A clamped sampler for ε- and energy networks** (I1): `implicit_sample(m, z₀, ρ, ps, st; rng,
   steps)`, the reverse SDE with hard-clamped coordinates replaced at every step. Today only
   `proxdm_sample` exists, and it is unconditional.
4. **A categorical answer.** `implicit_mixture` returns a mixture of Gaussians over $\mathbb
   R^{10}$; a classification wants a `CategoricalBelief`. Needed: a projection that assigns each
   root to its nearest one-hot vector and sums the weights, with the residual mass (roots far from
   every class) reported, not dropped.
5. **Energy networks at this size, on a GPU backend** (I6): second derivatives of a 794-input
   network, which Zygote and ForwardDiff-over-Zygote handle and Enzyme does not yet.
6. **Optional: the backward signatures** ([[Backward Signatures]]), only if the generative model
   is fine-tuned discriminatively through the classification query. Pure denoising training
   needs none of it.

The data loader (MLDatasets.jl) belongs in the tutorial's environment, never in a package's
dependencies.

## 5. What the tutorial would show

Classification accuracy and calibration; the class prototypes (modes) beside samples; an
ambiguous digit with a two-component answer; inpainting of the lower half of a digit together with
its label; and the cost per query against a diffusion classifier with ten denoising passes.

Related: [[Implicit Diffusion Learners]], [[Inference Signatures]], [[Composing Diffusion Factors]],
[[Open Problems in Implicit Diffusion Learning]], [[Language Models]], [[Motivating Examples]]
