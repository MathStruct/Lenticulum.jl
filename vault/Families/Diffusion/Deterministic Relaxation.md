#derivation #design

> The implicit diffusion learner with its randomness removed. The relaxations are, in order:
> fixing the noise (deterministic, still a smoothed MAP), using one noise level without noise
> (a **fixed point of the Tweedie denoiser**, a deep equilibrium model whose layer is the
> denoiser), and the zero-noise limit (modes of the data density, but an ill-conditioned field).
> The DEQ correspondence is exact and is checked to $10^{-10}$.

> Sources: original to this vault (design and analysis); Romano, Elad & Milanfar, *The Little Engine that Could: Regularization by Denoising (RED)*, SIAM J. Imaging Sci. 10(4) (2017); Bai, Kolter & Koltun, *Deep Equilibrium Models*, NeurIPS 2019; Efron, *Tweedie's Formula and Selection Bias*, JASA 2011; Song et al. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) (the probability-flow ODE); code: `implicit.jl` (`noisefree_nodes`, `field_nodes`)
>
> Theory (CT-ML wiki): [Least Fixed Point](https://mathstruct.org/CategoryTheory-ML-Wiki/Least-Fixed-Point) · [Initial Algebra](https://mathstruct.org/CategoryTheory-ML-Wiki/Initial-Algebra) · [Contextual Equivalence](https://mathstruct.org/CategoryTheory-ML-Wiki/Contextual-Equivalence)

## 1. Where the randomness is, and what each relaxation removes

| level | what is random | after relaxation | output |
|---|---|---|---|
| RED-Diff as published | $(t, \varepsilon)$ redrawn every step | — | a stochastic approximation of a smoothed MAP |
| **fixed nodes** (sample-average approximation) | nothing | the expectation replaced by a fixed quadrature | a deterministic smoothed MAP |
| **one level, no noise** | nothing | one $t$, $\varepsilon = 0$ | a fixed point of the denoiser at that level |
| zero-noise limit $t \to 0$ | nothing | the data density itself | modes of $p_0$ — sharp, but stiff |
| probability-flow ODE | the initial noise only | a deterministic map from noise to sample | a sample, determined by its seed |

The probability-flow ODE is deterministic *given* its initial noise, so it is a deterministic
sampler rather than a deterministic relation. The rest are relations.

## 2. Fixed nodes: the deterministic smoothed MAP

Replacing $\mathbb E_{t,\varepsilon}$ by a fixed set of nodes gives a deterministic field whose roots
are stationary points of
$\Phi_K(z) + \tfrac12\lVert P(z-z_0)\rVert^2$, with
$\Phi_K(z) = -\sum_k w_k\lambda_{t_k}\tfrac{\sigma_{t_k}}{\alpha_{t_k}}\log p_{t_k}(\alpha_k z + \sigma_k\varepsilon_k)$
([[Implicit Diffusion Learners]] §3). Two refinements are worth having:

- **Antithetic pairs** $\pm\varepsilon_k$ cancel the control-variate tilt $-\sum w_k\lambda_k\varepsilon_k$ *exactly*, so $\Phi_K$ has no spurious linear term.
- Few nodes break symmetries of the true expectation: on the circle the $y \mapsto -y$ symmetry is lost and a merged branch sits slightly off the axis. More nodes shrink this.

This is the relaxation the package uses by default: it is cheap, it keeps the smoothing over
several scales, and it makes inference a deterministic function, so the
[[Backpropagation through Implicit Inference|adjoint]] is exact.

## 3. One level, no noise: a DEQ whose layer is the denoiser

Take a single node, $t$ fixed and $\varepsilon = 0$: $g(z) = \lambda_t\,\varepsilon_\theta(\alpha_t z, t)$. Tweedie's
denoiser at level $t$ is $\hat x_0(x) = (x - \sigma_t\varepsilon_\theta(x, t))/\alpha_t$, so

$$
\varepsilon_\theta(\alpha_t z, t) = 0 \quad\Longleftrightarrow\quad \hat x_0(\alpha_t z) = z .
$$

With hard inputs and free outputs ($\rho_F = 0$) the relation is the **fixed-point set of the
denoiser on the free coordinates**:

$$
z^\star_F = \bigl[\hat x_0(\alpha_t z^\star)\bigr]_F,\qquad z^\star_C = x .
$$

That is a deep equilibrium model in the sense of [[DEQ as a Relation]]: the layer is
$T(z) = (\,\hat x_0(\alpha_t z)_F,\; x_C\,)$, the input enters by clamping, and the backward pass is
the same IFT, with $J_{FF} = \lambda_t\alpha_t\,\partial_x\varepsilon_\theta$ (equivalently
$I - \partial_z T$ up to scale). Checked: for every root found, $\lvert\hat x_0(\alpha_t z^\star)_y - z^\star_y\rvert \le 10^{-9}$
(test suite, "deterministic relaxation").

This is **Romano–Elad–Milanfar's RED fixed point**, $z = D(z)$, with the diffusion model's
Tweedie denoiser as $D$. RED-Diff is its multi-level, stochastic generalisation.

So the three implicit families of [[Implicit Learners]] meet here. The diffusion family's
deterministic relaxation **is** an equilibrium-family model, with a layer that was trained as a
denoiser rather than end to end. That has a consequence for training: a DEQ trained end to end
need not be a denoiser of anything, whereas this one is, so it carries a density the DEQ lacks
(it can be sampled, and its relation has a probabilistic reading).

With a soft anchor ($\rho_F > 0$) the root solves
$\lambda_t\,\varepsilon_\theta(\alpha_t z^\star, t)_F + \rho_F^2 \odot (z^\star_F - z_{0,F}) = 0$, i.e.

$$
z^\star_F \;=\; z_{0,F} - \frac{\lambda_t}{\rho_F^2} \odot \varepsilon_\theta(\alpha_t z^\star, t)_F ,
$$

an anchored fixed point. For an exact score it is the proximal step of the level-$t$ smoothed
log-density in the metric $\rho^2$: the step a plug-and-play ADMM iteration takes.

## 4. The level is a bias–conditioning trade-off

At a single level, the inferred radius of a unit circle (component width $s = 0.05$) follows
$R - (s^2 + \sigma_t^2/\alpha_t^2)/(2R)$:

| $t$ | $\sigma_t$ | inferred radius at $x = 0.6$ | fixed-point error |
|---|---|---|---|
| 0.002 | 0.015 | 0.9981 | $1.0\times10^{-11}$ |
| 0.010 | 0.045 | 0.9977 | $5.3\times10^{-10}$ |
| 0.030 | 0.109 | 0.9927 | $6.7\times10^{-10}$ |
| 0.060 | 0.202 | 0.9766 | $1.3\times10^{-10}$ |

Small $t$ means little smoothing bias, but $\varepsilon_\theta$ varies on the scale $\sigma_t$, so the
field is stiff, basins of attraction are small, and for a *learned* network the region where
$\varepsilon_\theta$ is accurate shrinks too: networks are worst at small $t$. Multi-level fixed nodes
(§2) are the compromise. Coarse levels give wide basins, fine levels give little bias, and
annealing from coarse to fine (solve at a coarse node set, warm-start a finer one) gets both.

## 5. What the deterministic relaxation loses

- **Multimodality.** A point returns one branch, and which one depends on the start. A sampler would return all branches in proportion.
- **Uncertainty.** There are no error bars; the posterior is a Dirac ([[The Diffusion Factor]] §4).
- **Calibration.** The smoothing makes the relation the ridge of $p_t$, not the support of $p_0$; the bias is computable (§4) but present.

And what it gains: a deterministic, differentiable, warm-startable map. That is exactly what
an implicit *learner* needs, and what a factor graph's message schedule can call repeatedly.

Related: [[Implicit Diffusion Learners]], [[Inference Signatures]],
[[Backpropagation through Implicit Inference]], [[DEQ as a Relation]], [[The Equilibrium Family]],
[[ProxDM and Proximal Alternatives]]
