#definition #theorem #derivation #design

> **The derivation behind [[ImplicitREDDiff]], carried through.** A diffusion model defines a
> relation $R_\theta \subseteq Z$ as the zero set of a residual field. Inference is root-finding on
> that field and backpropagation is the adjoint of the root. Read through Tweedie's formula,
> RED-Diff's stop-gradient field turns out to be the **exact gradient** of a smoothed
> log-density. So the scalar energy and the multivariate energy of the original sketch are
> one object, and inference is a proximal step.
>
> Implemented in `lib/VariationalDiffusion.jl/src/implicit.jl` and `analytic.jl`; see [[implicit]], [[analytic]].

> Sources: original to this vault (design and analysis); builds on Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021, [arXiv:2011.13456](https://arxiv.org/abs/2011.13456); Mardani, Song, Kautz & Vahdat, *A Variational Perspective on Solving Inverse Problems with Diffusion Models*, [arXiv:2305.04391](https://arxiv.org/abs/2305.04391), Proposition 2; Efron, *Tweedie's Formula and Selection Bias*, JASA 2011; code: `implicit.jl`, `analytic.jl`
>
> Bibliography: [[Bibliography#^song2021sde|Song et al. 2021]] · [[Bibliography#^mardani2024reddiff|Mardani et al. 2024]] · [[Bibliography#^efron2011tweedie|Efron 2011]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Least Fixed Point](https://mathstruct.org/CategoryTheory-ML-Wiki/Least-Fixed-Point) · [Bisimulation](https://mathstruct.org/CategoryTheory-ML-Wiki/Bisimulation)

## 1. From functions to relations

Explicit learning fits $f_\theta : X \to Y$. Implicit learning fits a relation
$R_\theta \subseteq Z = Z_1 \times \dots \times Z_n$ through a residual
$r_\theta : Z \to E$:

$$
z \in R_\theta \quad:\Longleftrightarrow\quad r_\theta(z) = 0 .
$$

The direction (what is input, what is output) is not part of the model. It is chosen at
inference time, which is what makes one model serve every conditional. For a diffusion model
the residual comes from the score, and the rest of this note derives it.

## 2. Polarity is a precision vector

Split the coordinates of $z \in \mathbb R^n$ into inputs $X$, outputs $Y$ and latents $U$ with
diagonal selection matrices $P_x + P_y + P_u = \mathrm{Id}$ (pairwise orthogonal), and weight
them, $P = \rho_x P_x + \rho_y P_y + \rho_u P_u$. In the code this is a **precision vector**
$\rho \in [0, \infty]^n$, which is also what `precision_vector(f, polarity)` returns for a
[[The Diffusion Factor|DiffusionFactor]]:

| $\rho_i$ | coordinate $i$ is | constraint |
|---|---|---|
| $\infty$ | a **hard** input | $z_i = z_{0,i}$, imposed by projection |
| large, finite | a **soft** input (noisy evidence) | penalty $\tfrac12\rho_i^2(z_i - z_{0,i})^2$ |
| small, $> 0$ | an output **anchored** at a previous value | a proximal pull towards $z_{0,i}$ |
| $0$ | a free output or latent | none |

Two corrections to the original sketch:
- The typical ordering is $\rho_x \gg \rho_u \ge \rho_y \approx 0$, not the reverse: inputs are pinned hardest.
- A *nonzero* $\rho_y$ is not a nuisance. It anchors the output at its previous value $y_0$, and it is what makes inference **warm-startable and branch-selecting** (§6).

Write $F = \{i : \rho_i < \infty\}$ for the free coordinates and $C$ for the clamped ones.

## 3. The residual, and the theorem that makes it a gradient

RED-Diff's regulariser gradient (Mardani et al., Proposition 2) is, with weight $\lambda_t = \lambda\sigma_t/\alpha_t$,

$$
g_\theta(z) \;=\; \mathbb E_{t,\varepsilon}\bigl[\lambda_t\,(\varepsilon_\theta(\alpha_t z + \sigma_t\varepsilon, t) - \varepsilon)\bigr],
$$

the gradient of the score-matching term with the network Jacobian dropped (the "stop-gradient").
Adding the polarity term gives the **residual** on the free coordinates:

$$
r(z) \;=\; g_\theta(z) + \rho^2 \odot (z - z_0), \qquad r_F(z) = 0,\quad z_C = z_{0,C}.
$$

This is the right form of the "multivariate energy" of [[ImplicitREDDiff]]. Compared with the
sketch: the data term enters as $\rho^2$, not $\rho$, because it is the *gradient* of
$\tfrac12\lVert P(z_0 - z)\rVert^2$; the factor $\alpha_t$ from the reparametrisation is absorbed
into $\lambda_t$; and the residual is not squared inside the expectation.

> **Theorem (the stop-gradient field is conservative).** If $\varepsilon_\theta = \varepsilon^\ast$ is the
> optimal noise predictor, then
> $$
> g(z) \;=\; \nabla_z\,\Phi(z), \qquad \Phi(z) \;=\; -\,\mathbb E_{t,\varepsilon}\Bigl[\lambda_t\tfrac{\sigma_t}{\alpha_t}\,\log p_t(\alpha_t z + \sigma_t\varepsilon)\Bigr],
> $$
> and for any fixed set of noise nodes the same identity holds node by node, up to the
> constant vector $-\sum_k w_k\lambda_{t_k}\varepsilon_k$ (the gradient of a linear function).

*Proof.* Tweedie's formula for the Gaussian perturbation kernel gives
$\varepsilon^\ast(x, t) = \mathbb E[\varepsilon \mid x_t = x] = -\sigma_t\nabla_x\log p_t(x)$.
For a fixed $\varepsilon$, the chain rule gives
$\nabla_z \log p_t(\alpha_t z + \sigma_t\varepsilon) = \alpha_t \nabla\log p_t(\alpha_t z + \sigma_t\varepsilon)$.
Hence $\lambda_t\varepsilon^\ast(\alpha_t z + \sigma_t\varepsilon, t) = -\lambda_t\tfrac{\sigma_t}{\alpha_t}\nabla_z\log p_t(\alpha_t z+\sigma_t\varepsilon)$.
Averaging over nodes or integrating over $(t, \varepsilon)$ commutes with $\nabla_z$, and the
$-\lambda_t\varepsilon$ term does not depend on $z$. $\square$

Consequences:

1. **The scalar and the vector energy are one object.** Inference minimises
   $$
   \mathcal E(z) \;=\; \Phi(z) + \tfrac12\lVert P(z - z_0)\rVert^2,
   $$
   and the residual is its gradient. $\Phi$ is a *mixture of Gaussian-smoothed negative
   log-densities*: each noise level contributes $-\log p_t$ at scale $\sigma_t/\alpha_t$, with
   weight $\lambda\sigma_t^2/\alpha_t^2$.
2. **RED-Diff is not "biased" in the sense the vault used to say.** Its field is not the gradient
   of the regulariser it starts from. It *is* the exact gradient of $\Phi$, a different and
   well-defined prior. ([[RED-Diff as a Statistical Game]] §6 called it "approximate at its
   fixed point"; the precise statement is "exact for a smoothed prior".) The Gaussian
   calibration of that note's §4 is the special case where $\Phi$ is quadratic.
3. **For a learned network the field need not be a gradient.** $\partial_z g$ is then not
   symmetric, $\Phi$ does not exist, and only the residual is meaningful. So the *vector*
   formulation is the primary one: it is defined for every network, and it reduces to the
   scalar one exactly when the network is a true score.

The identity is checked numerically for a Gaussian mixture, whose $\varepsilon^\ast$ is closed form
(`analytic.jl`): field and $\nabla\Phi$ agree to $3\times10^{-12}$, and the Jacobian is symmetric to
machine precision (test suite, "the deterministic field is the gradient of a smoothed log-density").

## 4. Inference is a proximal point

With $\mathcal E = \Phi + \tfrac12\lVert P(z - z_0)\rVert^2$, a root of $r_F$ with $z_C = z_{0,C}$ is a
stationary point of

$$
z^\star \;\in\; \arg\min_{z_C = z_{0,C}}\; \Phi(z) + \tfrac12 \lVert P(z - z_0)\rVert^2 \;=\; \operatorname{prox}^{P^2}_{\Phi}(z_0)\ \text{restricted to the clamp},
$$

i.e. the **proximal operator of the smoothed log-prior in the metric $P^2$**. This makes
precise the slogan of [[The Diffusion Family]] ("a statistical game whose inversion is a
proximal operator") and connects it to [[ProxDM and Proximal Alternatives]]. ProxDM *learns* a
prox; here the prox is induced by a learned score.

## 5. Roots, stable roots, and the relation

$r(z) = 0$ also holds at maxima and saddles of $\mathcal E$. Those points are on the zero set
but are not plausible configurations. The relation one wants is the set of **stable** roots,

$$
R_\theta \;=\; \{ z : r_F(z) = 0,\ z_C = z_{0,C},\ \operatorname{sym}(\partial_{z_F} r_F) \succ 0 \},
$$

and `implicit_infer` reports both flags, `converged` and `stable`. When the field is a
gradient, stability is the second-order condition for a local minimum.

### The smoothing scale decides what relation you get

$\Phi$ smooths the data density at scales $\sigma_t/\alpha_t$, so the recovered relation is the
**ridge of a smoothed density**, not the support of the data. For a thin ring of radius $R$
(mixture components of width $s$) smoothed at a single level, the radial mode moves inward by,
to first order,

$$
r^\star \;\approx\; R - \frac{s^2 + \sigma_t^2/\alpha_t^2}{2R}.
$$

Measured on the unit circle, inferring $y$ at $x = 0$:

| noise levels used | largest $\sigma_t$ | inferred radius |
|---|---|---|
| $t \le 0.01$ | 0.045 | 0.997 |
| $t \le 0.05$ (the package default) | 0.172 | 0.980 |
| $t \le 0.1$ | 0.322 | 0.924 |
| $t \le 0.2$ | 0.584 | 0.658 |
| $t \le 1$ (RED-Diff's training range) | 1.000 | **0.000** |

A single noise-free level at $t = 0.03$ gives $0.9927$ and the formula predicts $0.9927$; at
$t = 0.06$, $0.9766$ against $0.977$.

**RED-Diff's own weighting, $\lambda_t \propto \sigma_t/\alpha_t$ over $t\in[t_{\min},1]$, smooths a circle until its
density peaks at the centre**, and then the relation is a single point. For implicit learning
the noise levels used *at inference* must stay below the curvature scale of the relation.
That is a design parameter (`levels` in `field_nodes`), and it is a bias–conditioning
trade-off: small $\sigma_t$ means little bias but a stiff, ill-conditioned field ($\varepsilon^\ast$
scales like $1/\sigma_t$ near the data).

## 6. Multi-valued relations and branches

On the circle, a hard input $x = 0.6$ has two stable outputs, $y \approx \pm 0.78$. Inference
returns the one whose basin contains the starting point; a small $\rho_y$ anchor at a previous
$y_0$ biases the choice. Near $x = 1$ the two branches merge: smoothing turns the
discriminant locus of [[Branches and the Discriminant]] into a region where only one root
survives, and starts from either side land on it. Outside the support ($x = 1.05$) inference
still answers with a point near the ridge: the "closest point to the variety" behaviour that
[[README]]'s table predicts for implicit learners.

Three caveats, all observed:
- With few noise samples the deterministic field breaks the $y \mapsto -y$ symmetry, so the merged root sits slightly off the axis. That is a sample-average artefact, smaller with more nodes.
- In flat regions far from data the field is tiny, and descent can stall without converging. The solver reports this rather than hiding it.
- Training through inference (see [[Backpropagation through Implicit Inference]] §7) only moves the branches inference visits. The rest of the old relation survives as other branches.

## 7. The pieces, and where they are worked out

| question | note |
|---|---|
| what inference *is*, in all its signatures | [[Inference Signatures]] |
| the Lagrangian, adjoint and gradient, worked out | [[Backpropagation through Implicit Inference]] |
| the deterministic limit, and why it is a DEQ | [[Deterministic Relaxation]] |
| the statistical-game reading, and what is missing | [[The Implicit Diffusion Factor as a Statistical Game]] |

````tabs
tab: Julia
**Docs:** [VariationalDiffusion API](https://mathstruct.org/Lenticulum.jl/dev/packages/variationaldiffusion/)
```julia
using VariationalDiffusion, LuxCore, Random, LinearAlgebra
sched = VPSDE()
θ = range(0, 2π; length = 49)[1:48]
ring = NoisePredictor(GaussianMixtureEps(sched, vcat(cos.(θ)', sin.(θ)'); s = 0.05), sched)  # exact score of a circle
ps, st = LuxCore.setup(Xoshiro(0), ring)
m = ImplicitDiffusion(ring, field_nodes(Xoshiro(1), 2; samples = 8))   # levels t ∈ [0.002, 0.05]
ρ = [Inf, 0.0]                              # x hard input, y free output
up, _ = implicit_infer(m, [0.6, 0.5], ρ, ps, st)     # start above the axis
dn, _ = implicit_infer(m, [0.6, -0.5], ρ, ps, st)    # start below
(up.z[2], dn.z[2])                          # ≈ (0.78, -0.78): two branches of one relation
(up.converged, up.stable)                   # (true, true)
```
````
