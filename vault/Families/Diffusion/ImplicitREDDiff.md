#model #design

> **The founding note of the project.** Carried through — inference, backpropagation, the
> deterministic relaxation and the statistical-game reading — in [[Implicit Diffusion Learners]].
> Worked out in full in [[The Diffusion Family]] and
> implemented in `lib/VariationalDiffusion.jl`; the energy below is the one the code computes
> (`reddiff.jl`, `factor.jl`).

> Sources: original to this vault (design and analysis; no single paper). Builds on Mardani, Song, Kautz & Vahdat, *A Variational Perspective on Solving Inverse Problems with Diffusion Models* (2023) — [arXiv:2305.04391](https://arxiv.org/abs/2305.04391); Fang, Díaz, Buchanan & Sulam, *Beyond Scores: Proximal Diffusion Models* (2025) — [arXiv:2507.08956](https://arxiv.org/abs/2507.08956); the [Implicit Layers tutorial](https://implicit-layers-tutorial.org/); code: `reddiff.jl`, `factor.jl`
>
> Bibliography: [[Bibliography#^mardani2024reddiff|Mardani et al. 2024]] · [[Bibliography#^fang2025proxdm|Fang et al. 2025]]

## Implicit learning: relations instead of functions

**Implicit** means replacing learned *functions* by learned *relations*
([Implicit Layers tutorial](https://implicit-layers-tutorial.org/)).

| Explicit machine learning | Implicit learning |
|---|---|
| approximator: functions $f_\theta : X \to Y$ | approximator: relations $R_\theta \subseteq X_1 \times \cdots \times X_n$ |

How do we learn a relation?

- Introduce an error (energy) space $E$, assumed multivariate.
- Learn a function
  $$r_\theta : X_1 \times \cdots \times X_n \to E$$
  and read the relation off its approximate zero set:
  $$(x_1,\ldots,x_n) \in R_\theta \;:\Longleftrightarrow\; r_\theta(x_1,\ldots,x_n) \approx 0.$$

## A simple example: polynomials versus varieties

| Aspect | Explicit | Implicit |
|---|---|---|
| **Approximator** | multivariate polynomials | algebraic varieties |
| **Inference** | forward evaluation | root finding |
| **Backpropagation** | reverse-mode automatic differentiation | implicit function theorem / differential algebra |
| **Universal approximation** | continuous functions on compacta (Weierstraß) | compact smooth manifolds (Nash–Tognoli) |
| **Well-posedness** | always single-valued | may be multi-valued or empty; in general only the closest point to the variety |
| **Loss** | $\lVert f_\theta(x) - y\rVert^2$ | $\lVert r_\theta(x_1,\ldots,x_n)\rVert^2$ |
| **Symmetry** | fixed direction from input to output | symmetric: no distinguished input or output |
| **Cost of inference** | cheap | expensive (Newton's method, …) |
| **Layer connections** | directed acyclic graph | arbitrary connected graph |

Implicit layers, deep equilibrium networks (DEQs) and neural ODEs realise this to a limited
degree; see `lib/ImplicitLayers.jl`.

## Implicit RED-Diff

### Choosing input and output channels

Let $P_{\mathrm{in}}, P_{\mathrm{out}}, P_{\mathrm{latent}} \in \{0,1\}^{n\times n}$ be diagonal
selection matrices with

$$
P_{\mathrm{in}} + P_{\mathrm{out}} + P_{\mathrm{latent}} = \mathrm{Id}, \qquad
P_{\mathrm{in}}P_{\mathrm{out}} = P_{\mathrm{in}}P_{\mathrm{latent}} = P_{\mathrm{out}}P_{\mathrm{latent}} = 0,
$$

so that $x_{\mathrm{in}} = P_{\mathrm{in}}x$ and $x_{\mathrm{out}} = P_{\mathrm{out}}x$: **the
choice of input and output is a choice of masks, made at inference time, not at training
time.** Weight the three blocks with precisions:

$$P = \rho_{\mathrm{in}}P_{\mathrm{in}} + \rho_{\mathrm{out}}P_{\mathrm{out}} + \rho_{\mathrm{latent}}P_{\mathrm{latent}}.$$

In the code this is `precision_vector` in `factor.jl`; the limit $\rho_{\mathrm{in}} \to \infty$
is implemented as a hard clamp (projection) rather than an infinite penalty.

### The energy

$$E(x_0, x) \;=\; \mathbb{E}_{t,\varepsilon}\Bigl[\omega(t)\,\bigl\lVert \varepsilon_\theta(\alpha_t x + \sigma_t\varepsilon,\, t) - \varepsilon \bigr\rVert_2^2\Bigr] \;+\; \tfrac12\,\bigl\lVert P(x_0 - x)\bigr\rVert^2$$

The first term is the diffusion prior (the entropy side of the free energy, see
[[RED-Diff as a Statistical Game]]); the second ties $x$ to the observed $x_0$ on the chosen
channels. Its gradient in $x$ comes from the
[RED-Diff](https://arxiv.org/abs/2305.04391) machinery: the score network supplies the prior
gradient without differentiating through $\varepsilon_\theta$.

Alternatively, [ProxDM](https://arxiv.org/abs/2507.08956) learns a proximal operator instead
of a score; see [[ProxDM and Proximal Alternatives]].
