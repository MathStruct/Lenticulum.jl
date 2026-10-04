#definition #theorem #design

> Is an implicit diffusion learner a [parameterized statistical game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)?
> **Yes, once a polarity is chosen**: the composite of a *prior game*, whose energy is the
> smoothed negative log-density, and a *likelihood game*, whose energy is the clamp. Its
> inversion is the proximal inference of [[Implicit Diffusion Learners]] §4. Two things are
> missing: a real entropy for the inversion (the posterior is a Dirac) and, for a learned
> non-conservative network, a scalar loss. The two ways of training it — the game's
> block-diagonal gradient and the Lagrangian of the original sketch — are different semantics,
> and both are now implemented.

> Sources: St Clere Smithe & Perin, *AutoBayes*, [arXiv:2503.18608](https://arxiv.org/abs/2503.18608), Definitions 1, 20, 22, 27–29, Theorem 23, Remarks 24, 30; Mardani et al. [arXiv:2305.04391](https://arxiv.org/abs/2305.04391) §3 (the KL decomposition behind RED-Diff); Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 (the denoising loss as a likelihood bound); code: `implicit.jl`, `factor.jl`, `statistical_game.jl`
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Para Construction](https://mathstruct.org/CategoryTheory-ML-Wiki/Para-Construction) · [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor)

## 1. The polarity chooses the game

A diffusion model is one joint density over $Z$; it has no direction. A statistical game
$c : X \multimap Y$ does. The polarity supplies it, with the dictionary of [[Channels and Polarity]]:

| AutoBayes | here |
|---|---|
| $X$ (unobserved, solved for) | output coordinates, $\rho = 0$ |
| $Y$ (observed, clamped) | input coordinates, $\rho = \infty$ or large |
| $\llbracket c\rrbracket$ (latent) | latent coordinates |

Different polarities give different games from the same network: a $3^n - 2^n$-element family,
one for each way to read the relation ([[The Diffusion Factor]] §2).

## 2. The elements, one by one

Following RED-Diff's derivation, minimising $\mathrm{KL}(q \,\Vert\, p(x, u \mid y))$ splits into three parts:

$$
\underbrace{\mathbb E_q\bigl[-\log p(y \mid x, u)\bigr]}_{\text{likelihood energy}}
\;+\;
\underbrace{\mathbb E_q\bigl[-\log p_\theta(x, u)\bigr]}_{\text{prior energy}}
\;-\;
\underbrace{H(q)}_{\text{entropy}}
\;+\; \log p(y).
$$

Every element of Definition 20 then has a home:

| Definition 20 / 27 | the implicit diffusion factor | status |
|---|---|---|
| generative model, prior $\pi$ on $X$ and kernel $c : X \rightsquigarrow \llbracket c\rrbracket \times Y$ | the diffusion joint $p_\theta(z)$, read as prior on $(X, \llbracket c\rrbracket)$ times conditional of $Y$ | **present**, implicitly: only its (smoothed) score is available |
| inversion $c'_\pi : Y \rightsquigarrow X \times \llbracket c\rrbracket$ | proximal inference, `implicit_infer` | **present**; a Dirac at $z^\star$ |
| energy $l^c$ (pointwise) | the clamp, $\tfrac12\lVert P(z - z_0)\rVert^2 = -\log$ of a Gaussian observation of precision $\rho^2$ | **present** |
| the prior as its own game (Remark 24) | energy $\Phi_\theta(z)$, the smoothed $-\log p_\theta$ of [[Implicit Diffusion Learners]] §3, entropy 0 | **present** for an exact score; for a learned network only $\nabla\Phi$ exists (the residual) |
| entropy $H^c$ | $H(q)$ of the inversion | **missing**: $-\infty$ for a Dirac; needs $q = \mathcal N(\mu, \Sigma)$ |
| loss $F^c$ | $\mathcal E(z^\star) = \tfrac12\lVert P(z^\star - z_0)\rVert^2 + \Phi_\theta(z^\star)$, up to the entropy constant | **present** for an exact score and for an energy-parametrised network (`implicit_energy`, [[energy]]); a denoising-loss estimate otherwise |
| parameters $\Theta$ (Definition 27, $\mathbf{Para}$) | network weights; **also** the precisions $\rho$ and λ | **present**; $\rho$ is differentiable by the adjoint |

> [!important] A correction to [[RED-Diff as a Statistical Game]] §2
> That note files the score-matching term as the game's *entropy* and then warns that it "is not
> an entropy". The decomposition above puts it where it belongs: it is the **energy of the prior
> game** (Remark 24, "priors are games too"), a cross-entropy $\mathbb E_q[-\log p_\theta]$. The
> entropy slot holds $H(q)$, the inversion's own entropy. The composite of prior game and
> likelihood game then has loss $F = \mathbb E_q[l + \Phi_\theta] - H(q)$, the variational free energy
> with the smoothed prior, by Theorem 23. The remaining defect is not a mislabelled term but a
> degenerate $q$.

## 3. Composition

Two implicit diffusion factors sharing a coordinate compose like any games: energies add,
entropies chain (Definition 22), and with vector energies the sum becomes a direct sum
([[Scalar and Multivariate Energy]]). The multivariate version is the natural one here, because
the residual is defined for every network while the scalar loss is not (§2, prior-game row).
The composite residual of a factor graph is the stacked residuals of its factors, and
inference on the graph is a root of the stack.

What composition does *not* fix is the message problem of [[The Diffusion Factor]] §4: a factor
whose prior cannot be divided out sends posteriors, not likelihoods.

## 4. Two ways to train it, and what each means

### (A) The game's gradient: block-diagonal, generative

Definition 29 differentiates the loss with the inversion **held fixed**, i.e. the
`DiagonalCoupling` of [[Factors are Parameterized Statistical Games]]:

$$
\nabla_\theta F \;=\; \mathbb E_{q}\bigl[\nabla_\theta \Phi_\theta(z)\bigr]\Big|_{q = \delta_{z^\star}} .
$$

Since $\Phi_\theta$ is, up to $\theta$-independent constants, a weighted denoising loss (an
upper bound on $-\log p_\theta$ under the ELBO weighting), this is **denoising-score-matching
training on the completed configuration** $z^\star$. Alternating with inference is EM with a
diffusion prior: E-step, infer the missing coordinates; M-step, train the denoiser on the
completed data. It learns the *whole* relation, because it fits a density.

### (B) The Lagrangian of the original sketch: exact through the inversion, discriminative

Differentiating a downstream loss $\ell(z^\star(\theta))$ *through* inference keeps the term
Definition 29 drops: the dependence of the inversion on $\theta$ (Remark 30's laxness). The
adjoint of [[Backpropagation through Implicit Inference]] computes it exactly. This is
supervised implicit learning, the way a DEQ is trained.

| | (A) game gradient | (B) bilevel / adjoint |
|---|---|---|
| differentiates through inference | no (stop-gradient) | yes (implicit function theorem) |
| objective | free energy, i.e. density fit | a task loss on the output |
| needs | parameter gradient of the denoising loss | one adjoint solve, plus one parameter-VJP per node |
| learns | every branch of the relation | the branches inference visits (§7 of the backpropagation note: the old upper arc of the circle survived) |
| coupling in [[Factors are Parameterized Statistical Games]] | `DiagonalCoupling` | `ExactCoupling` on the inversion |

They are complementary. (A) shapes the density; (B) shapes the answer the inference procedure
actually returns, including its smoothing bias, which (A) ignores. A practical scheme is (A)
to pretrain the relation and (B) to fine-tune for a task.

## 5. The deterministic relaxation as a game

With the noise-free single-level relaxation ([[Deterministic Relaxation]] §3) the prior energy
is $-\log p_t$ at one level and the inversion is the denoiser's fixed point. That is the same game
as a DEQ factor with an energy, except that the energy has a density reading. The equilibrium
family's factors "contribute no entropy" ([[DEQ as a Relation]] §5); this one contributes a
prior energy, and would contribute an entropy if its inversion were Gaussian.

## 6. What is left to do

1. **A Gaussian inversion.** Keep $\sigma > 0$ in RED-Diff's variational family, iterate on
   $(\mu, \Sigma)$, and $H(q)$ becomes finite. The Laplace approximation $\Sigma = J_{FF}^{-1}$ at
   $z^\star$ is already computed by the adjoint, so a first version costs nothing extra.
2. ~~**A scalar loss for learned networks**~~ — done for energy-parametrised networks:
   with $\varepsilon_\theta = \sigma_t\nabla_x E_\theta$ the field is the gradient of
   `implicit_energy`, exactly ([[energy]]). A network that outputs $\varepsilon$ directly still
   has only the denoising-loss estimate.
3. ~~**AD for `epsilon_vjp_params`** on a Lux network~~ — done, through any AD backend ([[backends]]).
4. ~~**The factor interface**~~ — done: `DiffusionFactor(…; prox = ImplicitProx(nodes))` inverts
   with the deterministic solver; `implicit_solution` reports, `implicit_factor_pullback`
   differentiates ([[implicit_factor]]).

Related: [[Implicit Diffusion Learners]], [[Inference Signatures]],
[[Backpropagation through Implicit Inference]], [[Deterministic Relaxation]],
[[RED-Diff as a Statistical Game]], [[Factors are Parameterized Statistical Games]],
[[Scalar and Multivariate Energy]]
