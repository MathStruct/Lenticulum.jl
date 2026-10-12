#implementation

> Proximal diffusion models: a diffusion prior that is queried through its **proximal operator**
> instead of its score. Implemented here are the interface, an exact oracle, ProxDM's sampler,
> proximal inference for the implicit learner and its adjoint, ProxDM as a factor's inversion,
> and the training loss.

> Sources: Fang, Díaz, Buchanan & Sulam, *Beyond Scores: Proximal Diffusion Models*, [arXiv:2507.08956](https://arxiv.org/abs/2507.08956) (Algorithm 1, Eq. 9); Zhu et al., *Denoising Diffusion Models for Plug-and-Play Image Restoration*, CVPR 2023 (DiffPIR, the half-quadratic splitting); code: `proxdm.jl`
>
> Bibliography: [[Bibliography#^fang2025proxdm|Fang et al. 2025]] · [[Bibliography#^zhu2023diffpir|Zhu et al. 2023]]
>
> Theory: [[ProxDM and Proximal Alternatives]] · [[Deterministic Relaxation]] · [[Inference Signatures]]

**Notation.** As everywhere in this vault, $z \in Z$ is the joint state and the prior is
$p_t$ on $Z$. The paper writes $x$ for the same thing.

## 1. The object

$$
\operatorname{prox}_{-\lambda\log p_t}(v) \;=\; \arg\min_u\ \tfrac12\lVert u - v\rVert^2 - \lambda\log p_t(u),
\qquad u - v = \lambda\,\nabla\log p_t(u).
$$

The optimality condition is a gradient step evaluated at the **new** point: a backward
(implicit) Euler step, where a score is a forward one. That is the whole idea of ProxDM:
discretise the reverse SDE backwards, and the network you need is a prox, not a score.

| type | what it is |
|---|---|
| `AbstractProximalPredictor` | anything with `proximal(p, v, t, λ, ps, st) -> (u, st)` |
| `ProxNetwork(model, schedule; input, ad)` | a Lux model in ProxDM's parametrisation $f_\theta(v; t, \lambda) = v - \sqrt\lambda\,\varepsilon_\theta(v; t, \lambda)$; conditioned on **two** scalars |
| `MixtureProx(::GaussianMixtureEps)` | the exact prox of a Gaussian mixture, by Newton with backtracking; the oracle a perfectly trained `ProxNetwork` would match |

`MixtureProx` uses $\nabla\log p_t$ and its Hessian directly from the mixture, not
$-\varepsilon/\sigma_t$, so it is valid at $t = 0$, where the sampler's last step needs it.

## 2. The sampler (Algorithm 1)

On a uniform grid $0 = t_0 < \dots < t_N = T$ with $\gamma_k = \int_{t_{k-1}}^{t_k}\beta$:

- **PDA** (`hybrid = false`): $z_{k-1} = \operatorname{prox}_{-\frac{2\gamma_k}{2-\gamma_k}\log p_{t_{k-1}}}\bigl(\tfrac{2}{2-\gamma_k}(z_k + \sqrt{\gamma_k}\,\xi_k)\bigr)$, which needs $\gamma_k < 2$; `proxdm_sample` throws otherwise.
- **PDA-hybrid** (`hybrid = true`): $z_{k-1} = \operatorname{prox}_{-\gamma_k\log p_{t_{k-1}}}\bigl((1+\tfrac12\gamma_k)z_k + \sqrt{\gamma_k}\,\xi_k\bigr)$: the linear drift explicit, the score implicit; no step-size limit.

The last step evaluates the prox at $t_0 = 0$, so the output is already denoised; there is no
separate final denoising step.

**Validation** (one Gaussian, mean $(0.5, -1)$, std $0.3$, exact prox, 1,500 samples):

| steps | 20 | 40 | 100 | 400 |
|---|---|---|---|---|
| sample std (true 0.30) | 0.248 | 0.268 | 0.286 | 0.293 |

The mean is right at every step count; the spread converges at first order, as a backward
Euler scheme should. The sampler on the circle mixture gives radius $1.002 \pm 0.024$.

## 3. Proximal inference for the implicit learner

`prox_infer(p, z₀, ρ, t, ps, st; λ)` is the implicit learner's query with a proximal prior.
It splits the relaxed problem

$$
\min_{u, w}\ -\log p_t(u) + \tfrac1{2\lambda}\lVert u - w\rVert^2 + \tfrac12\lVert P(w - z_0)\rVert^2,
\qquad P = \operatorname{diag}(\rho),
$$

and alternates the two proximal maps (half-quadratic splitting, as in DiffPIR):

$$
u \leftarrow \operatorname{prox}_{-\lambda\log p_t}(w), \qquad
w_i \leftarrow \frac{u_i/\lambda + \rho_i^2 z_{0,i}}{1/\lambda + \rho_i^2}\ (\rho_i < \infty), \qquad
w_i \leftarrow z_{0,i}\ (\rho_i = \infty).
$$

Hard clamps are exact; the free coordinates ($\rho_i = 0$) take $w_i = u_i$. As $\lambda \to 0$
the coupling becomes $u = w$ and the problem becomes the implicit learner's
([[Deterministic Relaxation]]). It returns an `ImplicitSolution` with `z = w` (which satisfies
the clamps) and `residual` $= \lVert u - w\rVert$.

On the circle mixture, the query "$x = 0.6$, which $y$?" from $y_0 = \pm 0.5$ returns
$y = \pm 0.7936$, symmetric to the last digit, in 20 iterations.

## 4. Training: proximal matching

`proximal_matching_loss` is one sample of the paper's Eq. 9:

$$
\ell_{PM}\bigl(\varepsilon_\theta(z_t + \sqrt\lambda\,\xi;\ t, \lambda),\ \xi;\ \zeta\bigr),
\qquad \ell_{PM}(a, b; \zeta) = 1 - \exp\bigl(-\lVert a - b\rVert^2/(d\zeta^2)\bigr).
$$

The squared loss would train the MMSE denoiser $\mathbb E[z \mid v]$; this bounded loss, as
$\zeta \to 0$, trains the **MAP** denoiser, which is the prox. Shrink $\zeta$ over training.
The loss is a plain function of `ps`, so any Lux training loop and any AD backend trains it.

## 5. The adjoint of proximal inference

`prox_infer` returns the fixed point of the splitting. With $a_i = 1/(1 + \lambda\rho_i^2)$ it
satisfies, on the coordinates that are not hard-clamped ($F$),

$$
w_F = a \odot \operatorname{prox}(w)_F + (1 - a)\odot z_{0,F}, \qquad w_C = z_{0,C},
$$

a root of $r(w_F) = w_F - a\odot\operatorname{prox}(w)_F - (1-a)\odot z_{0,F}$, so the implicit
function theorem applies as it does to `implicit_infer` ([[Backpropagation through Implicit Inference]]).
With $D$ the prox's input Jacobian at $w$ and $u = \operatorname{prox}(w)$, one linear solve

$$
(I - \operatorname{diag}(a)\,D_{FF})^\top \mu = \bar z_F, \qquad y = a\odot\mu \text{ on } F,\ 0 \text{ on } C,
$$

gives every cotangent:

| w.r.t. | cotangent |
|---|---|
| parameters $\theta$ of the prox | $(\partial_\theta\operatorname{prox})^\top y$ |
| hard-clamped inputs $z_{0,C}$ | $\bar z_C + (D^\top y)_C$ |
| soft and free targets $z_{0,F}$ | $(1 - a)\odot\mu$ |
| precisions $\rho_F$ | $-2\lambda\rho_F\odot a^2\odot(u_F - z_{0,F})\odot\mu$ |

`prox_pullback(p, sol, z₀, ρ, t, z̄, ps, st; λ, linsolve)` computes them; it is also the
`ChainRulesCore.rrule` of `prox_infer`, so Zygote differentiates through proximal inference
without unrolling it. The matrix is $I - \operatorname{diag}(a)D_{FF}$, not a Jacobian of a
field: the prox is already a solved problem, and its derivative is what the adjoint needs.
Only vector-Jacobian products of the prox appear, so `linsolve = KrylovSolve()` runs GMRES on
them and never forms $D$.

The prox's derivatives come from three places:

| prox | input VJP / Jacobian | parameter VJP |
|---|---|---|
| `MixtureProx` | exact: $D = (I - \lambda\nabla^2\log p_t(u))^{-1}$, the implicit function theorem on the prox's own optimality condition | exact: $\lambda(\partial_\mu\nabla\log p_t)^\top D^\top y$, valid at $t = 0$ |
| `ProxNetwork(…; ad)` | $y - \sqrt\lambda(\partial_v\varepsilon_\theta)^\top y$ through DifferentiationInterface | $-\sqrt\lambda(\partial_\theta\varepsilon_\theta)^\top y$ through DifferentiationInterface |
| anything else, or no `ad` | central differences of `proximal` | `ArgumentError` |

**Validation** (`RING`, $t = 0.005$, $\lambda = 0.01$). Against central differences of the
whole solve, for a hard clamp, a soft target, a soft clamp with a free coordinate, and all
coordinates soft, every cotangent (clamps, targets, precisions, mixture means) agrees to
between 1e-8 and 1e-5 relative, limited by the finite differences, not the adjoint; the test
suite asserts 1e-4 relative against a cheaper reference (below). GMRES and the dense solve agree to 1e-12; Zygote through the `rrule` equals
the pullback exactly. A `ProxNetwork` (a two-layer MLP) with `AutoZygote()`: input VJP and
Jacobian equal finite differences to 1e-8, a weight's cotangent through the whole solve to
3e-6 relative.

Finite differences need care here. The splitting contracts slowly when the answer can slide
along the relation (a soft clamp, or an untrained network: 350 iterations), and then its
stopping rule leaves an error of about $\text{tol}/(1-\kappa)$, with $\kappa$ the contraction factor. A
difference step of 1e-6 divides that by 1e-6 and disagreed with the adjoint by 1e-3; steps of
1e-4 to 1e-3 agree. Even then the reference was fragile: the same check passed alone and failed
inside the full suite at 1.5e-4, so the tests run the reference solves for a fixed 1000 sweeps
instead of to a tolerance. The adjoint does not have this problem: it solves the linear system
to its own tolerance.

## 6. As a factor's inversion

`DiffusionFactor(blocks, pred; prox = ProxDM(prox; t, λ, tol, maxiters, linsolve))` makes
ProxDM usable in a factor graph, alongside `REDDiff` and `ImplicitProx`. `invert` and
`factor_message` run `prox_infer` on the factor's state space with the polarity's precisions,
warm-started at the incoming points (the same problem `implicit_solution` builds for
`ImplicitProx`), and return a `DiracBelief` on the target. `implicit_solution` returns the
report and `implicit_factor_pullback` the backward pass through `prox_pullback`.

**Two parameter trees.** It is the first factor in the project with two models: the
`NoisePredictor` (the prior; the free energy's score term needs it) and the prox (the
inversion). Its parameters and states are `(predictor = …, prox = …)`, set up by
`LuxCore.setup(rng, f)`; a `MixtureProx` gets `(μ = …,)`. This is the structure
[[ProxDM and Proximal Alternatives]] §4 anticipated: the inversion is a separate learned object,
not a function of the prior's parameters, so the factor cannot be one Lux layer over one tree.
The factor-level pullback returns `ps = (predictor = 0, prox = ps̄)`, and the tests check that
moving the prior's means leaves the inversion unchanged while changing the score term.

The score summand uses fixed nodes (`nodes`, by default 8 from `Xoshiro(0)`) at weight
$\lambda = 1$, since `ProxDM`'s own `λ` is the splitting's coupling and has no score meaning.

**Validation** (`RING`, `MixtureProx`): the factor's answer equals bare `prox_infer` exactly;
the two branches from opposite warm starts are symmetric to 1e-8; the pullback's input and
precision cotangents match finite differences of `invert`; with a `ProxNetwork`, a weight's
cotangent in the `prox` tree matches finite differences of the factor's answer to 1e-3.

**A dispatch trap.** The methods for a ProxDM factor dispatch on
`DiffusionFactor{N,D,P,<:ProxDM} where {N, D<:Tuple, P<:NoisePredictor}`. Written with
unbounded `D` and `P` the signature is not a subtype of the plain `DiffusionFactor` signature (the
struct's bounds are part of it), so Julia never prefers it and silently calls the generic method.

## 7. Implementation difficulties and open ends

- **Non-convex priors make the prox multivalued.** For a mixture, $-\log p_t$ is not convex,
  and $\operatorname{prox}$ can have several minimisers. `MixtureProx` uses Newton where
  $I - \lambda\nabla^2\log p_t$ is positive definite and a gradient step otherwise, and returns
  the local minimiser reached from $v$. A learned `ProxNetwork` makes the same choice
  implicitly, wherever training put it.
- **The PDA step-size limit.** $\gamma_k < 2$ fails for coarse grids near $t = T$ under the
  default VP schedule. The hybrid scheme removes it; PDA throws with a message saying so
  instead of returning garbage.
- **Backpropagating at a different noise level.** `prox_pullback` takes `t` separately, but
  away from the inference level the fixed-point equation no longer holds and the result is not
  the gradient of anything; whether it is a useful, better-conditioned surrogate
  ([[Backpropagation through Implicit Inference]] §9) is untested.
- **No derivative in `t` or `λ`.** Both are treated as hyperparameters.
- **`stable` is not assessed** by the splitting; it reports `true` when converged. The
  adjoint's matrix $I - \operatorname{diag}(a)D_{FF}$ is nonsingular wherever the splitting
  contracts, which is when it converges at all.
- **The sampler is first order** (§2), as backward Euler is; no higher-order scheme yet.
- **Point messages only.** A Laplace Gaussian for proximal inference would need the Hessian
  of the relaxed problem, which the prox gives only through its Jacobian
  ($\nabla^2(-\log p_t) = (D^{-1} - I)/\lambda$ at the prox's output); not built.
- **The two trees are separate even when they could be shared.** For `MixtureProx` of the
  factor's own mixture, both trees hold the same means; training one does not move the other.

Related: [[implicit]], [[reddiff]], [[backends]], [[analytic]],
[[ProxDM and Proximal Alternatives]], [[Inference Signatures]], [[Deterministic Relaxation]]
