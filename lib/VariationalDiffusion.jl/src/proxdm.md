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
- **PDA-hybrid** (`scheme = :hybrid`, or `hybrid = true`): $z_{k-1} = \operatorname{prox}_{-\gamma_k\log p_{t_{k-1}}}\bigl((1+\tfrac12\gamma_k)z_k + \sqrt{\gamma_k}\,\xi_k\bigr)$: the linear drift explicit, the score implicit; no step-size limit.
- **Trapezoidal** (`scheme = :trapezoidal`): the drift averaged over both ends of the step (Crank–Nicolson),

  $$
  z_{k-1} = z_k + \tfrac{\gamma_k}{2}\bigl[(\tfrac12 z_k + s_k) + (\tfrac12 z_{k-1} + s_{k-1})\bigr] + \sqrt{\gamma_k}\,\xi_k,
  $$

  which is again one prox: $z_{k-1} = \operatorname{prox}_{-\lambda\log p_{t_{k-1}}}(v)$ with
  $\lambda = \frac{2\gamma_k}{4-\gamma_k}$ and $v = \frac{(1+\gamma_k/4)z_k + (\gamma_k/2)s_k + \sqrt{\gamma_k}\,\xi_k}{1-\gamma_k/4}$; it needs $\gamma_k < 4$.
  The explicit score $s_k = \nabla\log p_{t_k}(z_k)$ is free: the previous step returned
  $z_k = \operatorname{prox}_\lambda(v)$, and the prox's optimality condition gives
  $s_k = (z_k - v)/\lambda$. The first step is a hybrid step, which supplies the first score.

The last step evaluates the prox at $t_0 = 0$, so the output is already denoised; there is no
separate final denoising step.

**Validation** (one Gaussian, mean $(0.5, -1)$, std $0.3$, exact prox, 1,500 samples):

| steps | 20 | 40 | 100 | 400 |
|---|---|---|---|---|
| sample std (true 0.30) | 0.248 | 0.268 | 0.286 | 0.293 |

The mean is right at every step count; the spread converges at first order, as a backward
Euler scheme should. The trapezoidal scheme is second order. Its error in the spread, against
PDA and the hybrid scheme (same Gaussian, 20,000 samples, common random numbers; the Monte Carlo
floor is about 0.001):

| steps | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|
| PDA | −0.097 | −0.054 | −0.028 | −0.015 | −0.007 |
| hybrid | −0.099 | −0.056 | −0.030 | −0.016 | −0.007 |
| trapezoidal | −0.023 | −0.007 | −0.001 | (floor) | (floor) |

Halving the step halves the first-order error and divides the trapezoidal one by 3 to 6 before
it reaches the floor. For a linear drift with constant coefficients and additive noise the
trapezoidal step keeps the stationary variance exactly (the factors $1 \pm \gamma/4$ cancel in it), which is why the
spread, the hard part for backward Euler, is right.

On the circle mixture (radial width 0.050, 3,000 samples), the radius's spread:

| steps | 20 | 50 | 100 |
|---|---|---|---|
| PDA | 0.015 | 0.029 | 0.038 |
| hybrid | 0.016 | 0.030 | 0.038 |
| trapezoidal | 0.027 | 0.044 | 0.048 |

All three put the mean radius at 1.001 to 1.002. Fifty trapezoidal steps beat a hundred PDA steps;
the remaining error sits at small $t$, where the ring's fine structure is and a uniform grid
in $t$ is coarse (the clamped sampler of [[sampler]] uses a log-SNR grid for that reason).

The free score holds only as far as the prox is exact. A learned `ProxNetwork` is an
approximate prox; $(u - v)/\lambda$ is then its implied score, consistent with its own
implicit steps but carrying its training error into the explicit half too.

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
- **Backpropagating at a different noise level** gives a surrogate, not a gradient (§8). In
  the one experiment so far it trains no better than the exact adjoint.
- **No derivative in `t` or `λ`.** Both are treated as hyperparameters.
- **`stable` is not assessed** by the splitting; it reports `true` when converged. The
  adjoint's matrix $I - \operatorname{diag}(a)D_{FF}$ is nonsingular wherever the splitting
  contracts, which is when it converges at all.
- **A log-SNR time grid** for `proxdm_sample` would move the steps to small $t$; with it the
  first steps near $T$ get larger, which PDA's $\gamma_k < 2$ and the trapezoidal $\gamma_k < 4$
  would then hit (the hybrid first step has no limit).
- **Point messages only.** A Laplace Gaussian for proximal inference would need the Hessian
  of the relaxed problem, which the prox gives only through its Jacobian
  ($\nabla^2(-\log p_t) = (D^{-1} - I)/\lambda$ at the prox's output); not built.
- **The two trees are separate even when they could be shared.** For `MixtureProx` of the
  factor's own mixture, both trees hold the same means; training one does not move the other.

## 8. Experiment: backpropagating at a higher noise level

[[Backpropagation through Implicit Inference]] §9 asks whether the adjoint should be taken at
a noise level $t_b$ above the forward's $t_f$: the gradient of a more smoothed problem, biased
but better conditioned, with graduated smoothing as the hoped-for benefit. `prox_pullback`
takes the noise level as its own argument, so the surrogate is one call with `t = t_b`. Away
from $t_f$ the fixed-point equation no longer holds at $w$, so the result is not the gradient of
anything.

**Setup** (`examples/proxdm_backprop_noise.jl`). The task of §7 there: 64
mixture components on the unit circle (width 0.05), means trained by Adam so that proximal
inference "$x \mapsto y$" ($x$ clamped) reproduces $y = x^2 - 0.5$ on 9 training inputs, with
gradients from the adjoint only. `MixtureProx`, $t_f = 0.005$ ($\sigma = 0.027$), $\lambda = 0.01$,
150 epochs; $t_b \in \{0.005, 0.01, 0.02, 0.05, 0.1\}$, i.e. $\sigma(t_b)$ from 0.027 to 0.32.

**The surrogate's direction.** Cosine to the exact gradient, and its norm relative to the exact
one, on the 9 training points:

| $t_b$ | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|
| at the start | 0.98 (0.77) | 0.89 (0.53) | 0.39 (0.30) | 0.28 (0.26) |
| after exact training | 0.98 (0.82) | 0.80 (0.66) | 0.47 (0.79) | 0.34 (0.43) |

Up to $\sigma(t_b) \approx 3\sigma(t_f)$ it points the same way and is shorter (smoother); beyond, the
direction decorrelates.

**Training** (learning rate 0.01, five training sets: the grid and four random draws of 9
inputs; median and range):

| $t_b$ | final loss | loss spikes (+10%) | test RMSE, warm start | test RMSE, cold start |
|---|---|---|---|---|
| 0.005 (exact) | 1.3e-4 [4e-7, 6e-4] | 3 [0, 11] | 0.14 [0.08, 0.30] | 0.30 [0.28, 0.34] |
| 0.01 | 7.5e-5 [5e-6, 3e-4] | 5 [1, 13] | 0.14 [0.07, 0.28] | 0.29 [0.26, 0.30] |
| 0.02 | 1.1e-4 [3e-7, 3e-4] | 2 [0, 8] | 0.14 [0.09, 0.29] | 0.23 [0.15, 0.33] |
| 0.05 | 2.6e-4 [2e-5, 2e-3] | 5 [0, 8] | 0.12 [0.05, 0.31] | 0.28 [0.05, 0.32] |
| 0.1 | 6.6e-4 [3e-4, 1.7e-2] | 2 [0, 3] | 0.16 [0.12, 0.25] | 0.22 [0.10, 0.24] |

At learning rate 0.03 (grid only) the exact adjoint reaches loss 1e-3 fastest (36 epochs, against
43–53 for $t_b$ up to 0.05) and $t_b = 0.1$ never does.

**Reading.** The hypothesis is not supported here. Surrogates up to $t_b = 0.05$ train about as
well as the exact gradient, none consistently better, and $t_b = 0.1$ is too biased: it stalls in
one of five runs. On the grid training set alone, $t_b = 0.05$ looked best (test RMSE 0.05 against
0.08, cold start 0.05 against 0.31) and $t_b = 0.02$ had no loss spikes against eleven; neither
survived the other four sets, which is why the table reports five.

**Why this is not the last word.** The motivation was conditioning, and this problem has none to
fix: one free coordinate, an exact prox, an adjoint solve that is a scalar division. The cases
where a high-$t_b$ adjoint could pay off are a learned prox (whose Jacobian at small $t$ carries
its errors), many free coordinates, and a forward at very small $t_f$, which is where the
stiffness argument applies. The 9-point task also overfits (test RMSE 0.14 at training loss
1e-4), so generalisation differences of this size are noise.

Related: [[implicit]], [[reddiff]], [[backends]], [[analytic]],
[[ProxDM and Proximal Alternatives]], [[Inference Signatures]], [[Deterministic Relaxation]]
