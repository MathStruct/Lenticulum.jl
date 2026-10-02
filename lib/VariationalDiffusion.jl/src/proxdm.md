#implementation

> Proximal diffusion models: a diffusion prior that is queried through its **proximal operator**
> instead of its score. Implemented here are the interface, an exact oracle, ProxDM's sampler,
> proximal inference for the implicit learner, and the training loss.

> Sources: Fang, Díaz, Buchanan & Sulam, *Beyond Scores: Proximal Diffusion Models*, [arXiv:2507.08956](https://arxiv.org/abs/2507.08956) (Algorithm 1, Eq. 9); Zhu et al., *Denoising Diffusion Models for Plug-and-Play Image Restoration*, CVPR 2023 (DiffPIR, the half-quadratic splitting); code: `proxdm.jl`
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
| `ProxNetwork(model, schedule; input)` | a Lux model in ProxDM's parametrisation $f_\theta(v; t, \lambda) = v - \sqrt\lambda\,\varepsilon_\theta(v; t, \lambda)$; conditioned on **two** scalars |
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

## 5. Implementation difficulties and open ends

- **Non-convex priors make the prox multivalued.** For a mixture, $-\log p_t$ is not convex,
  and $\operatorname{prox}$ can have several minimisers. `MixtureProx` uses Newton where
  $I - \lambda\nabla^2\log p_t$ is positive definite and a gradient step otherwise, and returns
  the local minimiser reached from $v$. A learned `ProxNetwork` makes the same choice
  implicitly, wherever training put it.
- **The PDA step-size limit.** $\gamma_k < 2$ fails for coarse grids near $t = T$ under the
  default VP schedule. The hybrid scheme removes it; PDA throws with a message saying so
  instead of returning garbage.
- **No adjoint for `prox_infer` yet.** The fixed point satisfies
  $u = \operatorname{prox}(w(u))$, so the implicit function theorem applies just as for
  `implicit_infer`. It needs the prox's input Jacobian and parameter VJP, which is why
  `ProxNetwork` would get an `ad` field when this is added ([[backends]]).
- **`stable` is not assessed** by the splitting; it reports `true` when converged.
- **A `ProxDM` factor inversion** (alongside `REDDiff` and `ImplicitProx` in `factor.jl`) is the
  natural next step and would make ProxDM usable inside a factor graph. It is the first case
  with two parameter trees (the prior's $\varepsilon_\theta$ and the prox network), the
  structure [[ProxDM and Proximal Alternatives]] §4 anticipated.

Related: [[implicit]], [[reddiff]], [[backends]], [[analytic]],
[[ProxDM and Proximal Alternatives]], [[Inference Signatures]], [[Deterministic Relaxation]]
