#derivation #theorem #algorithm

> The Lagrangian of the original sketch, carried through: state equation, adjoint equation and
> gradient for an implicit diffusion learner. Also covered: the hard/soft clamp split, what the
> adjoint costs, why **no noise has to be stored**, and what to do when inference has not
> reduced the energy. Every gradient below is checked against finite differences of the full
> solve in `lib/VariationalDiffusion.jl/test/runtests.jl`.

> Sources: original to this vault (design and analysis); the adjoint method as in [[Backpropagation by the Implicit Function Theorem]] and [[DEQ as a Relation]] §4; Bai, Kolter & Koltun, *Deep Equilibrium Models*, NeurIPS 2019; Fung et al., *JFB: Jacobian-Free Backpropagation for Implicit Networks*, AAAI 2022; Mardani et al. [arXiv:2305.04391](https://arxiv.org/abs/2305.04391); code: `implicit.jl` (`implicit_pullback`), `analytic.jl` (`epsilon_jacobian`, `epsilon_vjp_params`)
>
> Bibliography: [[Bibliography#^bai2019deq|Bai et al. 2019]] · [[Bibliography#^fung2022jfb|Wu Fung et al. 2022]] · [[Bibliography#^mardani2024reddiff|Mardani et al. 2024]]
>
> Theory (CT-ML wiki): [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Compiler Correctness](https://mathstruct.org/CategoryTheory-ML-Wiki/Compiler-Correctness)

## 1. Setup

From [[Implicit Diffusion Learners]]: free coordinates $F$ (finite precision), clamped
coordinates $C$ (precision $\infty$), and the residual on $F$

$$
r_F(z; z_0, \rho, \theta) \;=\; g_\theta(z)_F + \rho_F^2 \odot (z_F - z_{0,F}),
\qquad
g_\theta(z) = \sum_k w_k\lambda_{t_k}\bigl(\varepsilon_\theta(\alpha_k z + \sigma_k\varepsilon_k, t_k) - \varepsilon_k\bigr),
$$

with the noise nodes $(t_k, \varepsilon_k, w_k)$ **fixed** (sample-average approximation; why this
matters is §5). Inference returns $z^\star$ with $r_F(z^\star) = 0$ and $z^\star_C = z_{0,C}$. Its
Jacobian on the free block is

$$
J_{FF} \;=\; \partial_{z_F} r_F \;=\; \Bigl[\sum_k w_k\lambda_{t_k}\alpha_{t_k}\,\partial_x\varepsilon_\theta(x_k, t_k)\Bigr]_{FF} + \operatorname{diag}(\rho_F^2),
\qquad x_k = \alpha_k z^\star + \sigma_k\varepsilon_k .
$$

A downstream loss $\ell(z^\star)$ (e.g. $\ell = \tfrac12\lVert P_y z^\star - y_{\text{true}}\rVert^2$) has
cotangent $\bar z = \partial\ell/\partial z^\star$.

## 2. The Lagrangian

$$
\mathcal L(z_F, \lambda;\, z_0, \rho, \theta) \;=\; \ell(z) + \langle \lambda,\, r_F(z; z_0, \rho, \theta)\rangle,
\qquad z = (z_F, z_{0,C}).
$$

### State equation: $\partial\mathcal L/\partial\lambda = 0$

$$
r_F(z^\star; z_0, \rho, \theta) = 0 .
$$

This is the inference problem: solve it first, with any of the point signatures of
[[Inference Signatures]].

### Adjoint equation: $\partial\mathcal L/\partial z_F = 0$

$$
J_{FF}^\top \lambda \;=\; -\,\bar z_F .
$$

One linear solve with the **transpose** of the free-block Jacobian. For an exact score $J_{FF}$ is
symmetric ([[Implicit Diffusion Learners]] §3), so the transpose is irrelevant; for a learned
network it is not symmetric, and the transpose matters. Solve it densely for small $n$ (form
$J_{FF}$ from $n$ VJPs) or matrix-free with GMRES for large $n$; each GMRES iteration costs one
VJP of $\varepsilon_\theta$ per node.

### Gradient: read everything off $\lambda$

By the implicit function theorem, $d\ell = \bar z_C\,dz_{0,C} + \lambda^\top \partial r_F$ along every
input direction. Embedding $\lambda$ in $\mathbb R^n$ with zeros on $C$:

| w.r.t. | gradient | needs |
|---|---|---|
| network parameters $\theta$ | $\displaystyle\sum_k w_k\lambda_{t_k}\,\bigl(\partial_\theta\varepsilon_\theta(x_k, t_k)\bigr)^{\!\top}\lambda$ | one parameter-VJP per node |
| hard-clamped inputs $z_{0,C}$ | $\bar z_C + J_{FC}^\top\lambda$ | the off-diagonal Jacobian block |
| soft targets / anchors $z_{0,F}$ | $-\rho_F^2 \odot \lambda$ | nothing extra |
| precisions $\rho_F$ | $2\rho_F \odot (z^\star_F - z_{0,F}) \odot \lambda$ | nothing extra |

The last row means **the polarity weights are learnable**. Soft-clamp precisions can be trained
by the same adjoint, so "how much to trust an input" becomes a parameter rather than a
hyperparameter.

`implicit_pullback(m, sol, z₀, ρ, z̄, ps, st)` returns `(z₀ = …, ρ = …, ps = …)`. In the
backpropagation signature of the sketch it is

$$
\Delta Y \;\mapsto\; (\Delta X, \Delta\Theta, \Delta\rho),
$$

and the incoming energy does not need its own cotangent (§6).

## 3. Why this is the right derivative

The adjoint gives the **exact** derivative of the map $(z_0, \rho, \theta) \mapsto z^\star$ defined
by the deterministic residual, wherever $J_{FF}$ is invertible. Checked against central finite
differences of the full solve (tolerance $10^{-12}$), on a circle relation with exact score:

| derivative | adjoint | finite difference |
|---|---|---|
| $d\ell/dx$ (hard input) | −0.06184743 | −0.06184742 |
| $d\ell/d\mu_{2,10}$ (a mixture mean, i.e. $\theta$) | +0.02653255 | +0.02653255 |
| $d\ell/dz_{0,x}$ (soft input) | +0.02502245 | +0.02502245 |
| $d\ell/d\rho_y$ | +0.01145560 | +0.01145560 |
| $d\ell/d\rho_x$ | −0.00007650 | −0.00007650 |

For a Gaussian prior everything is linear: $y^\star = \rho^2 y_0/(\kappa + \rho^2)$ and
$dy^\star/dy_0 = \rho^2/(\kappa + \rho^2)$, matched to $10^{-6}$.

## 4. The cost

Per backward pass: one Jacobian evaluation (or one GMRES solve), and one parameter-VJP of
$\varepsilon_\theta$ per noise node. No unrolling, so **memory is independent of the number of
inference iterations**, the defining advantage of implicit differentiation. With $K$ nodes this
is $K$ network VJPs, the same as one step of RED-Diff with $K$ samples, done once rather than
once per iteration.

## 5. Does the noise from inference have to be stored?

**No**, and the reason is structural. There are three cases:

1. **Fixed nodes (what the package does).** The noise $\varepsilon_k$ is part of the *model*, a fixed
   quadrature of the expectation. The backward pass re-evaluates the Jacobians at $z^\star$ with
   the same nodes; nothing from the forward trajectory is needed except $z^\star$.
2. **Fresh noise in inference (RED-Diff as published).** The forward pass is a stochastic
   approximation whose limit is a root of the *expected* field
   $\bar g(z) = \mathbb E_{t,\varepsilon}[\dots]$. The IFT applies to $\bar g$, at the limit point.
   The trajectory's noise is irrelevant. The backward pass needs unbiased estimates of
   $\partial_z\bar g$ and $\partial_\theta\bar g$, which fresh noise provides.
3. **The catch in case 2.** $\lambda = -J^{-\top}\bar z$ involves an *inverse* of an expectation, and
   $\mathbb E[\hat J]^{-1} \ne \mathbb E[\hat J^{-1}]$. Plugging a noisy Jacobian into the solve gives
   a biased gradient. Fix: use many nodes in the backward pass, or the same fixed node set for
   the Jacobian and the residual. Fixing the nodes is the simplest way to make the whole
   thing exact.

So save $z^\star$, not the noise. If the noise is fixed it is part of $\theta$'s computational
graph anyway.

## 6. When the energy is not reduced

The IFT is a statement about a **root**. At a point that is not a root it computes the derivative
of nothing in particular. `implicit_pullback` therefore refuses non-converged solutions. The
options, from most to least principled:

| situation | what to do |
|---|---|
| converged, stable | the adjoint (§2) is exact |
| converged, **unstable** (saddle or maximum of $\mathcal E$) | do not backpropagate; the point is not in the relation. Restart inference, or anchor with $\rho_y > 0$ |
| not converged, residual small | continue iterating from the current state (warm start), then backpropagate |
| not converged, budget exhausted | backpropagate through the **unrolled** last $k$ steps (exact for the truncated map), or use **Jacobian-free** backpropagation (replace $J_{FF}^{-1}$ by the identity, as in JFB) as a descent direction |
| no root at all (input outside the relation) | inference returns the nearest ridge point; the residual reports the distance |

The loss-side question ("what if inference *increased* the energy?") does not arise for the
solver here: Newton steps are only taken where $\operatorname{sym}(J_{FF}) \succ 0$, with
backtracking on $\lVert r\rVert$. Elsewhere the step is a descent step on $\mathcal E$ when the field is a
gradient. A learned non-conservative field has no energy to decrease, only a residual. That is
why convergence is defined on $\lVert r_F\rVert$ and not on $\mathcal E$.

Observed in training (§7): the first prototype dropped about 60% of solves as non-converged,
and its gradients blew the model up until those were excluded. With the stability guard and
the Newton/descent switch, the package's solver dropped none.

## 7. Learning a relation by backpropagating through inference

The experiment that closes the loop. Start from the circle, a ring of 64 mixture components
whose means are $\theta$. Pairs $(x_i, y_i)$ come from a *different* curve, $y = x^2 - 0.5$. Train
$\theta$ by Adam on $\tfrac12(y^\star(x_i;\theta) - y_i)^2$, where $y^\star$ is the inferred output, with
gradients from the adjoint only and no denoising loss at all.

| | before | after 150 epochs |
|---|---|---|
| training loss (mean $\tfrac12$ squared error) | $1.8\times10^{-1}$ | $1.0\times10^{-5}$ |
| test RMSE, warm start | 0.59 | **0.012** |
| test RMSE, cold start $y_0 = -0.5$ | 0.59 | 0.26 |
| dropped solves | — | 0 |

The cold-start number is one test point ($x = 0.55$) landing on the **upper arc of the old
circle**, which training never touched. Bilevel training reshapes the branch that inference
visits and leaves the others. An implicit learner trained this way is a relation with
*several* branches, and inference returns the branch of its starting basin. To fit a whole
relation, either train generatively as well (the statistical-game semantics of
[[The Implicit Diffusion Factor as a Statistical Game]]), or anchor the output ($\rho_y > 0$) at an
amortised guess.

## 8. Where AD is needed, and where it is not

| quantity | exact score (closed form) | learned network |
|---|---|---|
| $g_\theta(z)$ | forward pass | forward pass |
| $\partial_x\varepsilon_\theta$ (inference Newton steps, adjoint solve) | closed form | input JVP/VJP, or finite differences for small $n$ |
| $(\partial_\theta\varepsilon_\theta)^\top\lambda$ (parameter gradient) | closed form | **reverse-mode AD**: one VJP per node |

`VariationalDiffusion` has no AD dependency of its own: `epsilon_jacobian` has a
finite-difference fallback, and `epsilon_vjp_params` is an interface method that closed-form
predictors implement exactly. A Lux network gets both from the backend named in its predictor,
`NoisePredictor(…; ad = AutoZygote())` (or Enzyme, ForwardDiff, Mooncake, Reactant), through a
package extension ([[backends]]). A trained 5k-parameter MLP on the circle reproduces the
finite-difference derivatives of this section to 8 digits (`examples/circle_mlp.jl`).
Compare [[RED-Diff as a Statistical Game]] §3, where the stop-gradient was "the difference
between a package with an AD dependency and one without". Training *through* inference brings
back exactly one VJP per node, no more.

Related: [[Implicit Diffusion Learners]], [[Inference Signatures]], [[Deterministic Relaxation]],
[[The Implicit Diffusion Factor as a Statistical Game]], [[DEQ as a Relation]],
[[Backpropagation by the Implicit Function Theorem]], [[implicit]], [[backends]]
