#implementation

> Implicit inference and its backward pass for a diffusion model: a deterministic residual field,
> a solver that only reports roots it trusts, and an adjoint that turns one linear solve into
> gradients for parameters, inputs, anchors and precisions. The theory is in
> [[Implicit Diffusion Learners]] and [[Backpropagation through Implicit Inference]].

> Sources: code: `implicit.jl`
>
> Theory (CT-ML wiki): [Least Fixed Point](https://mathstruct.org/CategoryTheory-ML-Wiki/Least-Fixed-Point) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category)

## 1. What is in the file

| name | is |
|---|---|
| `FieldNodes`, `field_nodes`, `noisefree_nodes` | the fixed $(t_k, \varepsilon_k, w_k)$ that make the field deterministic |
| `ImplicitDiffusion` | the relation: a `NoisePredictor` plus a node set plus λ |
| `prior_field`, `prior_jacobian` | $g_\theta(z)$ and $\partial g/\partial z$ |
| `implicit_residual` | $r = g + \rho^2\odot(z - z_0)$ on the free coordinates |
| `implicit_infer` → `ImplicitSolution` | a root, with `residual`, `iters`, `converged`, `stable` |
| `implicit_pullback` | the adjoint: `(z₀ = …, ρ = …, ps = …)` |

## 2. Design decisions

- **Deterministic by construction.** RED-Diff redraws $(t, \varepsilon)$ every step; here the nodes are fixed, so inference is a function and the implicit function theorem applies. `field_nodes` uses antithetic pairs so that the $-\varepsilon$ control variate cancels exactly.
- **Polarity as a precision vector**, the same encoding as `precision_vector(f, polarity)`: `Inf` is a hard clamp, imposed by projection, never as a penalty.
- **The solver trusts Newton only where it should.** A Newton step is taken when the symmetric part of $J_{FF}$ is positive definite, with backtracking on $\lVert r_F\rVert$; otherwise a fixed descent step along the field. Newton is attracted to every root, saddles and maxima included; restricting it to locally convex regions keeps it on stable roots.
- **Two flags, not one.** `converged` (the residual is small) and `stable` ($\operatorname{sym}J_{FF}\succ0$) are separate, because a converged unstable root is on the zero set but not in the relation one wants.
- **The pullback refuses non-converged states.** The IFT says nothing about a non-root, and silently returning a gradient there is what made the first prototype diverge.
- **No AD dependency.** The input Jacobian has a finite-difference fallback (`epsilon_jacobian`); the parameter VJP is an interface method (`epsilon_vjp_params`) that closed-form predictors implement exactly and a Lux network gets from the AD backend in its `ad` field ([[backends]]). A model with no parameters skips it.

## 3. Implementation difficulties

1. **The tolerance floor.** The field of a narrow mixture at small $t$ has a floating-point noise floor around $10^{-11}$ (it scales like $1/\sigma_t$). An absolute tolerance of $10^{-11}$ made a third of the solves "fail"; the tolerance is relative, `tol * (1 + ‖z‖)`.
2. **The prototype's solver.** A version with a fixed descent step in non-convex regions and an energy line search dropped about 60% of solves during training, and its training stalled at RMSE 0.15. The package's solver (Newton where convex, residual backtracking) dropped none and reached 0.012.
3. **Smoothing levels.** With RED-Diff's training range of noise levels, the deterministic field of a circle has its only root at the centre. The default `levels` stop at $t = 0.05$; the trade-off is measured in [[Implicit Diffusion Learners]] §5.
4. **Dense Jacobians.** `prior_jacobian` forms $J$ densely, which is fine for the small $n$ of factor blocks but not for images. Large $n$ needs a matrix-free GMRES in both the Newton step and the adjoint, each iteration costing one VJP per node.

## 4. How it is tested

`test/runtests.jl`, against the closed-form mixture of [[analytic]] and the Gaussian oracle:
- the field equals $\nabla\Phi$ and its Jacobian is symmetric;
- both branches of the circle are found, and they merge near the branch point;
- the smoothing bias is monotone in the levels and matches $R - (s^2+\sigma^2/\alpha^2)/(2R)$;
- noise-free roots are Tweedie fixed points;
- the Gaussian root and its derivative match the closed form;
- every adjoint cotangent matches finite differences of the full solve;
- training through inference reduces the loss by more than 50×;
- `implicit_roots` finds both branches of the circle, and the Laplace covariance of a Gaussian
  query equals the closed form $1/(\kappa + \rho^2)$.

## 5. All answers, and how sure (`answers.jl`)

Two functions build on `implicit_infer` without changing it.

**`implicit_roots`** solves from `z₀` and from perturbed starts (normal on the free coordinates,
the clamped ones fixed), keeps converged and stable answers, and removes duplicates. For an
energy-parametrised predictor the answers are ordered by the query's energy. On the circle,
"$y$ given $x = 0.6$" returns both branches; near $x = \pm 1$ they merge into one. It finds
what its starts reach: no guarantee of completeness
([[Open Problems in Implicit Diffusion Learning]] T1).

**`implicit_laplace`** inverts the symmetric part of $J_{FF}$, the residual's Jacobian on the
free coordinates at the answer, which is the curvature of the query's energy. The subtle part
is the units. With $x_k = \alpha_k z + \sigma_k\varepsilon_k$ and $\varepsilon_\theta = -\sigma_t\nabla_x\log p_t$,

$$
g(z) = \nabla_z\Bigl[\sum_k w_k\,\lambda\,\frac{\sigma_k^2}{\alpha_k^2}\bigl(-\log p_{t_k}(x_k)\bigr)\Bigr] + (\text{terms linear in } z),
$$

so the prior term is a sum of smoothed negative log-densities with weights
$\lambda\sigma_k^2/\alpha_k^2$. They sum to one exactly when $\lambda = 1/\sum_k w_k\sigma_k^2/\alpha_k^2$,
`density_lambda(schedule, nodes)`, and then the energy is a proper negative log-density in
the units of the clamp's Gaussian likelihood, and the covariance is in data units. The
linear terms come from the noise draws and cancel for antithetic nodes. For queries with only
hard inputs and free outputs, λ scales the curvature and leaves the answers unchanged.

On the circle with this λ, the standard deviation of $y$ is 0.135 at $x = 0$ (the smoothed
ring's width: data width 0.05 combined with the noise levels gives about 0.12), 0.17 at
$x = 0.6$ (the vertical line crosses the ring obliquely), and it grows towards the branch
points (0.31, then 0.41) and off the circle (0.53). Calibration is to the density **smoothed
at the field's noise levels**, so it is wider than the data's own spread, the same smoothing
that biases the answers inwards ([[Implicit Diffusion Learners]] §5).

**`laplace_belief`** returns that covariance as a `GaussianBelief` (marginalised onto chosen
free coordinates), and **`implicit_mixture`** combines all answers into a `MixtureBelief`. Its
weights are Laplace estimates of each answer's probability mass,
$w_k \propto e^{-E(z_k)}\,\det(\Sigma_k)^{1/2}$, with $E$ the query's energy. That needs an
energy: energy networks have one, and so does the closed-form mixture, whose energy is the
same formula with $E_\theta = -\log p_t$ in closed form (so ordering in `implicit_roots` uses it
too). With $x = 0.3$ on a circle whose upper half carries three times the data, the upper
answer gets weight 0.78. The unweighted circle gives 0.54 rather than 0.5, because the field's
random noise draws are not symmetric; the ratio of the two odds is 3.0. For plain ε-networks
the weights are equal.

What is still missing: λ chosen for calibration is not always the λ one wants for balancing soft
evidence.

Related: [[analytic]], [[reddiff]], [[factor]], [[energy_network]], [[Deterministic Relaxation]], [[Inference Signatures]],
[[The Implicit Diffusion Factor as a Statistical Game]]
