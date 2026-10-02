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
- **No AD dependency.** The input Jacobian has a finite-difference fallback (`epsilon_jacobian`); the parameter VJP is an interface method (`epsilon_vjp_params`) that closed-form predictors implement and a Lux network would implement with AD. A model with no parameters skips it.

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
- training through inference reduces the loss by more than 50×.

Related: [[analytic]], [[reddiff]], [[factor]], [[Deterministic Relaxation]], [[Inference Signatures]],
[[The Implicit Diffusion Factor as a Statistical Game]]
