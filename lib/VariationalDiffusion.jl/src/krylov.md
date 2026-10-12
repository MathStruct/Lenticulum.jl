#implementation

> Matrix-free linear algebra for the implicit solver: GMRES for the Newton step and the adjoint,
> Lanczos for the stability test, conjugate gradients and stochastic Lanczos quadrature for the
> Laplace approximation. With it a query's cost grows with the number of Krylov steps, not with
> the number of free coordinates, which is what generating an image (hundreds of free
> coordinates) needs.

> Sources: code: `krylov.jl`, `implicit.jl` (`_newton_system`, `_adjoint_solve`), `answers.jl` (`_laplace_marginal`); Saad & Schultz, *GMRES*, SIAM J. Sci. Stat. Comput. 1986; Knoll & Keyes, *Jacobian-free Newton–Krylov methods*, J. Comput. Phys. 2004; Ubaru, Chen & Saad, *Fast estimation of tr(f(A)) via stochastic Lanczos quadrature*, SIAM J. Matrix Anal. Appl. 2017
>
> Bibliography: [[Bibliography#^saad1986gmres|Saad & Schultz 1986]] · [[Bibliography#^knoll2004jfnk|Knoll & Keyes 2004]] · [[Bibliography#^ubaru2017slq|Ubaru et al. 2017]]
>
> Theory: [[Implicit Diffusion Learners]] · [[Backpropagation through Implicit Inference]] · [[Open Problems in Implicit Diffusion Learning]]

## 1. Where the solver meets $J_{FF}$

| use | dense (`DenseSolve`) | matrix-free (`KrylovSolve`) |
|---|---|---|
| Newton step $J_{FF}d = -r_F$ | LU of the block | restarted GMRES on $v \mapsto J_{FF}v$ (`prior_jvp`) |
| adjoint $J_{FF}^\top\lambda = -\bar z_F$ | the same LU, transposed | GMRES on $w \mapsto J_{FF}^\top w$ (`prior_vjp`) |
| stability, $\operatorname{sym}J_{FF}\succ0$ | Cholesky test | smallest Ritz value of Lanczos on the symmetric part |
| Laplace covariance on coordinates $I$ | inverse of the whole block | one conjugate-gradient solve per coordinate in $I$ |
| $\log\det\Sigma_{FF}$ (mixture weights) | from the inverse | stochastic Lanczos quadrature |

The precision diagonal $\operatorname{diag}(\rho_F^2)$ is added inside each product. For an energy
network or a closed-form mixture the field is a gradient, so $J$ is symmetric and the symmetric
part costs one product, not two.

## 2. Design decisions

- **One option, threaded through.** `linsolve` is a keyword of `implicit_infer`,
  `implicit_pullback`, `laplace_belief`, `implicit_mixture` (and so `implicit_roots`), and a field
  of `ImplicitProx`. The default stays `DenseSolve()`: exact, and faster while few coordinates
  are free.
- **No Krylov package.** GMRES with Givens rotations, CG and Lanczos with full
  reorthogonalisation are about 150 lines; a dependency would have been heavier than the code.
- **Deterministic.** Lanczos starts and Hutchinson probes come from `Xoshiro(seed)`, so the same
  query gives the same answer, which the implicit function theorem relies on.
- **Exact at small sizes.** With at most `lanczos` free coordinates the Lanczos space is the
  whole space and the stability test is exact; with at most `probes` the log-determinant uses
  unit vectors and is exact too. The tests check both against dense factorisations.
- **JVPs without forward mode.** Zygote has no forward mode, and a JVP through reverse mode
  costs one pullback per coordinate. So reverse-only backends take one central difference along
  the direction (two batched forward passes, about $10^{-8}$ relative error), while ForwardDiff and
  Enzyme use `DI.pushforward`. The adjoint uses VJPs only, so gradients stay exact either way.

## 3. What is approximate

- The smallest Ritz value approaches $\lambda_{\min}$ from **above**: a point whose symmetric
  Jacobian has a small negative eigenvalue can be reported stable when there are more free
  coordinates than Lanczos steps. Newton is then tried where the dense solver would have
  descended; the residual line search keeps that safe, but the `stable` flag is an estimate.
- The log-determinant with Rademacher probes has a standard error of roughly
  $\sqrt{2/\text{probes}}\,\lVert\log S\rVert_F$; the mixture weights inherit it.
- GMRES stops at `rtol` (default $10^{-8}$) or `maxiters`; the adjoint warns when it stops short.

## 4. Measured

The MNIST shape of [[MNIST in Both Directions]]: an MLP $802\to256\to256\to794$, 64 nodes batched
by noise level, Zygote, CPU, the 10 label coordinates clamped and the 784 image coordinates free
(with a unit anchor, since the network is untrained):

| | time |
|---|---|
| one dense $784\times784$ block of $J_{FF}$ | 33 s |
| `implicit_infer` with `KrylovSolve()` (2 Newton iterations, converged, stable) | 1.1 s |
| `implicit_pullback` with `KrylovSolve()` | 0.24 s |
| `laplace_belief` on 10 coordinates | 0.78 s |
| one `prior_jvp` / one `prior_vjp` | 3.2 ms / 5.7 ms |

The Krylov answer agrees with the dense one to $2\cdot10^{-15}$ (relative).

Related: [[implicit]], [[backends]], [[implicit_factor]]
