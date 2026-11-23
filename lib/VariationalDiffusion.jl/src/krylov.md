#implementation

> Matrix-free linear algebra for the implicit solver: GMRES or MINRES for the Newton step and the adjoint,
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
| Newton step $J_{FF}d = -r_F$ | LU of the block | GMRES on $J_{FF}v$ (`prior_jvp`) or MINRES on $J_{FF}^\top v$ (`prior_vjp`) |
| adjoint $J_{FF}^\top\lambda = -\bar z_F$ | shared dense solve on the transposed block | GMRES or MINRES on $J_{FF}^\top w$ (`prior_vjp`) |
| stability, $\operatorname{sym}J_{FF}\succ0$ | Cholesky test | smallest Ritz value of Lanczos on the symmetric part |
| Laplace covariance on coordinates $I$ | inverse of the whole block | one conjugate-gradient solve per coordinate in $I$ |
| $\log\det\Sigma_{FF}$ (mixture weights) | from the inverse | stochastic Lanczos quadrature |

The precision diagonal $\operatorname{diag}(\rho_F^2)$ is added inside each product. For an energy
network or a closed-form mixture the field is a gradient, so $J_{FF}$ is symmetric and the
symmetric part costs one product, not two; otherwise it averages the JVP and the VJP. This
holds at any block times: unequal times give the predictor's Jacobian unequal row scales
$\operatorname{diag}(\sigma)H\operatorname{diag}(\alpha)$ (LP-018), but the field weights
coordinate $i$ by $c\,\alpha_i/\sigma_i$, so each node adds $c\operatorname{diag}(\alpha)H\operatorname{diag}(\alpha)$
([[energy_network]] §3; `test/p37_minres.jl` checks it for an energy and for the mixture ring at
unequal levels).

## 2. Design decisions

- **One option, threaded through.** `linsolve` is a keyword of `implicit_infer`,
  `implicit_pullback`, `laplace_belief`, `implicit_mixture` (and so `implicit_roots`), and a field
  of `ImplicitProx`. The default stays `DenseSolve()`: exact, and faster while few coordinates
  are free.
- **Shared linear solvers.** `method = :auto` selects MINRES when the model is an
  `ImplicitDiffusion` whose predictor is an `EnergyNetwork` or a `GaussianMixtureEps`
  (`_symmetric_block`), at any block levels, and GMRES otherwise. `method = :minres` requires
  the same and `method = :gmres` always selects GMRES. P37's design first restricted MINRES to a
  common noise time on the free coordinates; Codex's test showed the field symmetric at unequal
  levels, and the restriction was dropped.
  MINRES applies the VJP in both solves. It uses the exact adjoint action and avoids the
  loss of symmetry that a finite-difference JVP can introduce on reverse-only backends.
  Both solvers take `maxiters` and use the true residual bound $\text{rtol}\,\lVert b\rVert$.
  Only GMRES uses `restart`. The trust-region step uses Steihaug CG with either setting.
  VariationalDiffusion owns CG, Lanczos and the log-determinant estimate. See [[Shared Solver Layer]].
- **Deterministic.** Lanczos starts and Hutchinson probes come from `Xoshiro(seed)`, so the same
  query gives the same answer, which the implicit function theorem relies on.
- **Exact at small sizes.** With at most `lanczos` free coordinates the Lanczos space is the
  whole space and the stability test is exact; with at most `probes` the log-determinant uses
  unit vectors with full Lanczos depth (`n` steps), overriding a smaller `lanczos` setting,
  and is exact too. Unit probes alone remove trace-sampling error, not truncation error. The tests check both against dense factorisations.
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
- The linear solvers use `rtol` (default $10^{-8}$) and a budget of `maxiters` steps.
  The adjoint warns when the solver stops short. MINRES assumes symmetry, so only the
  true residual can catch a wrong iterate produced on a nonsymmetric operator.

## 4. Measured

### MINRES against GMRES on an energy (P37, 2026-11-23)

`bench/minres_energy.jl`: an untrained `EnergyNetwork` MLP $(n+4)\to w\to w\to1$ (swish, Float64),
`noisefree_nodes(n)`, batched, Zygote, one BLAS thread; the system $J_{FF}^\top x = b$ applied by
the VJP, `rtol` $10^{-8}$, `atol` 0, at most 2000 steps. At $\rho = 0$ the block is indefinite
(at $n = 64$: 29 of 64 eigenvalues negative, condition number 61; at $n = 128$: 59 of 128,
condition number 621; `~/.cache/lenticulum/p37/measure.jl`).

| $n$ ($w$) | $\rho$ | GMRES(40): reason, products, true rel. residual, time | MINRES: reason, products, true rel. residual, time |
|---|---|---|---|
| 64 (256) | 0 | converged, 1023, $9.9\cdot10^{-9}$, 7.8 s | converged, 92, $6.1\cdot10^{-9}$, 0.73 s |
| 128 (256) | 0 | maxiters, 2051, 0.17, 19.0 s | converged, 323, $6.9\cdot10^{-9}$, 2.9 s |
| 128 (256) | 1 | converged, 5, $3.9\cdot10^{-11}$ | converged, 5, $3.9\cdot10^{-11}$ |
| 256 (512) | 0 | maxiters, 2051, 0.099, 78.3 s | converged, 956, $9.6\cdot10^{-9}$, 35.6 s |

Restarted GMRES loses its basis every 40 steps and stalls on these indefinite blocks; MINRES keeps
three vectors and converges. On a positive definite block the two are the same. End to end
(`implicit_infer` from a random start, 8 outer steps) the untrained network has no root near the
start at $\rho = 0$ under either method, and at $\rho = 1$ one Newton or trust-region step
converges under both, so the end-to-end times do not separate the methods (same table in the
benchmark's output); a trained model at a saddle was not measured.

### Earlier measurement

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

## LP-018 exact branch depth (2026-10-29)

Unit-vector probes remove sampling error but cannot remove Lanczos truncation. The exact
branch (`n <= probes`) now takes full depth `n` even if `lanczos` was smaller; stochastic
branches keep the requested depth. Evidence: `test/lp018.jl`, “LP-018.12 exact unit-vector
logdet uses full depth”, with `S = [2 1; 1 2]`, `lanczos = 1`, `probes = 2`: the reference
is `log(3)`, from the independent determinant, rather than `2log(2)`.
