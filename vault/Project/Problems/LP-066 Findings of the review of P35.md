#open-problem

> **Status: resolved (2026-11-19).** The review of P35 (`MixtureProx` takes a modified Newton step
> where $I - \lambda\nabla^2\log p_t$ is indefinite) found two cases the change made worse than before:
> NaN derivatives under ForwardDiff, and a convergence failure that turned out to come from the
> roundoff guard of the positive definite path, which predates P35. Both fixed, with regression tests.
> Related: [[Feature Backlog]] (P35), [[LP-018 Defects found in the review of the older packages|LP-018]],
> [[lib/VariationalDiffusion.jl/src/proxdm|proxdm]] "P35 indefinite curvature".

> Sources: implemented by the maintainer's session (Claude); reviewed by Codex (GPT, `codex exec -s
> workspace-write` in the main checkout, told not to edit the repository, probes in the session's
> scratch directory; one run). Brief, diff, answer and probes: `../Lenticulum.jl-Review-2026-11-19/p35/`.
> code: `lib/VariationalDiffusion.jl/src/proxdm.jl` (`_modified_newton_direction`, `proximal(::MixtureProx, …)`);
> test: `lib/VariationalDiffusion.jl/test/p35_mixture_prox.jl`.

**Problem.** The requirement given to the reviewer: `proximal(::MixtureProx, …)` returns a point
whose optimality residual is within `tol·(1+‖u‖)` and where $K = I - \lambda\nabla^2\log p_t(u)$ is positive
definite, or throws; the case of the full-size example that threw "exceeded maxiters=100" must
succeed; results where $K$ stays positive definite along the path stay bit for bit unchanged.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | ForwardDiff through `proximal` gives an all-NaN Jacobian (3D mixture, means $(\pm1, 0, 0)$, $s = 0.5$, $t = 0$, $\lambda = 1$): `eigen` of a dual matrix with a repeated eigenvalue | yes, reproduced (`codex-probes/dual3.jl`: old finite, new NaN) | the eigendecomposition of the modified step is taken on `stop_gradient(K)`; the answer's derivative comes from the last iterations, Newton steps on a positive definite $K$ with dual entries | the ForwardDiff lines of `p35_mixture_prox.jl`: finite, and equal to the exact `prox_jacobian` to 1e-8 |
| 2 | A weighted 2D mixture ($\lambda = 100$, $t = 0$, weights down to $e^{-34}$) now exceeds `maxiters = 100`; the old solver converged | yes, reproduced (`codex-probes/convergence.jl`). Mechanism, found by the session: both solvers reach the same minimiser, but the new one arrives at a point with residual 1e-8 where the full Newton step (to 2e-14) raised the objective by 2.8e-15, above the allowance `8eps·(1+|f0|)` ≈ 2.2e-15 of the LP-018 roundoff guard; $f_0 = -0.23$ is the small difference of terms of size about 50 ($\lambda\cdot\tfrac12\sum\log 2\pi v$), whose rounding is about 1e-14. The guard was too tight before P35; the old path did not hit it here | the allowance is `8eps·(1 + |f0| + s(u))`, with `s(u)` the size of the objective's terms: $\tfrac12\lVert u-v\rVert^2 + \lambda(\lvert\max_j\ell_j\rvert + \lvert\tfrac12\sum\log 2\pi v\rvert)$. Still used only where $K$ is positive definite and the recomputed residual decreases | the weighted-mixture lines of `p35_mixture_prox.jl` (residual and definiteness at the answer) |

Finding 2 makes the requirement "bit for bit unchanged where $K$ stays positive definite" hold only
up to the roundoff guard: a Newton step near the minimum that the old allowance refused can now be
accepted. The answers then differ within `tol`.

Not checked by the reviewer: the sampler, `prox_infer`, the factor; Float32 inputs (asked, not reported on).
