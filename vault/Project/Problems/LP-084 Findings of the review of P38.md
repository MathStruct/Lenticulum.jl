#problem

> **Status: closed (2026-11-23).** P38 (tangents through the shared Krylov solvers at an exact
> breakdown; Codex) and what its review found. Mistral's eighteen probes found one defect, in the scale
> of the tangent check: with the tangent only in $A$ and `atol = 0`, every solve ran a correction.
> It is fixed with a regression test. Related: [[LP-083 Findings of the review of P37|LP-083]] (where the
> limit was found), [[Shared Solver Layer]], [[Feature Backlog]] (P38).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; brief
> `~/.cache/lenticulum/codex-queue/p38-brief.md`, answer `../Lenticulum.jl-Review-2026-11-23/p38-breakdown/IMPL-p38-breakdown.md`,
> scripts in `scratch/p38/` of `../Lenticulum.jl-codex`). Mistral Large (`vibe --agent review`, eighteen
> probes, finished answer file: `scratch/p38r/ANSWER.md` of `../Lenticulum.jl-mistral`, copied to
> `../Lenticulum.jl-Review-2026-11-23/p38-breakdown/ANSWER-mistral.md`). code:
> `lib/ImplicitSolvers.jl/ext/ImplicitSolversForwardDiffExt.jl` (`_refine`, `_meets_tolerance`, `_gain`),
> `lib/ImplicitSolvers.jl/src/linsolve.jl` (`_tangent_refine`, `_krylov_result`). tests:
> `lib/ImplicitSolvers.jl/test/linsolve.jl` (the testsets named P38, LP-083 and LP-084), `lib/ImplicitSolvers.jl/test/device.jl`.

## What P38 does

After a GMRES or MINRES solve has converged on dual numbers, each partial direction's tangent residual
$\dot r_i = \dot b_i - \dot A_i x - A\dot x_i$ is checked against the same `Tolerance`; a failing
direction is corrected by a recursive `linsolve` on the primal operator, $A\delta_i = \dot r_i$. Nested
duals recurse one level at a time. Codex also fixed two places where a breakdown produced NaN tangents
before the check could run (complex GMRES's rotation, `_gmres_hypot`; MINRES's scaled dual norm).
Codex's comparison against `077e8f5a` (`scratch/p38/compare.jl`): 36 generic dual and 48 primal solves
bitwise equal in values, partials, iterations, `nfe` and status; 12 complex GMRES breakdowns had NaN
tangents before and match `DenseSolve` now.

One behaviour change that is not at a breakdown: in LP-082's restarted complex GMRES test the tangent
residual already missed the tolerance before P38 ($3.6\cdot10^{-10}$ against a bound of
$3.85\cdot10^{-12}$; Codex's `scratch/p38/lp082-before.log`), so a correction now runs there and the
tangents meet the tolerance ($8.9\cdot10^{-13}$). Under ForwardDiff a caller can therefore see more
`iterations` and `nfe` than before wherever the tangents carried along by the recurrence did not reach
the tolerance; the primal values never change.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| — | no defect: invariant subspaces of dimension 1–3, breakdowns in later restart cycles, some of several partials needing correction, a correction system itself at a breakdown, nested duals (outer, inner, only the mixed coefficient), complex Hermitian and non-Hermitian, `Float32`, tangents of $10^{-12}$ and $10^{8}$ with `atol = 0`, budgets, `nfe` against counting closures, preconditioners, the adjoint forms, JLArrays (probes 1–9); 80 random breakdowns with no converged report carrying a wrong derivative (probes 10, 17) | — | — |
| F1 | tangent only in $A$ (plain $b$), `atol = 0`: the check's scale $\max(\lVert\dot b_i\rVert, \lVert\partial_i(Ax)\rVert)$ collapses to $\lVert\dot r_i\rVert$, since $\partial_i(Ax) = -\dot r_i$ when $\dot b_i = 0$, so the check fails on every solve and a correction runs on the generic path: 12 iterations and 16 products instead of 6 and 8 (`probe18.jl`); the derivative itself right | right; reproduced by the new test before the fix (4 failures: iterations and `nfe`, both solvers) | fixed: the scale also takes $\lVert Ax\rVert/\lVert x\rVert\cdot\lVert\dot x_i\rVert$, an estimate of $\lVert A\dot x_i\rVert$ (the tangent system's right-hand side at the solution) from quantities at hand, no product. At a breakdown $\dot x_i$ is wrong (zero in the cases seen) and the check still fails as it must. Test "P38 tangent only in A, atol = 0: no correction on the generic path (LP-084, Mistral's F1)"; `probe18.jl` gives 6/8 after the fix; probes 1–8, 10, 17 rerun on the fix with the same results |
| O1 | probe 10's three "BAD" rows (derivatives far off) | not a defect: they are the cases whose correction ran out of `maxiters` (restart 1 or 2 stagnating) and report `:maxiters`, as the requirement allows; probe 17 classifies the same sweep by status (0 converged-but-wrong) | none |
| O2 | after one correction with an $A$-only tangent at a breakdown, the check repeated at the returned $x$ would fail again (one correction reduces $\dot r_i$ to $\text{rtol}$ times its old size) | right as a description, not a defect: the design runs one check and one correction per direction; the correction solve itself meets the tolerance on its own right-hand side | none |

## Gates

ImplicitSolvers (with the fix): passed, the new testset 8/8. MyceliumLearning and Lenticulum (both
tiers, `LENTICULUM_SLOW_TESTS=true`) passed on Codex's branch before the fix, which only removes
corrections; VariationalDiffusion and ImplicitLayers passed with the fix. Codex ran both documentation
builds on its version. Not run: device tests on a GPU, the `krylov_oracle.jl` file by Mistral (it ran in
the suite), triple-nested duals, `Float32` duals on JLArrays.
