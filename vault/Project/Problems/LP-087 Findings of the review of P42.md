#problem

> **Status: closed (2026-11-24).** P42 (`KrylovJL` under ForwardDiff: the primal system solved by Krylov.jl,
> the partials by P38's tangent correction; Codex) and what its review found. Mistral found one defect: a
> dual carried only by a preconditioner, with the operator and the right-hand side plain, threw a
> `MethodError`. It was right and is fixed, with a regression test. The same case with a matrix preconditioner also fails in the in-house GMRES; that is older
> than P42 and recorded as `@test_broken`. Related: [[LP-086 Findings of the review of P41|LP-086]],
> [[Shared Solver Layer]], [[Feature Backlog]] (P40, P42).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; brief
> `~/.cache/lenticulum/codex-queue/p42-brief.md`, answer
> `../Lenticulum.jl-Review-2026-11-24/p42-krylov-dual/IMPL-from-codex-p42-krylov-dual.md`, logs in
> `scratch/p42/` of `../Lenticulum.jl-codex`). Mistral Large (`vibe --agent review`, eight probes, a
> finished answer file, \$0: `../Lenticulum.jl-Review-2026-11-24/p42-krylov-dual/ANSWER-mistral.md`, probes
> beside it in `probes/`). code: `lib/ImplicitSolvers.jl/ext/ImplicitSolversKrylovExt.jl`
> (`_krylovjl_dual`, `_preconditioner`), `lib/ImplicitSolvers.jl/ext/ImplicitSolversForwardDiffExt.jl`
> (`_strip_dual_level`). tests: `lib/ImplicitSolvers.jl/test/krylovjl.jl`. bench: `bench/krylov_dual.jl`.

## What P42 does

Krylov.jl 0.10.12 requires an `AbstractFloat` scalar. When the working type of a `KrylovJL` solve is a
dual, the wrapper strips one dual level from the operator, the right-hand side and the preconditioners.
It then solves the primal system with the same `KrylovJL`, lifts the solution with zero partials,
computes the dual residual, and corrects each partial by P38's tangent correction. Nested duals recurse
one level at a time. `Complex{Dual}` works. A primal solve that fails reports its reason, and no
correction is made. P41's twelve `@test_broken` pass.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| F1 | plain `A` and `b` with a dual `left` (matrix or function) or `right`: all three methods throw `MethodError: no method matching Float64(::Dual)` (probes 3, 5, 5b). The dual branch looks only at the working type of `A` and `b`, so the dual preconditioner reaches Krylov.jl unstripped. With `ldiv = true` a dual factorisation gave the same `MethodError` instead of the documented `ArgumentError` | right (reproduced by `scratch/p42/f1.jl` in `../Lenticulum.jl-codex`). The solution does not depend on the preconditioner, so the derivative is zero | Krylov.jl runs only on a plain working type. So `_preconditioner` reduces a preconditioner's values to primal there: a matrix by `primal.(P)` (only when its element type is a dual), a function by `y .= primal.(P(x))`, a `UniformScaling` by its primal `λ`. With `ldiv = true` an object whose element type is a dual raises the `ArgumentError`. Test "LP-087: review of P42" in `krylovjl.jl` (three methods × matrix and function, a right preconditioner for GMRES, `ldiv` with a dual Cholesky; derivative exactly 0). It errored before the fix |
| — | found while reproducing F1, older than P42: the in-house `GMRES` with a dual **matrix** preconditioner and plain `A`, `b` throws a `MethodError`. A function preconditioner works (derivative 1e-16); `MINRES` refuses preconditioners by contract | right (same script) | `@test_broken` in the same testset; no row (small, no caller known) |
| — | checked clean: dual matrix, dual `b`, both; real and complex; $n \in \{1, 3, 8\}$; a callable with only a captured dual; nested tags in both orders; `jacobian` chunks 1 and 4; `hessian`; the report's `iterations` and `nfe` exactly the primal plus the dual residual plus each correction; `maxiters` 0, 1; `tol = 0`; zero primal `b`; NaN in one partial (`:nonfinite`); JLArrays; restarted GMRES; $\kappa \approx 10^6$ | — | — |

## Gates

`lib/ImplicitSolvers.jl` suite with the fix: 18,381 passed, 4 broken (three older root-solver checks and
the in-house case above) (`../Lenticulum.jl-codex/scratch/p42/IS-session2.log`). Codex's run before the
fix: VariationalDiffusion 2,359 passed, ImplicitLayers 1,968 passed (`scratch/p42/VD.log`, `IL.log`).
Neither uses `KrylovJL`, and the fix touches only the Krylov extension.

Non-dual solves are bitwise unchanged. The recorder `scratch/p42/record.jl` writes the 2,451 non-dual
solution and report records of P41's fixtures. Its output after the fix equals the record taken at
`02b3092c`, with the same flags (`-t auto --startup-file=no`, `JULIA_NUM_THREADS=auto`). Run without
`-t auto`, 18 Float32 records differ, both before and after the fix. The cause is the thread setting,
not this change: the output does not change from run to run at a fixed setting. Why the thread count
changes Float32 results was not examined.

## The measurement (`bench/krylov_dual.md`)

One primal solve plus one per partial is faster than the in-house solvers carrying the partials on
duals. At $n = 1000$ with 8 partials, `KrylovJL(:gmres)` takes 11.9 ms against 100.8 ms for the
in-house `GMRES`, and `KrylovJL(:minres)` 8.7 ms against 52.1 ms for `MINRES`. It needs 9× the operator
products (172 against 19), so on an operator that is expensive to apply the order may reverse; not
measured. The measurement goes into P40.

## Not run

GPU; the smoke runner as a whole (`krylov_dual` alone passed through it); the docs builds after the
session's note edits (Codex built both on its version); VariationalDiffusion and ImplicitLayers after
the fix.
