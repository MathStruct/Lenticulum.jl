#problem

> **Status: closed (2026-11-24).** P41 (Krylov.jl as an optional backend of `linsolve`: `KrylovJL` in
> ImplicitSolvers, its solves in `ImplicitSolversKrylovExt`; Codex) and what its review found. Mistral found
> two defects, both right and fixed with regression tests: CG threw on a nonfinite product instead of
> reporting `:nonfinite`, and `left = I` threw a `MethodError`. Related: [[Shared Solver Layer]] ("Why not
> Krylov.jl"), [[LP-085 Findings of the review of P39|LP-085]], [[Feature Backlog]] (P40, P41).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; brief
> `~/.cache/lenticulum/codex-queue/p41-brief.md`, answer
> `../Lenticulum.jl-Review-2026-11-24/p41-krylov-ext/IMPL-from-codex.md`, logs in `scratch/p41/` of
> `../Lenticulum.jl-codex`). Mistral Large (`vibe --agent review`, eleven probes, a finished answer file:
> `../Lenticulum.jl-Review-2026-11-24/p41-krylov-ext/ANSWER-mistral.md`, probes beside it in `probes/`).
> code: `lib/ImplicitSolvers.jl/src/linsolve.jl` (`KrylovJL`), `lib/ImplicitSolvers.jl/ext/ImplicitSolversKrylovExt.jl`.
> tests: `lib/ImplicitSolvers.jl/test/krylovjl.jl`.

## What P41 does

`KrylovJL(method = :gmres | :minres | :cg; memory, maxiters, ldiv)` is defined in the package; with
`using Krylov` the extension runs Krylov.jl's solver behind `linsolve`, without it a call raises an error
saying to load Krylov.jl. D2's bound goes to Krylov.jl as `atol` (with `rtol = 0`); the wrapper then checks
the true residual and warm-starts with a tightened target when Krylov.jl stopped early (the LP-082 case:
bare `gmres` with a left Jacobi preconditioner stops at a true residual of 0.49 against a bound of 0.01; the
wrapper converges to 2.5e-16). `nfe` counts every product. Krylov.jl 0.10.12 rejects ForwardDiff dual
element types (its scalar type must be an `AbstractFloat`): twelve `@test_broken` cases, three solvers ×
(dual right-hand side, dual matrix, nested duals, complex duals). A dual system whose primal residual is
already within the bound reaches P38's tangent correction without a Krylov iteration and works.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| F1 | `KrylovJL(:cg)` throws `ErrorException("… not symmetric positive definite")` when an operator product turns NaN or Inf during the solve; GMRES and MINRES through the same wrapper report `:nonfinite` (probe 10) | right (reproduced). Krylov.jl's `cg` checks $\gamma = r^\top M r \ge 0$ and calls `error` when it fails, which NaN does | the wrapper runs CG on a `CgWorkspace`; when that error is thrown, a nonfinite residual or preconditioned residual in the workspace gives a `:nonfinite` report at the last iterate, and a finite one (a preconditioner that is not positive definite) an `ArgumentError`. Test "LP-086: review of P41" in `lib/ImplicitSolvers.jl/test/krylovjl.jl` (NaN and Inf after two products, all three methods; `left = -I`); it errored before the fix |
| F2 | `left = I` (a `UniformScaling`) with `ldiv = false` throws a `MethodError`: the wrapper treats any non-matrix as a function (probe 9) | right (reproduced). `I` is the natural "no preconditioner" and Krylov.jl's own default | `UniformScaling` passes to Krylov.jl as is (it takes it by `mul!` and `ldiv!`) and becomes `v -> P*v` for P38's correction; same test, `I` and `2.0I` for each method with and without `ldiv`, and a dual right-hand side with `left = I`, `right = 2.0I` |
| — | not counted as defects: `ldiv = true` with a function (outside the contract; Krylov.jl fails the same way); a host `Diagonal` preconditioner on JLArrays scalar-indexes inside LinearAlgebra's `mul!` (Krylov.jl fails the same way; a device `Diagonal` works) | agreed | none |
| — | checked clean: true residual on every return path and never `:converged` above the bound (3,644 oracle checks, three methods, four element types); `nfe` against a counting callable; `iterations ≤ maxiters`; `maxiters` 0 and 1; `tol = 0`; zero, NaN and Inf in `b` and `A`; $n = 0, 1$; integer and `BigFloat`; mixed element types; JLArrays with `allowscalar(false)`; the error without Krylov.jl | — | — |

## Gates

`lib/ImplicitSolvers.jl` suite after the fixes: 15,795 passed, 15 broken (the twelve above and three
older), 3 min 12 s (`scratch/p41/IS-session.log` in `../Lenticulum.jl-codex`).
Not run: the suites of the packages that depend on ImplicitSolvers (the change adds a type, an extension
and one guard line in `linsolve` for `KrylovJL`; nothing they call changed), a hardware GPU, the docs
builds after the session's note edits (Codex ran both on its version).
