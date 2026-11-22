#implementation

> SolverReport records the primal residual at the returned point and the amount
> of work performed.
>
> Sources: [[Shared Solver Layer]], D3; code: `lib/ImplicitSolvers.jl/src/report.jl`; tests: `lib/ImplicitSolvers.jl/test/tolerance_report.jl`, `lib/ImplicitSolvers.jl/test/linsolve.jl`, `lib/ImplicitSolvers.jl/test/roots.jl`
>
> Theory (CT-ML wiki): —

A solver may converge, use its iteration budget, encounter a nonfinite value,
or stall. All four outcomes carry a report. The caller decides how to use an
unconverged iterate. The constructor applies `primal` even when a caller
constructs a report directly.

For roots, iteration counts are steps and `nfe` counts the initial residual and
each subsequent evaluation. For GMRES, iterations count Arnoldi steps and `nfe`
also includes true residual checks. A successful dense factorization and solve
counts as one iteration. For a dense callable, every basis probe counts toward
`nfe`. The relative bound scales with the problem, so the recorded norm by itself
does not estimate the error in the solution of an ill-conditioned system.
