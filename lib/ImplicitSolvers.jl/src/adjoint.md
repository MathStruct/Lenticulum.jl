#implementation

> The implicit-function adjoint uses the same linear solver and tolerance as a
> forward linear system.
>
> Sources: [[Shared Solver Layer]], D6; code: `lib/ImplicitSolvers.jl/src/adjoint.jl`; tests: `lib/ImplicitSolvers.jl/test/adjoint_fd.jl`
>
> Theory (CT-ML wiki): —

For a residual Jacobian $J$, the adjoint solves $J^	oplambda=g$. For a
fixed-point map of Jacobian $J$, the residual Jacobian is $I-J$ and the
solve uses $(I-J^	op)lambda=g$. The fixed-point operator applies the identity
and subtracts the supplied transposed action without forming a matrix for GMRES.
DenseSolve can materialize the action with basis probes.

The caller supplies the transpose action and constructs any parameter
cotangents. A singular adjoint system can stall or exhaust its budget, just like
any other linear system. This interface requires no AD rules.
