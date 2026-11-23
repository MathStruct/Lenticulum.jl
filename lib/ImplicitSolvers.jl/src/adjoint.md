#implementation

> The implicit-function adjoint uses the same linear solver and tolerance as a
> forward linear system.
>
> Sources: [[Shared Solver Layer]], D6; code: `lib/ImplicitSolvers.jl/src/adjoint.jl`; tests: `lib/ImplicitSolvers.jl/test/adjoint_fd.jl`
>
> Theory (CT-ML wiki): —

For a residual Jacobian $J$, the adjoint solves $J^*\lambda=g$. For a
fixed-point map of Jacobian $J$, the residual Jacobian is $I-J$ and the
solve uses $(I-J^*)\lambda=g$. The fixed-point operator applies the identity
and subtracts the supplied adjoint action without forming a matrix for GMRES or MINRES.
DenseSolve can materialize the action with basis probes.
MINRES requires the operator being solved, $J^*$ or $I-J^*$, to be Hermitian.

The GMRES preconditioners apply to the system being solved, $J^*$ or $I-J^*$.
Both use the same true-residual rule. The complex adjoint checks are in
`lib/ImplicitSolvers.jl/test/adjoint_fd.jl`.

The caller supplies the adjoint action and constructs parameter cotangents.
A singular adjoint system can stall or exhaust its budget.
