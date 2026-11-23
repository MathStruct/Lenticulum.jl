#algorithm #implementation

> Restarted GMRES solves general square systems, MINRES solves Hermitian systems,
> and dense LU and Cholesky handle direct solves. Both Krylov solvers use the
> supplied operator action and check the true residual after each cycle.
>
> Sources: [[Shared Solver Layer]], D4; code: `lib/ImplicitSolvers.jl/src/linsolve.jl`, `lib/VariationalDiffusion.jl/src/krylov.jl`; tests: `lib/ImplicitSolvers.jl/test/linsolve.jl`, `lib/ImplicitSolvers.jl/test/device.jl`, `lib/ImplicitSolvers.jl/test/krylov_oracle.jl`
>
> Theory (CT-ML wiki): —

GMRES minimizes the residual over the current affine Krylov space. It builds
the Arnoldi basis with two passes of Gram–Schmidt and uses Givens rotations
to solve the small least-squares problem. The estimated residual can end a
cycle. The true residual at the updated iterate decides convergence.

A left approximate inverse $M$ maps the residual at the start of a cycle to
$M(b-Ax)$. A right approximate inverse $N$ maps the cycle correction back to
the original coordinates. The Arnoldi operator is $MAN$ with omitted sides
acting as the identity. The cycle estimate for a left preconditioner scales the
D2 bound by $\lVert M(b-Ax)\rVert/\lVert b-Ax\rVert$ at the cycle start.
After the update, one original operator product recomputes $b-Ax$. Its norm
goes in the report and is compared to the D2 bound with the scale $\lVert b\rVert$.
An estimate that meets its target can therefore trigger another restart.
The original unpreconditioned path uses the same products as before.

Arnoldi
breakdown is measured relative to the operator product before orthogonalization,
which preserves the test under rescaling of the system. A singular triangular
step returns the best iterate from the completed columns of the cycle.

GMRES retains the basis as backend vectors and applies updates by broadcast.
Only the Hessenberg matrix, rotations and least-squares right-hand side are
host arrays. JLArrays exercises the vector path with scalar indexing disabled.

Dense solves use host factorization. A matrix on a device is copied in bulk.
For a callable, basis vectors follow the right-hand side's backend and their
images are copied to the host. The returned solution follows the right-hand
side's backend. Cholesky takes positive definiteness and symmetry as a caller
assertion. Dense solves reject preconditioners and use host direct factorization.
A dense report has `iterations = 1` after a successful factorization and `0` when the matrix
has nonfinite entries or the factorization fails; MyceliumLearning's graph adjoint tells a singular
$I - D^\top$ from an unmet tolerance by it (LP-081).

Complex GMRES uses the conjugating inner product and unitary Givens rotations.
For each pair $(a,b)$ the rotation has first row
$(c,s)=(\overline a,\overline b)/\operatorname{hypot}(|a|,|b|)$ and second row
$(-\overline s,\overline c)$. It maps the pair to a real nonnegative diagonal
entry and zero below it. This covers the complex diagonal system from W1
review finding 2 and the complex matrix and function fixtures in the W5 oracle.
A zero preconditioned residual ends a cycle without division. The true residual
determines whether the solver has converged or stalled.

## MINRES

Paige–Saunders MINRES minimizes the residual over a Krylov space for a real
symmetric or complex Hermitian operator, including indefinite systems. The
three-term Lanczos recurrence uses the real part of the conjugating inner product
for the tridiagonal diagonal. Each new column is reduced by the previous two
Givens rotations and one new rotation. The resulting two superdiagonal entries
and diagonal give the next search direction from the two previous directions.
The current Givens rotation both weights the step in that direction and updates
the residual estimate. Only the running Lanczos vectors, search directions and
iterate use vector storage on the backend of the right-hand side.

An operator product at the starting iterate initializes the residual for a
nonzero right-hand side. The solver
uses one further product to check the true residual at the end of each cycle,
when the estimate meets the D2 bound, Lanczos breaks down or the step budget is
spent. If the true residual misses the bound, a changed iterate starts a new
cycle on that residual while the budget lasts. An unchanged iterate stalls.
The report counts Lanczos steps and all products separately. The scalar
decisions use primal values.

MINRES assumes Hermitian symmetry. The true residual judges the resulting
iterate if that assumption fails. The current implementation supports only
unpreconditioned systems. A preconditioned variant requires a symmetric positive
definite preconditioner.
