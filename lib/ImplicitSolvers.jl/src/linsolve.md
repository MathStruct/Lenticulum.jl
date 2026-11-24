#algorithm #implementation

> Restarted GMRES solves general square systems, MINRES solves Hermitian systems,
> and dense LU and Cholesky handle direct solves. Both Krylov solvers use the
> supplied operator action and check the true residual after each cycle.
>
> Sources: [[Shared Solver Layer]], D2 and D4; code: `lib/ImplicitSolvers.jl/src/linsolve.jl`, `lib/ImplicitSolvers.jl/ext/ImplicitSolversForwardDiffExt.jl`, `lib/ImplicitSolvers.jl/ext/ImplicitSolversKrylovExt.jl`, `lib/VariationalDiffusion.jl/src/krylov.jl`; tests: `lib/ImplicitSolvers.jl/test/linsolve.jl`, `lib/ImplicitSolvers.jl/test/device.jl`, `lib/ImplicitSolvers.jl/test/krylov_oracle.jl`, `lib/ImplicitSolvers.jl/test/krylovjl.jl`
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

## Tangents at a breakdown

Under ForwardDiff the GMRES and MINRES arithmetic differentiates the Krylov
iterate, including the basis vectors. Stopping decisions use the primal values.
When the differentiated iterate satisfies the tangent system, its partials
are returned as computed. At an exact breakdown the primal system can converge
in an invariant Krylov space while a tangent needs a direction outside it.
Two examples are an eigenvector right-hand side and a right-hand side with
zero primal and nonzero tangent. The regression tests for LP-083 compare both
with the derivative of `DenseSolve`.

After primal convergence, the ForwardDiff extension extracts each partial of
the already available residual. For direction $i$ the tangent residual is

$$
\dot r_i = \dot b_i - \dot A_i x - A\dot x_i.
$$

The D2 scale for this check is
$\max(\lVert\dot b_i\rVert, \lVert\operatorname{partials}_i(Ax)\rVert, \lVert Ax\rVert\,\lVert\dot x_i\rVert/\lVert x\rVert)$.
The existing product $Ax$ supplies the second and third norms. The third estimates
$\lVert A\dot x_i\rVert$, the size of the tangent system's right-hand side at the solution: without it a
tangent only in $A$ (plain $b$) has $\operatorname{partials}_i(Ax) = -\dot r_i$, the scale is the residual
itself, and with `atol = 0` every solve ran a correction ([[LP-084 Findings of the review of P38|LP-084]]). A passing direction needs
only the broadcast and reductions for the check. For each failing direction,
a recursive `linsolve` solves $A\delta_i=\dot r_i$ with the same algorithm,
tolerance and preconditioners, stripping the current dual tag from their
outputs. The extension adds $\delta_i$ to the corresponding partial of $x$.
Both extraction and reconstruction use backend broadcasts. MINRES evaluates
one product at its zero-primal early exit for dual inputs and counts it in
`nfe`. At complex Arnoldi breakdown the Givens norm uses the magnitude of the
remaining entry directly, because differentiating the magnitude of the zero
entry would introduce NaN partials before the residual check. If a scaled dual
norm gives MINRES a nonfinite value, it checks the norm of the primal vector
to distinguish a zero Lanczos direction from a nonfinite product.

A nested dual retains its inner tags. The extension first checks the inner
value's tangents, then checks the outer partials including their mixed
coefficients. An inner correction can change the mixed residual through
$\dot A_i$, so it triggers one fresh product. A recursive correction solve
handles the inner tags of a failing outer partial, even when that partial's
primal value is zero.

The primal iterate and reported primal residual are kept. Each correction has
its own iteration budget and contributes its work to the report. If a
correction fails, the first failure supplies the report's status. This check
acts only after a converged Krylov solve with dual-valued output. Dense solves
and failed primal solves return directly. Accuracy is governed by the residual
tolerance and the conditioning of the system.

## Krylov.jl backend (P41)

`KrylovJL(:gmres; memory=40, maxiters=200)` selects Krylov.jl's restarted GMRES
after `using Krylov`. The same algorithm type selects `:minres` for Hermitian
systems or `:cg` for Hermitian positive definite systems. Krylov.jl takes the
left and right approximate inverse actions as its `M` and `N`. Matrices use
`mul!`, and callable actions copy their output into Krylov.jl's buffer. With
`ldiv=true`, the preconditioners go directly to Krylov.jl for `ldiv!`. MINRES
and CG accept only a Hermitian positive definite left preconditioner.

The first Krylov attempt uses `atol` equal to D2's bound at scale $\lVert b\rVert$
and `rtol=0`, in the primal real type. The wrapper evaluates $b-Ax$ after each
attempt and uses its primal norm to decide convergence. If the residual misses
D2 and decreases, the wrapper warm-starts another attempt with the remaining
iteration budget. It multiplies the D2 bound by the ratio of Krylov.jl's final
residual estimate to the true norm, capping the next target at the previous
one. An unchanged iterate or a residual that fails to decrease stalls when
budget remains. The two-dimensional Jacobi test in `krylovjl.jl` exercises
an early stop by bare GMRES and convergence after a warm start.

Iteration counts come from Krylov.jl. The wrapper counts operator products
directly so internal restarts, warm starts, and true-residual checks all
contribute to `nfe`. The workspace follows the right-hand side's storage and
is allocated per attempt. Cross-call workspace reuse is deferred to Q11.

Two difficulties found by the review ([[LP-086 Findings of the review of P41|LP-086]]). Krylov.jl's
`cg` calls `error` when $r^\top M r$ is negative or NaN, so a nonfinite operator product threw where
GMRES and MINRES report `:nonfinite`. The wrapper runs CG on a `CgWorkspace` and reads the residual it
holds after the error: nonfinite gives a `:nonfinite` report at the last iterate, finite (a preconditioner
that is not positive definite) an `ArgumentError`. A `UniformScaling` preconditioner (`I`, `2.0I`) is
passed to Krylov.jl as is; it is neither a matrix nor callable, and was first taken for a function.

### ForwardDiff with Krylov.jl (P42)

Krylov.jl 0.10.12 requires its working real scalar to be an `AbstractFloat`.
When the working type carries a dual tag, the wrapper asks the ForwardDiff
extension to strip that tag from the operator, right-hand side and
preconditioners. The Krylov extension has no ForwardDiff dependency. A plain
matrix keeps its storage and native matrix product. A dual matrix is stripped
by broadcast, and a callable strips the tag from its output. The stripped
system uses the same `KrylovJL` method and D2 true-residual checks.

The wrapper lifts the resulting value to the working type with zero partials.
After convergence it computes the dual residual and uses P38 to correct each
partial that misses tolerance. Stripping removes one tag at a time from real
duals and from both components of `Complex{Dual}`. A nested value re-enters the
same solver with its inner tags intact. If the value solve fails, its reason
propagates with the lifted iterate and the wrapper skips tangent correction.
Dual inputs with `ldiv=true` still raise `ArgumentError` because the tag stripping
works on preconditioner actions and their values.
A preconditioner that carries a dual which `A` and `b` do not (the review's F1,
[[LP-087 Findings of the review of P42|LP-087]]) is reduced to its primal values
inside the Krylov.jl call: the solution does not depend on it, so its derivative
is zero. Such a factorisation with `ldiv=true` raises the same `ArgumentError`.

The report's `iterations` adds the value solve and each correction solve.
`nfe` adds all their operator products, one product for the dual residual
after value convergence, and one more if P38 refreshes that residual after an
inner correction at this level. For a callable the initial application at zero
discovers the working element type and also counts in `nfe`. Preconditioner
applications are outside this count. The reported residual comes from the
value solve. The P42 tests in `krylovjl.jl` check the count against a callable's
counter and compare real and complex tangents with dense LU.

## AD backend matrix (Q5)

Measured 2026-11-24 with `lib/ImplicitSolvers.jl/test/ad_matrix.jl`.

| Entry point | Zygote | ForwardDiff | Enzyme | Mooncake |
|---|---|---|---|---|
| `linsolve/DenseSolve` | broken: mutation/unsupported construct | pass | broken: mutation/unsupported construct | pass |
| `linsolve/GMRES` | broken: mutation/unsupported construct | pass | broken: mutation/unsupported construct | pass |
| `linsolve/MINRES` | pass | pass | broken: mutation/unsupported construct | pass |
| `linsolve/KrylovJL` | broken: mutation/unsupported construct | pass | broken: mutation/unsupported construct | broken: mutation/unsupported construct |
| `solve_root/Picard` | pass | pass | broken: mutation/unsupported construct | pass |
| `solve_root/Broyden` | broken: differentiates through a solver loop | pass | broken: mutation/unsupported construct | pass |
| `solve_columns/NewtonDescent` | broken: differentiates through a solver loop | pass | broken: mutation/unsupported construct | pass |

Enzyme forward mode is also broken on DenseSolve with a runtime activity error. Numerical agreement leaves the choice of differentiation rule to the Q6 checks.

These rows use direct DI gradient checks. DifferentiationInterfaceTest 0.11 requires JLArrays at or below 0.3 when present, and the device tests require JLArrays 0.4 or 0.5.

The standalone matrix took 5m13s before the split. KrylovJL and the column driver run with `LENTICULUM_SLOW_TESTS=true` or `only`.
