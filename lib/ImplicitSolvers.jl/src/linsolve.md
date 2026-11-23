#algorithm #implementation

> Dense LU or Cholesky and restarted GMRES solve real square systems. GMRES
> uses the supplied operator action and checks the true residual after each cycle.
>
> Sources: [[Shared Solver Layer]], D4; code: `lib/ImplicitSolvers.jl/src/linsolve.jl`, `lib/VariationalDiffusion.jl/src/krylov.jl`; tests: `lib/ImplicitSolvers.jl/test/linsolve.jl`, `lib/ImplicitSolvers.jl/test/device.jl`
>
> Theory (CT-ML wiki): —

GMRES minimizes the residual over the current affine Krylov space. It builds
the Arnoldi basis with two passes of Gram–Schmidt and uses Givens rotations
to solve the small least-squares problem. The estimated residual can end a
cycle. The true residual at the updated iterate decides convergence. Arnoldi
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
assertion. This layer provides neither preconditioning nor a device-native
direct factorization.
A dense report has `iterations = 1` after a successful factorization and `0` when the matrix
has nonfinite entries or the factorization fails; MyceliumLearning's graph adjoint tells a singular
$I - D^\top$ from an unmet tolerance by it (LP-081).

GMRES refuses complex systems with an `ArgumentError`: its Givens rotations are those of real
arithmetic, and on a complex diagonal system it diverged to a residual of $2\cdot10^{14}$ (GLM's review
of W1, finding 2). `DenseSolve` takes complex matrices, by LU, though the layer is written for real ones.
A cycle that starts from a zero residual returns at once instead of normalising it.
