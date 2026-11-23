#algorithm #implementation

> Newton descent and the trust-region method find roots in independent columns.
> Each column has its own step, convergence test and solver report.
>
> Sources: [[Shared Solver Layer]] D8–D11, Nocedal and Wright, *Numerical Optimization* (2006), Algorithm 4.1 and Theorem 4.1, Moré and Sorensen, *Computing a Trust Region Step* (1983), Steihaug, *The Conjugate Gradient Method and Trust Regions in Large Scale Optimization* (1983). Code: `lib/ImplicitSolvers.jl/src/descent.jl`. Tests: `lib/ImplicitSolvers.jl/test/descent.jl`, `lib/ImplicitSolvers.jl/test/device.jl`.
>
> Bibliography: [[Bibliography#^nocedal2006numopt|Nocedal & Wright 2006]] · [[Bibliography#^more1983trustregion|Moré & Sorensen 1983]] · [[Bibliography#^steihaug1983cg|Steihaug 1983]]
>
> Theory (CT-ML wiki): —

`solve_columns` passes a matrix of unfinished columns and their original indices
to a named tuple of callbacks. Residual and scale callbacks establish the D2
bound. The caller supplies the stability test, inverse Jacobian action, merit
and quadratic model as needed. The driver shares evaluations across columns
and retires each column independently. Its report counts every residual
evaluation for that column, including rejected trials. A zero iteration budget
still evaluates and tests the residual at the initial point.

Newton descent backtracks on the residual norm when the supplied linearisation
is stable. It halves the damping until the Armijo test succeeds or the damping
reaches its lower bound. In the latter case it takes the first update at or
below the damping threshold. An unstable column steps along the negative
residual. A separate trial residual callback can discard model state during
backtracking.

The trust-region method minimises the quadratic model
$$
g^\top p + \tfrac12 p^\top H p,\qquad \lVert p\rVert\leq\Delta.
$$
The dense step uses an eigendecomposition and bisection for the Moré–Sorensen
optimality conditions. In the hard case it extends the shifted solution along
a lowest eigenvector to the boundary. Dense factorization uses host storage
with bulk transfers for a device Hessian and gradient. Steihaug CG uses the
supplied Hessian action on the array backend and stops at the boundary, on
negative curvature, or when its linear residual meets its relative criterion.

The ratio of actual to predicted reduction shrinks the radius to one quarter
of the step norm below $1/4$. It doubles the radius above $3/4$ when the step
reaches at least $99\%$ of the boundary. A ratio above $10^{-4}$ accepts the
step. When the predicted decrease is at most
$100\epsilon_a\max(|f|,1)$, merit rounding can dominate the ratio. A decrease
in the residual norm then accepts the trial. The arithmetic epsilon
$\epsilon_a$ can come from the merit evaluator, for example a Float32 network
with Float64 solver coordinates. A radius below $10\epsilon_a(1+s)$, with
the caller's scale $s$, stops the solver when further progress is below that
working resolution. A trial point whose residual is not finite counts as ratio
0: the step is rejected and the radius shrinks to a quarter of the step, so
the iteration backs away from a region where the relation is undefined (LP-085;
VariationalDiffusion's loop before P39 got a NaN ratio, kept the radius and
retried the same step until `maxiters`). The report is `:stalled` when the residual remains
outside tolerance, including at a Gauss–Newton non-root minimum.

The driver requires independent columns and real trust-region models. The
caller owns derivatives and the choice of merit, and decides what stability
means. Root convergence is a residual statement and can be reported at an
unstable point. The Steihaug step searches the Krylov space generated from the
model gradient, which can miss a negative-curvature direction orthogonal to it.
