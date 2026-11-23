#algorithm #implementation

> Picard, good inverse Broyden, and limited-memory inverse Broyden find roots
> from out-of-place residual evaluations.
>
> Sources: [[Shared Solver Layer]], D5; code: `lib/ImplicitSolvers.jl/src/roots.jl`, `lib/ImplicitLayers.jl/src/solve.jl`; tests: `lib/ImplicitSolvers.jl/test/roots.jl`, `lib/ImplicitSolvers.jl/test/device.jl`
>
> Theory (CT-ML wiki): —

Picard subtracts a damped residual. Broyden applies an approximate inverse
Jacobian $H$ to the residual and updates $H$ so it maps the observed residual
change $y$ to the step $s$. The rank-one update is

$$
H \gets H + \frac{(s-Hy)s^\top H}{s^\top Hy}.
$$

The implementation divides out the primal norms of $s$ and $Hy$ before testing
the denominator against $\sqrt{\epsilon}$. The normalized test avoids an absolute
cutoff that would discard valid secants for small residuals. Dense Broyden keeps
a backend matrix. Limited Broyden keeps the last few pairs of vectors
representing the additive rank-one terms, which can optionally be retained by the
caller. Swapping the two histories applies the transpose of the stored inverse.

The solvers use neither line search nor a convergence guarantee for arbitrary
initial guesses. Finite differences are a separate diagnostic. ForwardDiff
differentiates the executed steps. At an exact primal warm start, the solve
returns at step zero and preserves the input tangent even when the implicit
derivative differs. The test with $F(u,p)=u-p$ records this limitation.
