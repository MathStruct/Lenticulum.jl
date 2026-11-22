#implementation

> A solve accepts a finite residual when it is within an absolute floor plus a
> relative allowance. Both contributions use the primal residual precision.
>
> Sources: [[Shared Solver Layer]], D2–D3; code: `lib/ImplicitSolvers.jl/src/tolerance.jl`, `lib/ImplicitSolvers.jl/ext/ImplicitSolversForwardDiffExt.jl`; tests: `lib/ImplicitSolvers.jl/test/tolerance_report.jl`
>
> Theory (CT-ML wiki): —

With residual norm $r$ and scale $s$, the bound is $mathrm{atol}+mathrm{rtol},s$.
The two defaults are independent and each is the larger of $10^{-9}$ and $30epsilon$.
The scale is the current iterate norm for a root and the right-hand-side norm for
a linear system. These are global Euclidean criteria. Message passing has its
own per-group scales.

A dual comparison can consult its partials when the values tie. The ForwardDiff
extension recursively strips the partials, and the solvers take the reporting norm
on primal entries. This also covers an exact zero residual with a nonzero or
nonfinite tangent. The norm uses the array backend's LinearAlgebra implementation.

`Tolerance` refuses a negative or NaN `atol` or `rtol` (on its primal value). Such a bound can never be
met, not even by a zero residual, and GMRES then ran on past an exact solve and reported `:nonfinite`
(GLM's review of W1, finding 1, [[LP-078 Findings of the reviews of the Mycelium coverage tests and the Anderson and block AD fixes|LP-078]]).
A bound that rounds to `Inf` in the residual's type (`1e300` for a `Float32` problem) is not refused.
