#implementation

> A central finite-difference stencil builds a dense reference Jacobian on
> host vectors.
>
> Sources: [[Shared Solver Layer]], D7; code: `lib/ImplicitSolvers.jl/src/fd.jl`; tests: `lib/ImplicitSolvers.jl/test/adjoint_fd.jl`
>
> Theory (CT-ML wiki): —

The second-order centered stencil balances truncation error of order $h^2$ and
a roundoff term of order $epsilon/h$. This gives a default relative multiplier
$epsilon^{1/3}$, scaled per coordinate by $max(1,|u_j|)$. The denominator uses
the actual separation of the rounded probes. An explicit step that rounds to
indistinguishable probes raises an error.

The stencil uses two evaluations per coordinate and supports rectangular
Jacobians. For empty input it evaluates the output once to determine the row
count. It is a reference for diagnostics and small systems. The root and
matrix-free adjoint paths require no dense finite-difference Jacobian.
