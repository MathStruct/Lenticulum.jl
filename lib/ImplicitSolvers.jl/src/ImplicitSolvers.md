#implementation

> ImplicitSolvers collects residual-based root and linear solvers with a common
> stopping criterion, an adjoint interface, and a reference Jacobian stencil.
>
> Sources: [[Shared Solver Layer]]; code: `lib/ImplicitSolvers.jl/src/ImplicitSolvers.jl`; tests: `lib/ImplicitSolvers.jl/test/runtests.jl`
>
> Theory (CT-ML wiki): —

The module loads LinearAlgebra and exports the solver settings, solve entry points,
reports, and finite differences. ForwardDiff is optional. Its extension unwraps
dual numbers for decisions while the solver steps work with the original values.
Implicit differentiation starts from a supplied Jacobian action. The caller owns
the rules that differentiate the residual with respect to parameters.
