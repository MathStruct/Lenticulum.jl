# ImplicitSolvers

ImplicitSolvers provides linear solves, derivative-free root finding and the linear
systems used by implicit-function adjoints. Solves return a solution and a
`SolverReport`. Its residual is the primal Euclidean norm at the returned iterate.

```julia
using ImplicitSolvers

u, report = solve_root(u -> u .- cos.(u), [0.5], Picard(damping=0.5))
A = [2.0 1.0; 0.0 3.0]
x, report = linsolve(v -> A * v, [1.0, 2.0], GMRES())
```

`Tolerance(; atol, rtol)` uses the iterate norm for roots and the right-hand-side
norm for linear systems. `DenseSolve` uses a host factorization, including for
device inputs. GMRES keeps the vector operations on the input array backend and
uses host storage for the small Krylov least-squares problem.

ForwardDiff can differentiate the steps a solver takes. At a primal root supplied
as the initial guess, a zero-step solve returns the initial guess's tangent.
Use the adjoint helpers to compute sensitivities of the implicit equation with
the caller's Jacobian products and parameter derivatives.

## API

```@index
Modules = [ImplicitSolvers]
```

```@autodocs
Modules = [ImplicitSolvers]
```
