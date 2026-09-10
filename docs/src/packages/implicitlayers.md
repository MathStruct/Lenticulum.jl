# [ImplicitLayers](@id implicitlayers)

```@meta
CurrentModule = ImplicitLayers
```

SciML's implicit layers — **deep equilibrium networks** and **neural ODEs** — as factors.

Dependencies are `LuxCore`, `Random` and `LinearAlgebra`: neither
`DeepEquilibriumNetworks.jl` nor `DiffEqFlux.jl` is required, because what these wrappers
need is the inner Lux network, not the assembled layer.

## Three wrappers, and the difference matters

| wrapper | you supply | polarities |
|---|---|---|
| `LuxFactor` | the **assembled** `DeepEquilibriumNetwork` / `NeuralODE` | 1 |
| `DEQFactor` | the **cell** ``g_\theta(z,x)`` | 2, if the residual is square |
| `NeuralODEFactor` | the **dynamics** ``f_\theta(z,t)`` | 2, always |

`LuxFactor` works today on anything in the SciML ecosystem and asks nothing of it. It gives
you graph membership, parameter management and free-energy accounting — and one direction,
because a Lux layer has already chosen which side is the input.

The other two keep the *relation* instead of the solved function, and get both directions:

```julia
# a DEQ: keep the residual r(x,z) = z - g(z,x)
f = DEQFactor(my_cell, (x = 4, z = 4); solver = BroydenSolver())
solve_state(f, x, ps, st)     # solve for z  — what SciML's DEQ does
solve_input(f, z, ps, st)     # solve for x  — what it cannot be asked
```

The reverse solve needs ``\dim x = \dim z``; otherwise the residual is not square and
`supports_polarity` refuses that direction rather than solving it badly.

## Neural ODEs are bidirectional for free

A flow is a diffeomorphism, so integrating backwards inverts it — same integrator, endpoints
swapped. No solver, no dimension condition, no convergence question:

```julia
f = NeuralODEFactor(my_net, 4; tspan = (0.0, 1.0), integrator = RK4Integrator())
flow_forward(f, z₀, ps, st)   # t₀ → t₁
flow_reverse(f, z₁, ps, st)   # t₁ → t₀
flow_logdet(f, z₀, ps, st)    # plus the change-of-variables correction
```

Integrators are fixed-step and explicit (`RK4Integrator`, `EulerIntegrator`) deliberately: an
adaptive solver takes different steps forwards and backwards, which breaks the round trip.

## Solvers

`PicardSolver` is the naive "keep applying the layer" loop and converges only if the
iteration is a contraction. `BroydenSolver` is derivative-free quasi-Newton and converges on
plenty of problems where Picard diverges — it is the default, and it is what
`DeepEquilibriumNetworks.jl` reaches for too.

Both return a `SolveReport` (converged, iterations, residual, function evaluations) rather
than throwing. A solver that stopped early is an inexact inversion, which is legal.

`ift_sensitivity` implements the implicit function theorem,
``\partial z^\ast/\partial x = (I - \partial_z g)^{-1}\partial_x g`` — one linear solve, no
unrolling.

## Known gaps

- `fd_jacobian` is finite differences: ``O(n)`` forward passes, fine for tests, not for scale.
  The real answer is a VJP from automatic differentiation.
- Broyden stores a dense inverse Jacobian; real DEQs use limited-memory Broyden.
- Explicit RK4 diverges quietly on stiff systems, and reverse integration is unstable for
  strongly dissipative dynamics.
- Every inversion returns a `DiracBelief`.

## API

```@index
Modules = [ImplicitLayers]
```

```@autodocs
Modules = [ImplicitLayers]
```
