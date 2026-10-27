# [ImplicitLayers](@id implicitlayers)

```@meta
CurrentModule = ImplicitLayers
```

SciML's implicit layers — **deep equilibrium networks** and **neural ODEs** — as factors.

There is no hard SciML dependency: neither
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

## Optional automatic differentiation

Load `DifferentiationInterface`, `ADTypes` and your backend package. For a DEQ solve,
pass `ad=AutoForwardDiff()`, `AutoZygote()`, `AutoEnzyme()` or `AutoMooncake()` to
`solve_state`. Load ForwardDiff as well for the EnzymeCore/Mooncake extensions.
The weak extensions implement the IFT, without differentiating root-finder
iterations. Both input and parameter derivatives are tested against the linear-cell
closed form and finite differences. Root convergence is required; singular residual
Jacobians throw. Only first-order solved-state derivatives are covered; the reverse
`solve_input` has no AD rule. Lux state is fixed throughout the state root solve and
advances once at the final root. Solve reports, initial guesses and state have no tangent.

`deq_sensitivity(...; wrt=:ps, ad)` gives the dense parameter sensitivity directly.
`DEQFactor(...; ift=DenseIFT())` defaults to dense LU. To use matrix-free IFT
rules, load Krylov and select `ift=KrylovIFT(rtol=1e-10, atol=1e-12, itmax=400,
memory=20)`. Restarted GMRES uses only residual JVPs (forward) or VJPs (reverse).
It throws on nonconvergence and checks the final residual explicitly. Krylov.jl
provides the operator-product interface directly, without another operator package.
All four backends are checked against the linear exact solution, including a
200-dimensional contractive example that forbids the dense state-Jacobian API.
`deq_sensitivity` stays dense. The native Enzyme/Mooncake rules keep their existing
inner ForwardDiff path in dense mode; in Krylov reverse mode each uses its own
reverse backend. GPU and higher-order IFT rules remain untested. Enzyme outer
closures capturing arrays need DI constant contexts or
`AutoEnzyme(function_annotation=Enzyme.Const)`; see `solve.md` for a reproducer.

Core's `energy_jacobian(...; wrt=:ps, ad)`, `energy_gradient` and
`free_energy_gradient` apply to DEQ, NeuralODE and scored Lux factors. Built-in
Euler/RK4 endpoint flows use functional Lux state threading and differentiate the
discrete map with ForwardDiff, Zygote, Mooncake and Enzyme. For Enzyme, pass
`AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))` (or Forward):
constant model/stage memory flows through an active fold, requiring runtime activity.
The factor derivative and graph training entry points check the mode before AD and
raise an `ArgumentError` with that configuration when it is missing. Arbitrary outer
DI calls must select it themselves. Plain `AutoEnzyme()` works for DEQ and scored
LuxFactor paths. The ODE activity check does not require ForwardDiff; DEQ IFT does.
`neuralode.md` describes the activity requirement. Tests derive both scalar discrete derivatives,
check nonempty Lux state and fit a known ODE parameter using Mycelium's graph
`parameter_step`. `LuxFactor(...; dims=(nin,nout), residual=true)` opts into squared
endpoint-residual scoring; the default wrapper still delegates energy to downstream losses.

## Neural ODE endpoint integration

The continuous flow has an inverse wherever solutions exist uniquely in both directions.
Numerical reverse integration approximates that inverse; Euler/RK4 do not invert their
discrete forward map, even in exact arithmetic. Both endpoint polarities are supported:

```julia
f = NeuralODEFactor(my_net, 4; tspan=(0.0,1.0), integrator=RK4Integrator())
flow_forward(f, z₀, ps, st)
flow_reverse(f, z₁, ps, st)
flow_logdet(f, z₀, ps, st)
```

`RK4Integrator(steps=50)` remains the default. Euler's round-trip error is ``O(h)``.
RK4 has fourth-order forward error and an ``O(h^4)`` round-trip bound for smooth dynamics;
the scalar linear round trip is sharper, ``O(h^5)``. Its doubling ratio is 32, while the
forward-error ratio is 16. `lib/ImplicitLayers.jl/examples/flow_roundtrip.jl` reproduces the
table in the implementation note: at ``a=-5,T=1,N=50``, Euler's residual is 0.394994 with a
20.9% start-point error, RK4's is ``6.953127\times10^{-7}``. At ``N=1`` RK4's is 895.18.
Inverting a contracting flow separately amplifies absolute endpoint perturbations by
``e^{|a|T}``; a better integrator cannot remove this conditioning.

## Optional OrdinaryDiffEq extension

Install the solver subpackages, then load them with `ImplicitLayers`:

```julia
using ImplicitLayers, OrdinaryDiffEqCore, OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK
adaptive = SciMLIntegrator(Tsit5(); reltol=1e-9, abstol=1e-12)
fixed = SciMLIntegrator(ImplicitMidpoint(); adaptive=false, dt=0.02,
                       reltol=1e-12, abstol=1e-12)
f = NeuralODEFactor(my_net, 4; integrator=fixed)
```

Core/SciMLBase supply the solve interface; Tsit5 and SDIRK supply the selected algorithms.
These are weak dependencies, so the full OrdinaryDiffEq umbrella and its other solvers
are unnecessary. The test environment resolves Core 4.19.0, Tsit5 2.1.5 and SDIRK 2.9.8.

The extension solves an out-of-place `ODEProblem` over either time direction and returns
only the final state. It signs positive `dt` automatically, disables intermediate storage,
and rejects failed/incomplete solves. CNF divergence is a finite-difference trace integrated
as an augmented state; the built-in quadrature is unchanged. Lux state is threaded through
every RHS evaluation, including rejected stages and divergence probes.

`inversion_kind` gives Euler/RK4 and adaptive Tsit5 `SolverInversion(integrator)`.
Midpoint and Trapezoid are symmetric methods ([Hairer–Lubich–Wanner II.3](https://doi.org/10.1007/3-540-30666-8));
with `adaptive=false`, matching grids (`dt` divides the interval), a default step limiter,
and only `dt`, `adaptive`, `abstol`, `reltol`, `maxiters` options, the trait permits
`ExactInversion` up to nonlinear-solve tolerance and roundoff. `assemble` also requires empty
Lux state. This describes the discrete inverse, not exactness against the continuous ODE.
Tests use `abstol=reltol=1e-12` and check relative round-trip residual below ``10^{-10}``.

Load SciMLSensitivity to enable SciML endpoint parameter AD:

```julia
using SciMLSensitivity, OrdinaryDiffEqTsit5
itg = SciMLIntegrator(Tsit5(); abstol=1e-12, reltol=1e-10)
# Optional continuous adjoint:
adjoint_itg = SciMLIntegrator(Tsit5(); abstol=1e-12, reltol=1e-10,
    sensealg=GaussAdjoint(autojacvec=ZygoteVJP()))
f = NeuralODEFactor(my_net, 4; integrator=itg)
J = energy_jacobian(f, z0, nothing, y, ps, st; wrt=:ps, ad=AutoZygote())
```

The default `sensealg=nothing` resolves to `ForwardDiffSensitivity()` for small
parameter counts, as recommended by the
[SciMLSensitivity manual](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#Choosing-a-Sensitivity-Algorithm).
Flattened Lux parameters are passed as `ODEProblem.p`, with a zero-valued shape
template in the RHS. The sensitivity path requires recursively empty incoming and returned Lux state.
Nonempty incoming state falls back to legacy evaluation-state threading for primal
forward/reverse flows and inference. Factor and graph derivative entry points reject
AD with nonempty state before tracing; reference differences remain available.
This excludes mutable statistics whose evolution depends on adaptive/rejected stages.
An arbitrary outer DI call bypasses the factor/graph check, so its caller must enforce
the empty-state restriction, as with Enzyme runtime activity. Euler/RK4 state
threading is unchanged.

Tsit5 tests cover AutoForwardDiff (dual solves), AutoZygote and AutoMooncake with
the default, `InterpolatingAdjoint` and `GaussAdjoint` (both with `ZygoteVJP()`).
They compare with ``D_a z(T)=Tz_0e^{aT}`` and nonlinear finite differences; graph
`parameter_step` learns the continuous-time target ``e^{0.4}`` with the default.
Factor/graph derivative entry points explicitly reject AutoEnzyme on this SciML path.
Sensitivity selection leaves inversion labels and reversal semantics unchanged.
CNF/divergence AD, other solvers, callbacks, GPU and higher derivatives are untested.
Continuous adjoints can retain intermediate trajectories internally. Backsolve
reverse-conditioning concerns still apply.

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

- Dense IFT remains the default; matrix-free mode has no preconditioner interface.
- SciML coverage is limited to the tested Tsit5 endpoint algorithms above; divergence
  AD, nonempty-state sensitivities and higher-order solve rules are not covered.
- Broyden stores a dense inverse Jacobian; real DEQs use limited-memory Broyden.
- Built-in explicit methods have no error estimate and may diverge on coarse or stiff
  problems. The optional solvers do not remove inverse conditioning or establish GPU support.
- Every inversion returns a `DiracBelief`.

## API

```@index
Modules = [ImplicitLayers]
```

```@autodocs
Modules = [ImplicitLayers]
```

## Game quantities and conformance

The default LuxFactor requires a Dirac input observation `y` and contributes zero game
energy and entropy. With `residual=true`, its game instead scores Dirac endpoints in
the first/second channel order, with zero point-belief entropy. DEQFactor and NeuralODEFactor implement two games:

- `free_energy(f, π, y, ps, st)` scores incoming Dirac points: `π` on the first channel
  (`x` or start), `y` on the second (`z` or end), following GaussianFactor's fixed order.
  It equals `Mycelium.local_free_energy` for those messages. Entropy is scalar zero;
  non-Dirac arguments throw.
- `lens = first(assemble(f, p, ps, st))` restricts the relation to one polarity.
  `free_energy(lens, π, inputs, ps, st)` runs that inversion and scores its returned point:
  residual energy plus an anchor on the unobserved channel. `inputs` contains a named Dirac
  observation (or that Dirac directly). A trivial prior costs zero; an incompatible hard
  Dirac prior costs infinity; a proper Gaussian costs ``\tfrac12(u^*-\mu)^\top\Lambda(u^*-\mu)``
  relative to its mean, without a normalization constant. The inversion still uses the prior
  only as a warm start (DEQ), or ignores it (ODE). Soft observations, singular Gaussian
  anchors and other prior families throw. Channel precision does not soften Dirac priors.

The lens game is Definition 20's expectation under the point inversion, with zero
point-belief entropy. Both games return `(F, st)` in `ScalarEnergySpace()`;
`scalar_free_energy` also accepts these lenses and a scalarisation override `σ`.
Scoring arbitrary points uses `energy`/`scalar_energy` (L-ADR-0003), not a `free_energy`
keyword.

If the residual is affine in the unknown and vanishes at the solved point, the incoming
energy equals the solved residual energy plus a Gaussian anchor with precision ``J_u^\top J_u``
centered at the incoming unknown. `energy_jacobian` supplies this metric. The tests derive
`1/8` for `g(z,x)=z/2+x` at `x=1,z=3` in both directions, and `2` for zero ODE dynamics
at endpoints `1,3` in both directions. A nonlinear quadratic DEQ checks cubic absolute
energy error for the first-order Gauss–Newton approximation. The common factor checker
now covers the complete point game for both factors without LP-009 exemptions.

Solved-state IFT AD accepts `AutoForwardDiff()`, `AutoZygote()`, `AutoEnzyme()` and
`AutoMooncake()`. Load DI, the backend and (for Enzyme/Mooncake) ForwardDiff; missing
extensions and other solve backends are rejected before solving. Pass the backend to
`solve_state` and the outer derivative. Enzyme outer closures that capture arrays
need DI constant contexts or `function_annotation=Enzyme.Const`; see the implementation note `solve.md` for
the limitation and reproducer. GPU and higher-order IFT derivatives remain untested.
