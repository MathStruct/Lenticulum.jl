"""
    ImplicitLayers

SciML's implicit layers — deep equilibrium networks and neural ODEs — as Lenticulum factors.
The **equilibrium** family of `Implicit Learners.md`, alongside `VariationalDiffusion.jl`'s
diffusion family.

## The thesis

`DeepEquilibriumNetworks.jl` and `DiffEqFlux.jl` both hand you an `AbstractLuxLayer`: a
function ``x \\mapsto y`` whose direction is fixed when you construct it. But both are built
*out of* a relation —

```math
\\text{DEQ:}\\quad z = g_\\theta(z,x)
\\qquad\\qquad
\\text{NeuralODE:}\\quad \\frac{dz}{dt} = f_\\theta(z,t)
```

— and the layer is that relation with one direction chosen and the solve sealed inside. So
there are two ways to wrap them, and the difference is the whole point of this package:

| wrapper | what you give it | polarities | what you get |
|---|---|---|---|
| [`LuxFactor`](@ref) | the assembled `DeepEquilibriumNetwork` / `NeuralODE` | **1** | graph membership; a Lux layer with extra steps |
| [`DEQFactor`](@ref) | the *cell* ``g_\\theta`` | **2** (if square) | the relation; solvable for either channel |
| [`NeuralODEFactor`](@ref) | the *dynamics* ``f_\\theta`` | **2**, always | the flow; invertible by construction |

`LuxFactor` works today on anything in the SciML ecosystem and asks nothing of it. The other
two need the inner network rather than the assembled layer, and give back the bidirectionality
the rest of the project is about.

## The files

| file | note | supplies |
|---|---|---|
| `solve.jl` | [[solve]] | Picard, Broyden, `SolveReport`, FD Jacobians, the IFT |
| `deq.jl` | [[deq]] | `DEQFactor` — the fixed-point relation |
| `flow.jl` | [[flow]] | fixed-step RK, forwards **and backwards**, CNF divergence |
| `neuralode.jl` | [[neuralode]] | `NeuralODEFactor` — the flow relation |
| `luxfactor.jl` | [[luxfactor]] | `LuxFactor` — any Lux layer, one polarity |

Concept notes: [[The Equilibrium Family]], [[DEQ as a Relation]],
[[NeuralODE as an Invertible Factor]].

## Two properties worth knowing up front

**No SciML dependency, and no AD.** Dependencies are `LuxCore`, `Random`, `LinearAlgebra` —
the same choice `VariationalDiffusion.jl` makes and for the same reason. Solvers are
derivative-free (Broyden is what `DeepEquilibriumNetworks.jl` uses anyway) and integrators are
fixed-step explicit. Derivative information, where needed, comes from finite differences and
is honestly labelled a test-scale tool: the real answer is a VJP from automatic
differentiation. See [[solve]] §4.

**Every inversion returns a `DiracBelief`.** A root-find and an ODE solve both produce a
*point*. Transporting a distribution would need `GaussianBelief`, which lives in the top-level
`Lenticulum` package that no `lib/` package may depend on — the third factor package in a row
to hit this. See [[The Equilibrium Family]] §5.
"""
module ImplicitLayers

using DispatchDoctor: @stable
using LinearAlgebra: LinearAlgebra, I, dot, lu, issuccess, SingularException, tr
using Random: Random, AbstractRNG
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium

# --- Shared belief helpers -------------------------------------------------
# Reading a *point* out of an incoming message. Every factor here needs it, and none of them
# needs anything else from a belief, because every solver in this package takes a point and
# returns a point. That narrowness is itself the finding recorded in
# `The Equilibrium Family.md` §5.
_get(inputs, k) = (inputs isa NamedTuple && haskey(inputs, k)) ? getfield(inputs, k) : nothing
_point(::Nothing) = nothing
_point(::LenticulumCore.TrivialBelief) = nothing
_point(b::LenticulumCore.DiracBelief) = _vec(b.value)
_point(::LenticulumCore.SampleBelief) = nothing   # a particle set has no single point
_point(b) = hasproperty(b, :η) ? _canonical_mean(b) : nothing
_canonical_mean(b) = try
    Mycelium.belief_mean(b)
catch
    nothing
end
_vec(v::AbstractVector) = float.(v)
_vec(v::Real) = [float(v)]

include("solve.jl")
include("deq.jl")
include("flow.jl")
include("neuralode.jl")
include("luxfactor.jl")

# --- Solvers ---------------------------------------------------------------
export SolveReport, AbstractRootSolver, PicardSolver, BroydenSolver
export solve_root, fd_jacobian, ift_sensitivity

# --- DEQ -------------------------------------------------------------------
export DEQFactor, DEQModel, default_cell_input
export cell_value, solve_state, solve_input, deq_sensitivity
export inchannel, statechannel, indim, statedim

# --- Flows -----------------------------------------------------------------
export AbstractIntegrator, EulerIntegrator, RK4Integrator, nsteps
export integrate, integrate_with_divergence

# --- NeuralODE -------------------------------------------------------------
export NeuralODEFactor, NeuralODEModel, default_dynamics_input
export vectorfield, flow_forward, flow_reverse, flow_logdet
export startchannel, endchannel

# --- Any Lux layer ---------------------------------------------------------
export LuxFactor, LuxModel, apply_layer

# `residual` is extended for both factor types; exported once.
export residual

end # module
