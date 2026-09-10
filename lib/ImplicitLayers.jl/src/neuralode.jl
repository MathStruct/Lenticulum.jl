# ---------------------------------------------------------------------------
# The NeuralODE factor:   r(z₀, z₁) = z₁ - Φ_{t₀→t₁}(z₀)
#
# where Φ is the flow of  dz/dt = f_θ(z,t).  Unlike the DEQ, BOTH directions are integrations:
#
#     z₀ observed  →  z₁ = Φ_{t₀→t₁}(z₀)      integrate forwards
#     z₁ observed  →  z₀ = Φ_{t₁→t₀}(z₁)      integrate backwards
#
# and the second is exact in exact arithmetic because a flow is a diffeomorphism. No solver,
# no invertible-architecture constraint (RealNVP, coupling layers), no learned inverse. This
# is the cleanest bidirectional factor in the project, and the reason is a theorem about ODEs
# rather than anything about neural networks.
#
# See `neuralode.md` and `NeuralODE as an Invertible Factor.md`.
# ---------------------------------------------------------------------------

"""
    NeuralODEFactor(dynamics, dim; tspan = (0.0, 1.0), integrator = RK4Integrator(),
                    channels = (:z0, :z1), input = default_dynamics_input)

A neural ODE as a **bidirectional** factor: the relation
``\\{(z_0,z_1) : z_1 = \\Phi_{t_0\\to t_1}(z_0)\\}`` where ``\\Phi`` is the flow of
``dz/dt = f_\\theta(z,t)``.

`dynamics` is any `LuxCore.AbstractLuxLayer` computing ``f_\\theta(z,t)`` — the same object
you would hand to `DiffEqFlux.NeuralODE`. Parameters and state are its own.

```julia
f = NeuralODEFactor(my_net, 4; tspan = (0.0, 1.0), integrator = RK4Integrator(steps = 100))
```

Both channels have dimension `dim` — a flow cannot change the dimension of its state, which is
why this factor takes one `dim` rather than two and why a NeuralODE cannot be used as an
encoder that compresses. Contrast [`DEQFactor`](@ref), whose two channels are independent.

!!! note "Compared to DiffEqFlux.NeuralODE"
    `NeuralODE(model, tspan, Tsit5())` is a Lux layer: `(n)(x, ps, st)` builds an
    `ODEProblem` and returns `(ODESolution, st)`. Direction fixed at construction, output a
    solution object. Wrapping *that* gives you [`LuxFactor`](@ref) and one polarity; this
    factor keeps the flow and gets two.
"""
struct NeuralODEFactor{names,D,I<:AbstractIntegrator,T,F} <:
       LenticulumCore.AbstractLenticulumFactor
    dim::Int
    dynamics::D
    integrator::I
    tspan::T
    input::F
    channelnames::NamedTuple{names}
end

"""
    default_dynamics_input(z, t) = (z, t)

How the dynamics layer is called: `LuxCore.apply(dynamics, input(z, t), ps, st)`.
"""
default_dynamics_input(z, t) = (z, t)

function NeuralODEFactor(
    dynamics, dim::Int;
    tspan = (0.0, 1.0),
    integrator::AbstractIntegrator = RK4Integrator(),
    channels::Tuple{Symbol,Symbol} = (:z0, :z1),
    input = default_dynamics_input,
)
    dim > 0 || throw(ArgumentError("dim must be positive; got $dim"))
    tspan[1] == tspan[2] &&
        throw(ArgumentError("tspan must have distinct endpoints; got $tspan"))
    names = NamedTuple{channels}((channels[1], channels[2]))
    return NeuralODEFactor(dim, dynamics, integrator, (float(tspan[1]), float(tspan[2])),
                           input, names)
end

startchannel(f::NeuralODEFactor{names}) where {names} = names[1]
endchannel(f::NeuralODEFactor{names}) where {names} = names[2]

LuxCore.initialparameters(rng::AbstractRNG, f::NeuralODEFactor) =
    LuxCore.initialparameters(rng, f.dynamics)
LuxCore.initialstates(rng::AbstractRNG, f::NeuralODEFactor) =
    LuxCore.initialstates(rng, f.dynamics)
LuxCore.parameterlength(f::NeuralODEFactor) = LuxCore.parameterlength(f.dynamics)
LuxCore.statelength(f::NeuralODEFactor) = LuxCore.statelength(f.dynamics)

LenticulumCore.channels(f::NeuralODEFactor{names}) where {names} =
    (LenticulumCore.Channel(names[1], f.dim), LenticulumCore.Channel(names[2], f.dim))

LenticulumCore.islearnable(::NeuralODEFactor) = true

"""
    LenticulumCore.supported_polarities(f::NeuralODEFactor)

Both of them, always, with no dimension condition and no solver — because a flow is
invertible by construction.

This is the only factor in the project for which both directions are *equally* cheap and
*equally* exact. `GaussianFactor` is bidirectional but its two directions do different
arithmetic (a pushforward vs a likelihood); `DEQFactor`'s reverse direction is a root-find
that may not converge; this one runs the identical integrator with the endpoints swapped.
"""
function LenticulumCore.supported_polarities(f::NeuralODEFactor{names}) where {names}
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    return (LenticulumCore.Polarity(NamedTuple{names}((O, U))),
            LenticulumCore.Polarity(NamedTuple{names}((U, O))))
end

function LenticulumCore.supports_polarity(
    f::NeuralODEFactor{names}, p::LenticulumCore.Polarity
) where {names}
    keys(p) == names || return false
    a, b = p[names[1]], p[names[2]]
    return (a isa LenticulumCore.Observed && b isa LenticulumCore.Unobserved) ||
           (a isa LenticulumCore.Unobserved && b isa LenticulumCore.Observed)
end

LenticulumCore.energyspace(f::NeuralODEFactor) = LenticulumCore.EuclideanEnergySpace(f.dim)
LenticulumCore.scalarisation(::NeuralODEFactor) = LenticulumCore.SquaredNorm()

"""
    vectorfield(f, ps, st) -> (vf, stref)

The dynamics as a plain closure `vf(z, t)`, plus the `Ref` accumulating the threaded Lux
state.

The `Ref` is a wart with a cause: `integrate` wants a pure `(u,t) -> du` function, while
LuxCore wants `st` threaded in and out of every call. Something has to hold the state across
the integrator's inner calls, and a `Ref` is the smallest thing that does. See
`neuralode.md` §4.1.
"""
function vectorfield(f::NeuralODEFactor, ps, st)
    stref = Ref(st)
    vf = function (z, t)
        du, s = LuxCore.apply(f.dynamics, f.input(z, t), ps, stref[])
        stref[] = s
        return du
    end
    return (vf, stref)
end

"""
    flow_forward(f, z₀, ps, st) -> (z₁, st)
    flow_reverse(f, z₁, ps, st) -> (z₀, st)

Integrate ``t_0 \\to t_1`` and ``t_1 \\to t_0`` respectively. Same integrator, endpoints
swapped — that is the entire implementation of the reverse direction.
"""
function flow_forward(f::NeuralODEFactor, z₀, ps, st)
    vf, stref = vectorfield(f, ps, st)
    z₁ = integrate(vf, z₀, f.tspan[1], f.tspan[2], f.integrator)
    return (z₁, stref[])
end

function flow_reverse(f::NeuralODEFactor, z₁, ps, st)
    vf, stref = vectorfield(f, ps, st)
    z₀ = integrate(vf, z₁, f.tspan[2], f.tspan[1], f.integrator)
    return (z₀, stref[])
end

"""
    flow_logdet(f, z₀, ps, st) -> (z₁, Δlogdet, st)

Forward flow together with ``\\int \\operatorname{tr}\\partial_z f_\\theta\\,dt``, so that

```math
\\log p(z_1) = \\log p(z_0) - \\Delta\\!\\log\\!\\det
```

the instantaneous change of variables. **This is the piece that would make the factor
transport a density rather than a point** — and it cannot be used for that today, because the
belief type it would need (`GaussianBelief`, or a normalising-flow belief) is not reachable
from a `lib/` package. See `neuralode.md` §4.2.
"""
function flow_logdet(f::NeuralODEFactor, z₀, ps, st)
    vf, stref = vectorfield(f, ps, st)
    z₁, Δ = integrate_with_divergence(vf, z₀, f.tspan[1], f.tspan[2], f.integrator)
    return (z₁, Δ, stref[])
end

"""
    residual(f::NeuralODEFactor, z₀, z₁, ps, st) -> (r, st)

``r = z_1 - \\Phi_{t_0\\to t_1}(z_0)``, the vector energy.
"""
function residual(f::NeuralODEFactor, z₀, z₁, ps, st)
    ẑ₁, st = flow_forward(f, z₀, ps, st)
    return (z₁ .- ẑ₁, st)
end

LenticulumCore.energy(f::NeuralODEFactor, x, a, y, ps, st) = residual(f, x, y, ps, st)

# --- The open model --------------------------------------------------------

"""
    NeuralODEModel(factor, polarity)

The open model the polarity selects. Pure, and — unusually — *exactly invertible*, so the
same object serves as its own inverse with the time direction flipped.
"""
struct NeuralODEModel{F<:NeuralODEFactor,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::NeuralODEModel) = true
LenticulumCore.latentspace(::NeuralODEModel) = nothing

"""
    LenticulumCore.assemble(f::NeuralODEFactor, p, ps, st)

Produces a lens with `ExactInversion` — not `SolverInversion`.

That is a real claim, not an optimism: the inverse of a flow is the flow with negative time,
and it is exact in exact arithmetic. What it is *not* is exact in floating point, and the
gap is the integrator's round-trip error. `neuralode.md` §4.3 argues that
`ExactInversion` is nonetheless the right label, and says what would change if you disagree.
"""
LenticulumCore.assemble(f::NeuralODEFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(NeuralODEModel(f, p), LenticulumCore.ExactInversion()), st)

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:NeuralODEModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    f, p = lens.model.factor, lens.model.polarity
    target = only(LenticulumCore.unobserved_channels(p))
    return _flow_for(f, target, inputs, ps, st)
end

function _flow_for(f::NeuralODEFactor, target::Symbol, inputs, ps, st)
    s, e = startchannel(f), endchannel(f)
    if target === e
        z₀ = _point(_get(inputs, s))
        z₀ === nothing && return (LenticulumCore.TrivialBelief(), st)
        z₁, st = flow_forward(f, z₀, ps, st)
        return (LenticulumCore.DiracBelief(z₁), st)
    elseif target === s
        z₁ = _point(_get(inputs, e))
        z₁ === nothing && return (LenticulumCore.TrivialBelief(), st)
        z₀, st = flow_reverse(f, z₁, ps, st)
        return (LenticulumCore.DiracBelief(z₀), st)
    end
    throw(ArgumentError("channel :$target is not a channel of this NeuralODEFactor"))
end

"""
    Mycelium.factor_message(f::NeuralODEFactor, target, polarity, inputs, prior, ps, st)

A `DiracBelief` on `target`, obtained by integrating in the appropriate direction.

The `prior` is genuinely unused here — unlike [`DEQFactor`](@ref), where it warm-starts the
solver. A flow has nothing to warm-start.
"""
function Mycelium.factor_message(
    f::NeuralODEFactor{names}, target::Symbol, polarity, inputs, prior, ps, st
) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this NeuralODEFactor (has $names)"))
    return _flow_for(f, target, inputs, ps, st)
end

"""
    Mycelium.local_free_energy(f::NeuralODEFactor, msgs, ps, st)

``\\tfrac12\\|z_1 - \\Phi(z_0)\\|^2`` when both endpoints carry points, else `0.0`.

On the relation this is **identically zero**, because the flow is exact and both endpoints
agree by construction. That makes it a poor diagnostic and a good invariant: a nonzero value
means either the integrator is inaccurate or the two messages disagree, and the test suite
uses it that way.
"""
function Mycelium.local_free_energy(f::NeuralODEFactor, msgs, ps, st)
    z₀ = _point(_get(msgs, startchannel(f)))
    z₁ = _point(_get(msgs, endchannel(f)))
    (z₀ === nothing || z₁ === nothing) && return (0.0, st)
    r, st = residual(f, z₀, z₁, ps, st)
    return (sum(abs2, r) / 2, st)
end
