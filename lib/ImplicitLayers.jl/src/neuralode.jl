# ---------------------------------------------------------------------------
# The NeuralODE factor:   r(z₀, z₁) = z₁ - Φ_{t₀→t₁}(z₀)
#
# where Φ is the flow of  dz/dt = f_θ(z,t).  Unlike the DEQ, BOTH directions are integrations:
#
#     z₀ observed  →  z₁ = Φ_{t₀→t₁}(z₀)      integrate forwards
#     z₁ observed  →  z₀ = Φ_{t₁→t₀}(z₁)      integrate backwards
#
# The continuous inverse is another flow. Numerical time reversal only inverts the discrete
# map for symmetric methods on matching grids; Euler/RK4 merely approximate that inverse.
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
    channels[1] === channels[2] && throw(ArgumentError(
        "start and end channels must differ; got :$(channels[1]) for both"))
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

Both endpoint polarities are supported. Each runs the configured integrator with the
appropriate endpoints. Built-in fixed-step methods use equal work in both directions, but
reverse integration need not invert the discrete forward map. Adaptive/implicit solvers may
also require different work. Contracting flows amplify perturbations when inverted.
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

LenticulumCore.pointwise_energyspace(f::NeuralODEFactor) =
    LenticulumCore.EuclideanEnergySpace(f.dim)
LenticulumCore.pointwise_scalarisation(::NeuralODEFactor) = LenticulumCore.SquaredNorm()

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
    return _flow(f, z₀, ps, st, f.tspan[1], f.tspan[2], f.integrator)
end
function flow_reverse(f::NeuralODEFactor, z₁, ps, st)
    return _flow(f, z₁, ps, st, f.tspan[2], f.tspan[1], f.integrator)
end
function _flow(f, u, ps, st, t0, t1, itg::Union{EulerIntegrator,RK4Integrator})
    h = (t1 - t0) / nsteps(itg)
    return foldl(1:nsteps(itg); init = (u, st)) do (z, state), k
        _lux_step(f, z, ps, state, t0 + (k - 1) * h, h, itg)
    end
end
_rhs(f, z, t, ps, st) = LuxCore.apply(f.dynamics, f.input(z, t), ps, st)
function _lux_step(f, z, ps, st, t, h, ::EulerIntegrator)
    k, st = _rhs(f, z, t, ps, st)
    return z .+ h .* k, st
end
function _lux_step(f, z, ps, st, t, h, ::RK4Integrator)
    k1, s1 = _rhs(f, z, t, ps, st)
    k2, s2 = _rhs(f, z .+ (h / 2) .* k1, t + h / 2, ps, s1)
    k3, s3 = _rhs(f, z .+ (h / 2) .* k2, t + h / 2, ps, s2)
    k4, s4 = _rhs(f, z .+ h .* k3, t + h, ps, s3)
    return z .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4), s4
end
# Adaptive SciML and divergence paths retain their existing Ref interface.
_empty_flow_state(st::NamedTuple) = all(_empty_flow_state, values(st))
_empty_flow_state(st::Tuple) = all(_empty_flow_state, st)
_empty_flow_state(st) = false

function _flow(f, u, ps, st, t0, t1, itg::AbstractIntegrator)
    itg isa SciMLIntegrator && itg.sensealg !== nothing &&
        Base.get_extension(@__MODULE__, :ImplicitLayersSciMLSensitivityExt) === nothing &&
        throw(ArgumentError("SciMLIntegrator sensealg requires loading SciMLSensitivity"))
    vf, stref = vectorfield(f, ps, st)
    z = integrate(vf, u, t0, t1, itg)
    return z, stref[]
end

"""
    flow_logdet(f, z₀, ps, st) -> (z₁, Δlogdet, st)

Forward flow together with ``\\int \\operatorname{tr}\\partial_z f_\\theta\\,dt``, so that

```math
\\log p(z_1) = \\log p(z_0) - \\Delta\\!\\log\\!\\det
```

the instantaneous change of variables. **This is the piece that would make the factor
transport a density rather than a point**. It is not used for that yet: `GaussianBelief` is
now reachable (in `LenticulumCore`), but a flow needs a non-Gaussian, pushforward belief, which
does not exist. See `neuralode.md` §4.2.
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

The open model the polarity selects. Both directions integrate the dynamics; the inversion
label depends on the numerical method, time grid and deterministic dynamics assumption.
"""
struct NeuralODEModel{F<:NeuralODEFactor,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::NeuralODEModel) = true
LenticulumCore.latentspace(::NeuralODEModel) = nothing

"""
    LenticulumCore.assemble(f::NeuralODEFactor, p, ps, st)

Select the inversion label with `inversion_kind(f.integrator, f.tspan...)`.
Euler/RK4 and adaptive SciML methods use `SolverInversion(integrator)`: reversing time
approximates the continuous inverse, not the inverse of the discrete residual map.
Symmetric fixed-step SciML methods on matching grids use `ExactInversion`, up to nonlinear
solve tolerance and roundoff. Nonempty Lux state conservatively keeps `SolverInversion`,
since state updates can change the dynamics. See
`neuralode.md` §4.3 for convergence and conditioning limitations.
"""
function LenticulumCore.assemble(f::NeuralODEFactor, p::LenticulumCore.Polarity, ps, st)
    # Conservatively avoid promising an inverse when threaded Lux state may change the RHS.
    kind = st isa NamedTuple && isempty(st) ? inversion_kind(f.integrator, f.tspan...) :
           LenticulumCore.SolverInversion(f.integrator)
    return (LenticulumCore.BayesianLens(NeuralODEModel(f, p), kind), st)
end

const _NeuralODELens = LenticulumCore.BayesianLens{<:NeuralODEModel,
    <:Union{LenticulumCore.ExactInversion,LenticulumCore.SolverInversion}}

function LenticulumCore.invert(lens::_NeuralODELens, π, inputs, ps, st)
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

Mycelium.uses_cavity(::NeuralODEFactor) = true

"""
    Mycelium.local_free_energy(f::NeuralODEFactor, msgs, ps, st)

``\\tfrac12\\|z_1 - \\Phi(z_0)\\|^2`` when both endpoints carry points, else `0.0`.

This is zero when the end point was produced by the same numerical forward map.
Reverse-solved endpoints can have nonzero energy from round-trip error. The energy measures
endpoint consistency with the numerical map; it cannot establish continuous-flow accuracy.
"""
function Mycelium.local_free_energy(f::NeuralODEFactor, msgs, ps, st)
    z₀ = _point(_get(msgs, startchannel(f)))
    z₁ = _point(_get(msgs, endchannel(f)))
    (z₀ === nothing || z₁ === nothing) && return (0.0, st)
    r, st = residual(f, z₀, z₁, ps, st)
    return (sum(abs2, r) / 2, st)
end

# Same fixed channel order as GaussianFactor, independent of inversion polarity.
LenticulumCore.free_energy(f::NeuralODEFactor, π, y, ps, st) =
    Mycelium.local_free_energy(f, _incoming_point_messages(f, π, y), ps, st)
function LenticulumCore.entropy(f::NeuralODEFactor, π, y, ps, st)
    _incoming_point_messages(f, π, y)
    return (0.0, st)
end
