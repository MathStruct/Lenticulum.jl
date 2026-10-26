# ---------------------------------------------------------------------------
# LuxFactor: any Lux layer, as a factor. One polarity.
#
# This is the answer to "wrap the SciML models" taken literally, and it works today on
# `DeepEquilibriumNetwork`, `NeuralODE`, `NeuralDSDE`, `Chain`, anything — because all of them
# are `AbstractLuxLayer`s and nothing else is required.
#
# It is also the demonstration of why that is not enough. A Lux layer knows which side is
# input; `supported_polarities` therefore returns ONE element, `isunidirectional` is `true`,
# and the factor can only ever be run the way it was built. Wrapping a DEQ this way gives you
# a DEQ. Wrapping its *residual* gives you a relation — see `deq.md`.
#
#     LuxFactor(DeepEquilibriumNetwork(cell, solver), :x => :z)   → 1 polarity
#     DEQFactor(cell, (x = n, z = n))                              → 2 polarities
#
# Same network, same parameters, different amount of the framework used.
#
# See `luxfactor.md` and `DEQ as a Relation.md` §3.
# ---------------------------------------------------------------------------

"""
    LuxFactor(layer, :in => :out; dims = (nothing, nothing), postprocess = identity, residual = false)

Any `LuxCore.AbstractLuxLayer` as a **unidirectional** factor.

The point of least resistance for reusing the SciML ecosystem: a `DeepEquilibriumNetwork`, a
`NeuralODE`, a `Chain`, a `Boltz` vision backbone all satisfy the interface, so all of them
wrap without this package knowing anything about them.

`postprocess` adapts the layer's output to a plain array — needed for the SciML layers whose
call returns a solution object rather than a vector:

```julia
# DiffEqFlux's NeuralODE returns an ODESolution
LuxFactor(NeuralODE(net, (0.0, 1.0), Tsit5()), :x => :y;
          postprocess = sol -> Array(sol)[:, end])

# DeepEquilibriumNetworks returns the steady state directly, and reports the solve in `st`
LuxFactor(DeepEquilibriumNetwork(cell, NewtonRaphson()), :x => :z)
```

Set `residual=true` (with an explicit output dimension in `dims`) to score the
endpoint vector `y - layer(x)` with a squared norm. Core parameter derivatives and
Mycelium's fixed-message `parameter_step` can then train this factor. The default
keeps zero energy for compatibility with downstream-loss graphs. Direct losses on
`apply_layer` can also be differentiated with `LenticulumCore.parameter_gradient`.

!!! warning "One polarity, by construction"
    `isunidirectional(::LuxFactor) == true`. A Lux layer *is* a lens
    ([[Lux as a Parametric Lens]]); a factor only *becomes* one once a polarity is chosen, and
    a layer has already chosen. So this wrapper buys graph membership, parameter management
    and free-energy accounting — and none of the bidirectionality the rest of the project is
    about. Use it when the model genuinely is a function; use [`DEQFactor`](@ref) or
    [`NeuralODEFactor`](@ref) when it is not.
"""
struct LuxFactor{L,I,O,D,P} <: LenticulumCore.AbstractLenticulumFactor
    layer::L
    inchannel::I
    outchannel::O
    dims::D
    postprocess::P
    scored::Bool
end

function LuxFactor(
    layer, io::Pair{Symbol,Symbol};
    dims = (nothing, nothing),
    postprocess = identity,
    residual::Bool = false,
)
    first(io) === last(io) && throw(ArgumentError(
        "input and output channels must differ; got :$(first(io)) for both"))
    length(dims) == 2 || throw(ArgumentError("dims must be a 2-tuple; got $dims"))
    for d in dims
        (d === nothing || (d isa Int && d > 0)) ||
            throw(ArgumentError("channel dimensions must be positive Ints or nothing; got $dims"))
    end
    residual && dims[2] === nothing && throw(ArgumentError(
        "residual=true needs an explicit output dimension in dims"))
    return LuxFactor(layer, first(io), last(io), (dims[1], dims[2]), postprocess, residual)
end

LuxCore.initialparameters(rng::AbstractRNG, f::LuxFactor) =
    LuxCore.initialparameters(rng, f.layer)
LuxCore.initialstates(rng::AbstractRNG, f::LuxFactor) = LuxCore.initialstates(rng, f.layer)
LuxCore.parameterlength(f::LuxFactor) = LuxCore.parameterlength(f.layer)
LuxCore.statelength(f::LuxFactor) = LuxCore.statelength(f.layer)

LenticulumCore.channels(f::LuxFactor) = (
    LenticulumCore.Channel(f.inchannel, f.dims[1]),
    LenticulumCore.Channel(f.outchannel, f.dims[2]),
)

LenticulumCore.supported_polarities(f::LuxFactor) = (
    LenticulumCore.Polarity(NamedTuple{(f.inchannel, f.outchannel)}(
        (LenticulumCore.Observed(), LenticulumCore.Unobserved()))),
)

function LenticulumCore.supports_polarity(f::LuxFactor, p::LenticulumCore.Polarity)
    keys(p) == (f.inchannel, f.outchannel) || return false
    return p[f.inchannel] isa LenticulumCore.Observed &&
           p[f.outchannel] isa LenticulumCore.Unobserved
end

LenticulumCore.islearnable(f::LuxFactor) = LuxCore.parameterlength(f.layer) > 0

# The default delegates loss accounting to other graph nodes. Scored wrappers
# opt into endpoint residual energy while keeping a scalar free-energy space.
LenticulumCore.energyspace(::LuxFactor) = LenticulumCore.ScalarEnergySpace()
LenticulumCore.pointwise_energyspace(f::LuxFactor) = f.scored ?
    LenticulumCore.EuclideanEnergySpace(something(f.dims[2], 1)) : LenticulumCore.ScalarEnergySpace()
LenticulumCore.pointwise_scalarisation(f::LuxFactor) = f.scored ?
    LenticulumCore.SquaredNorm() : LenticulumCore.IdentityScalarisation()
function LenticulumCore.energy(f::LuxFactor, x, a, y, ps, st)
    f.scored || return (0.0, st)
    pred, st = apply_layer(f, x, ps, st)
    return (y .- pred, st)
end

"""
    apply_layer(f::LuxFactor, x, ps, st) -> (y, st)

One forward pass, with `postprocess` applied to the output.
"""
function apply_layer(f::LuxFactor, x, ps, st)
    y, st = LuxCore.apply(f.layer, x, ps, st)
    return (f.postprocess(y), st)
end

struct LuxModel{F<:LuxFactor} <: LenticulumCore.AbstractOpenModel
    factor::F
end
LenticulumCore.ispure(::LuxModel) = true
LenticulumCore.latentspace(::LuxModel) = nothing

# `TrivialInversion` is the honest label: there is nothing to invert. The layer runs forwards
# and that is all it can do — cf. `lens.md`, where TrivialInversion is what a prior gets
# because "there is nothing to infer".
LenticulumCore.assemble(f::LuxFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(LuxModel(f), LenticulumCore.TrivialInversion()), st)

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:LuxModel,LenticulumCore.TrivialInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    x = _point(_get(inputs, f.inchannel))
    x === nothing && return (LenticulumCore.TrivialBelief(), st)
    y, st = apply_layer(f, x, ps, st)
    return (LenticulumCore.DiracBelief(y), st)
end

"""
    Mycelium.factor_message(f::LuxFactor, target, polarity, inputs, prior, ps, st)

A `DiracBelief` on the output channel. Asking for a message on the **input** channel throws a
`PolarityError`-shaped `ArgumentError`, because that is precisely the direction a Lux layer
does not have.
"""
function Mycelium.factor_message(
    f::LuxFactor, target::Symbol, polarity, inputs, prior, ps, st
)
    if target === f.inchannel
        throw(ArgumentError(
            "a LuxFactor cannot emit on its input channel :$(f.inchannel) — a Lux layer has \
             one direction. Use DEQFactor or NeuralODEFactor if the model is a relation."))
    end
    target === f.outchannel || throw(ArgumentError(
        "channel :$target is not a channel of this LuxFactor \
         (has :$(f.inchannel), :$(f.outchannel))"))
    x = _point(_get(inputs, f.inchannel))
    x === nothing && return (LenticulumCore.TrivialBelief(), st)
    y, st = apply_layer(f, x, ps, st)
    return (LenticulumCore.DiracBelief(y), st)
end

function Mycelium.local_free_energy(f::LuxFactor, msgs, ps, st)
    f.scored || return (0.0, st)
    x, y = _point(_get(msgs, f.inchannel)), _point(_get(msgs, f.outchannel))
    (x === nothing || y === nothing) && return (0.0, st)
    return LenticulumCore.scalar_energy(f, x, nothing, y, ps, st)
end
Mycelium.uses_cavity(f::LuxFactor) = f.scored

# The default game delegates loss accounting; scored wrappers use incoming endpoints.
# Both use zero point-belief entropy.
function LenticulumCore.free_energy(f::LuxFactor, π, y::LenticulumCore.DiracBelief, ps, st)
    f.scored || return (0.0, st)
    π isa LenticulumCore.DiracBelief || throw(ArgumentError("scored LuxFactor needs Dirac endpoints"))
    return LenticulumCore.scalar_energy(f, _point(π), nothing, _point(y), ps, st)
end
LenticulumCore.entropy(::LuxFactor, π, y::LenticulumCore.DiracBelief, ps, st) = (0.0, st)
for op in (:free_energy, :entropy)
    @eval LenticulumCore.$op(::LuxFactor, π, y, ps, st) = throw(ArgumentError(
        "LuxFactor point-game quantities require a Dirac input observation; no distributional inversion entropy is supplied"))
end
