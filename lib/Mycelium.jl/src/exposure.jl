# ---------------------------------------------------------------------------
# Exposed parameters (L-SPEC-003 §1): a block of a factor's `ps` becomes a channel, so a
# variable in the graph can hold it. A prior on the weight is then an ordinary factor on that
# variable, two factors exposing a block to one variable share the weight, and inference over
# the weight is message passing like any other.
#
#   §1.1 every exposed channel hard-clamped (a Dirac): the factor is `f` with the blocks
#        substituted, generically, for every factor;
#   §1.2 an exposed block carrying a belief: the factor supplies `parameter_message` and
#        `parameter_free_energy` (the linear-Gaussian factor does, exactly, in `src/exposure.jl`
#        of the top-level package);
#   §1.3 weight tying is wiring: connect two exposed channels to one variable.
#
# See `exposure.md` and `L-SPEC-003 Exposed Parameters and Energy Channels.md`.
# ---------------------------------------------------------------------------

"""
    ExposedFactor

A factor with some parameter blocks moved from `ps` to channels; build it with
[`expose`](@ref).
"""
struct ExposedFactor{F,P,C,S} <: LenticulumCore.AbstractLenticulumFactor
    factor::F
    params::P        # NTuple of parameter names (keys of the factor's ps)
    chans::C         # NTuple of channel names, one per parameter
    shapes::S        # NTuple of the blocks' array sizes
    shadowed::Bool   # a channel of `factor` is named like one of its parameter blocks
end

"""
    expose(f, names; rng = Random.Xoshiro(0)) -> ExposedFactor

Move the parameter blocks `names` of `f` out of its `ps` and into channels. Each entry of
`names` is a parameter name (the channel gets the same name) or a pair `param => channel`.
A block must be an array; its channel carries `vec(block)`, column-major, of length
`length(block)`. Block sizes are read once from `initialparameters(rng, f)`.

- With every exposed channel observed as a point (`DiracBelief`), messages and free energies
  are exactly `f`'s with the blocks substituted (L-SPEC-003 §1.1).
- An exposed channel may also carry a belief, or be the target of a message, where `f`
  implements [`supports_parameter_polarity`](@ref), [`parameter_message`](@ref) and
  [`parameter_free_energy`](@ref) (§1.2). Otherwise such a message throws.
- Connecting exposed channels of several factors to one variable ties the weight (§1.3).

The prior on an exposed weight is a separate factor on its variable, as for any variable.
Game-level methods (`assemble`, `energy`, `free_energy`) are not forwarded; the exposed factor
is for graphs.
"""
function expose(f, names::Tuple; rng::AbstractRNG = Random.Xoshiro(0))
    params = map(n -> n isa Pair ? first(n) : n, names)
    chans = map(n -> n isa Pair ? last(n) : n, names)
    all(p -> p isa Symbol, params) && all(c -> c isa Symbol, chans) ||
        throw(ArgumentError("expose takes parameter names or `param => channel` pairs of Symbols"))
    allunique(params) || throw(ArgumentError("a parameter is exposed twice: $(params)"))
    own = Tuple(LenticulumCore.channelname(c) for c in LenticulumCore.channels(f))
    clash = filter(c -> c in own, chans)
    isempty(clash) || throw(ArgumentError(
        "exposed channel names $(clash) clash with the factor's channels $(own); rename with `param => channel`"))
    allunique(chans) || throw(ArgumentError("two exposed blocks share a channel name: $(chans)"))
    ps = LuxCore.initialparameters(rng, f)
    # `parameter_message` sees channels and exposed parameters in one namespace, so a channel
    # named like an exposed block would be read as that block. Such a factor keeps the clamped
    # path (§1.1, where the two live in separate containers) and refuses the belief path. A
    # factor whose belief path reads further parameter names checks those itself.
    shadowed = any(c -> c in params, own)
    shapes = map(params) do p
        haskey(ps, p) || throw(ArgumentError("$(typeof(f)) has no parameter block :$p; it has $(keys(ps))"))
        getfield(ps, p) isa AbstractArray || throw(ArgumentError(
            "parameter block :$p is a $(typeof(getfield(ps, p))), not an array; only array blocks can be exposed"))
        size(getfield(ps, p))
    end
    return ExposedFactor(f, params, chans, shapes, shadowed)
end
expose(f, name::Union{Symbol,Pair}; kw...) = expose(f, (name,); kw...)

"""
    supports_parameter_polarity(f, polarity) -> Bool

Whether `f` can compute a message when some of its parameter blocks are channels carrying
beliefs or are the target. `polarity` is keyed by `f`'s channel names **and the exposed
parameter names**. Defaults to `false`: only the hard-clamped case (§1.1) is supported.
"""
supports_parameter_polarity(f, polarity) = false

"""
    parameter_message(f, target, polarity, msgs, ps, st) -> (belief, st)

The likelihood message from `f` to `target` (one of `f`'s channels or an exposed parameter
name) when exposed blocks carry beliefs. `msgs` holds the incoming beliefs keyed by channel
and parameter name; `ps` lacks the exposed blocks. A belief on a block is over `vec(block)`.
Like any message it is a likelihood (L-ADR-0006): the prior on a weight is its own factor.
"""
parameter_message(f, target, polarity, msgs, ps, st) = throw(ArgumentError(
    "$(typeof(f)) has no `parameter_message`: an exposed block may only be observed as a point \
     (DiracBelief) and may not be a message target (L-SPEC-003 §1.2)"))

"""
    parameter_free_energy(f, msgs, ps, st) -> (𝐅, st)

`f`'s Bethe contribution when exposed blocks carry beliefs: `local_free_energy` with the
factor belief ranging over those blocks too. Keys as in [`parameter_message`](@ref).
"""
parameter_free_energy(f, msgs, ps, st) = throw(ArgumentError(
    "$(typeof(f)) has no `parameter_free_energy`: an exposed block must be observed as a point \
     (DiracBelief) for the graph free energy (L-SPEC-003 §1.2)"))

_own_channels(e::ExposedFactor) = Tuple(LenticulumCore.channelname(c) for c in LenticulumCore.channels(e.factor))

LenticulumCore.channels(e::ExposedFactor) = (LenticulumCore.channels(e.factor)...,
    map((c, s) -> LenticulumCore.Channel(c, prod(s)), e.chans, e.shapes)...)

function LuxCore.initialparameters(rng::AbstractRNG, e::ExposedFactor)
    ps = LuxCore.initialparameters(rng, e.factor)
    keep = Tuple(k for k in keys(ps) if !(k in e.params))
    return NamedTuple{keep}(map(k -> getfield(ps, k), keep))
end
LuxCore.initialstates(rng::AbstractRNG, e::ExposedFactor) = LuxCore.initialstates(rng, e.factor)
LenticulumCore.islearnable(e::ExposedFactor) = LenticulumCore.islearnable(e.factor)
LenticulumCore.isfrozen(e::ExposedFactor) = LenticulumCore.isfrozen(e.factor)
LuxCore.parameterlength(e::ExposedFactor) = LuxCore.parameterlength(e.factor) - sum(prod, e.shapes)
for g in (:energyspace, :scalarisation, :pointwise_energyspace, :pointwise_scalarisation)
    @eval LenticulumCore.$g(e::ExposedFactor) = LenticulumCore.$g(e.factor)
end
uses_cavity(::ExposedFactor) = true

# Rename an exposed factor's polarity or messages to `f`'s view: own channels unchanged,
# exposed channels under their parameter names.
_param_key(e::ExposedFactor, k::Symbol) = (i = findfirst(==(k), e.chans); i === nothing ? k : e.params[i])
function _param_view(e::ExposedFactor, p::LenticulumCore.Polarity)
    ks = map(k -> _param_key(e, k), keys(p))
    return LenticulumCore.Polarity(NamedTuple{ks}(values(p.assignment)), NamedTuple{ks}(values(p.precisions)))
end
_param_view(e::ExposedFactor, nt::NamedTuple) =
    NamedTuple{map(k -> _param_key(e, k), keys(nt))}(values(nt))

function _restrict(p::LenticulumCore.Polarity, ks)
    return LenticulumCore.Polarity(NamedTuple{ks}(map(k -> p[k], ks)),
        NamedTuple{ks}(map(k -> LenticulumCore.channel_precision(p, k), ks)))
end
_restrict(nt::NamedTuple, ks) = NamedTuple{Tuple(k for k in ks if haskey(nt, k))}(
    Tuple(getfield(nt, k) for k in ks if haskey(nt, k)))

_all_exposed_observed(e, p) = all(c -> p[c] isa LenticulumCore.Observed, e.chans)

function LenticulumCore.supports_polarity(e::ExposedFactor, p::LenticulumCore.Polarity)
    Set(keys(p)) == Set(Tuple(LenticulumCore.channelname(c) for c in LenticulumCore.channels(e))) || return false
    own = _own_channels(e)
    _all_exposed_observed(e, p) && LenticulumCore.supports_polarity(e.factor, _restrict(p, own)) && return true
    return !e.shadowed && supports_parameter_polarity(e.factor, _param_view(e, p))
end

# Every accepted assignment, as L-SPEC-001 asks. Each assignment is its own `Polarity` type,
# so listing all 3ⁿ compiles that many: above five channels in all (243 candidates) the list
# throws, as the contract permits, and the structural predicates below answer lazily through
# `polarities` instead.
const _EXPOSED_ENUMERATION_CAP = 5
function LenticulumCore.supported_polarities(e::ExposedFactor)
    n = length(LenticulumCore.channels(e))
    n <= _EXPOSED_ENUMERATION_CAP || throw(ArgumentError(
        "an exposed factor with $n channels has 3^$n candidate polarities; use `supports_polarity` \
         or the lazy `LenticulumCore.polarities` instead of listing them"))
    return Tuple(LenticulumCore.polarities(e))
end
# Exact, by the lazy search, stopping at the first (resp. second) accepted assignment. A
# structural shortcut from `f`'s predicates and the canonical block targets was wrong for
# factors whose hooks accept other assignments (LP-019 fourth review). The cost: a search
# that finds nothing, or exactly one, still visits all 3ⁿ candidates.
LenticulumCore.hasnopolarity(e::ExposedFactor) = isempty(LenticulumCore.polarities(e))
LenticulumCore.isunidirectional(e::ExposedFactor) =
    length(collect(Iterators.take(LenticulumCore.polarities(e), 2))) == 1

_clamped_value(b::LenticulumCore.DiracBelief, shape) = reshape(b.value isa Real ? [b.value] : collect(b.value), shape)

# The exposed blocks as points, or `nothing` if any exposed input is missing or not a point.
function _clamped_parameters(e::ExposedFactor, inputs)
    vals = map(e.chans, e.shapes) do c, s
        b = haskey(inputs, c) ? getfield(inputs, c) : nothing
        b isa LenticulumCore.DiracBelief ? _clamped_value(b, s) : nothing
    end
    any(isnothing, vals) && return nothing
    return NamedTuple{e.params}(vals)
end

function factor_message(e::ExposedFactor, target::Symbol, polarity, inputs, prior, ps, st)
    own = _own_channels(e)
    clamped = _clamped_parameters(e, inputs)
    if target in own && clamped !== nothing
        return factor_message(e.factor, target, _restrict(polarity, own), _restrict(inputs, own),
                              prior, merge(ps, clamped), st)
    end
    e.shadowed && throw(ArgumentError(
        "a channel of $(typeof(e.factor)) is named like one of its parameter blocks, so exposed blocks \
         can only be observed as points here; rename the channel"))
    return parameter_message(e.factor, _param_key(e, target), _param_view(e, polarity),
                             _param_view(e, inputs), ps, st)
end

function local_free_energy(e::ExposedFactor, beliefs, ps, st)
    own = _own_channels(e)
    clamped = _clamped_parameters(e, beliefs)
    clamped === nothing || return local_free_energy(e.factor, _restrict(beliefs, own), merge(ps, clamped), st)
    e.shadowed && throw(ArgumentError(
        "a channel of $(typeof(e.factor)) is named like one of its parameter blocks, so exposed blocks \
         must be observed as points for the free energy; rename the channel"))
    return parameter_free_energy(e.factor, _param_view(e, beliefs), ps, st)
end
