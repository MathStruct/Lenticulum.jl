# ---------------------------------------------------------------------------
# Channels and polarity.
#
# AutoBayes gives every open model three spaces: unobserved X, observed Y, latent ⟦c⟧.
# `ImplicitREDDiff.md` gives every factor three selection matrices with
#     P_in + P_out + P_latent = Id.
# These are the same trichotomy with crossed names:
#     X  <-> P_out    (inferred)
#     Y  <-> P_in     (clamped)
#     ⟦c⟧ <-> P_latent
# See `channels.md` and `Channels and Polarity.md`.
# ---------------------------------------------------------------------------

"""
    abstract type AbstractChannelPolarity

The role a channel plays in one particular use of a factor.
"""
abstract type AbstractChannelPolarity end

"""
    Observed()

AutoBayes' observed space ``Y``; the note's ``P_{in}``. Clamped to data. The posterior does
*not* range over this channel.
"""
struct Observed <: AbstractChannelPolarity end

"""
    Unobserved()

AutoBayes' unobserved space ``X``; the note's ``P_{out}``. This is what inference solves
for, and what the inversion produces a belief over.

!!! warning "The names cross"
    "Unobserved" is the *output* of inference. A factor asked to predict `y` from `x`
    marks `y` as `Unobserved()` and `x` as `Observed()` — not the other way round.
"""
struct Unobserved <: AbstractChannelPolarity end

"""
    Latent()

AutoBayes' latent space ``\\llbracket c \\rrbracket``; the note's ``P_{latent}``. Internal
scratch that composition hid. Either revealed (a free retyping) or marginalised (expensive).
"""
struct Latent <: AbstractChannelPolarity end

"""
    Channel{name,S}(space)

A named port of a factor, carrying the space its values live in.

!!! warning "Name shadowing"
    This type shadows `Base.Channel` inside `LenticulumCore` and is deliberately **not
    exported**. Refer to it as `LenticulumCore.Channel`, or import it explicitly.
"""
struct Channel{name,S}
    space::S
end
Channel(name::Symbol, space) = Channel{name,typeof(space)}(space)

channelname(::Channel{name}) where {name} = name
channelspace(c::Channel) = c.space

Base.show(io::IO, c::Channel{name}) where {name} = print(io, "Channel(:", name, ", ", c.space, ")")

"""
    default_precision(p::AbstractChannelPolarity)

The default weight ``\\rho`` for a channel of this polarity, as in
``P = \\rho_{in}P_{in} + \\rho_{out}P_{out} + \\rho_{latent}P_{latent}``.

`Observed` defaults to `Inf` (a hard clamp — a categorical *cup*); the others to `1`.
A finite `Observed` precision is a *soft* clamp, which is what makes noisy observations and
annealed conditioning expressible.
"""
default_precision(::Observed) = Inf
default_precision(::Unobserved) = 1.0
default_precision(::Latent) = 1.0

"""
    Polarity(assignment::NamedTuple, [precisions::NamedTuple])
    Polarity(; channel = polarity, ...)

An assignment of an [`AbstractChannelPolarity`](@ref) to each channel of a factor, together
with a precision ``\\rho`` per channel.

Because the assignment is a `NamedTuple`, the channel names live in the *type*, so an
assembled lens specialises at compile time rather than dispatching through a `Dict`.

```julia
Polarity(; x = Observed(), y = Unobserved())
```
"""
struct Polarity{names,T<:Tuple,R<:Tuple}
    assignment::NamedTuple{names,T}
    precisions::NamedTuple{names,R}
end

function Polarity(assignment::NamedTuple{names}) where {names}
    return Polarity(assignment, NamedTuple{names}(map(default_precision, values(assignment))))
end
Polarity(; kwargs...) = Polarity(NamedTuple(kwargs))

Base.getindex(p::Polarity, name::Symbol) = getfield(p.assignment, name)
Base.keys(p::Polarity{names}) where {names} = names
Base.:(==)(a::Polarity, b::Polarity) = a.assignment == b.assignment && a.precisions == b.precisions

"""
    channel_precision(p::Polarity, name::Symbol)

The weight ``\\rho`` attached to channel `name`.
"""
channel_precision(p::Polarity, name::Symbol) = getfield(p.precisions, name)

"""
    select(p::Polarity, ::Type{P}) where {P<:AbstractChannelPolarity}

The tuple of channel names carrying polarity `P`. This is the diagonal selection matrix
``P_{in}`` / ``P_{out}`` / ``P_{latent}`` as a tuple of `Symbol`s.
"""
function select(p::Polarity{names}, ::Type{P}) where {names,P<:AbstractChannelPolarity}
    return filter(n -> getfield(p.assignment, n) isa P, names)
end

observed_channels(p::Polarity) = select(p, Observed)
unobserved_channels(p::Polarity) = select(p, Unobserved)
latent_channels(p::Polarity) = select(p, Latent)

"""
    ispartition(p::Polarity)

Check ``P_{in} + P_{out} + P_{latent} = \\mathrm{Id}`` with pairwise-zero products: every
channel gets exactly one polarity. Guaranteed by the `NamedTuple` representation, so this
is a total function returning `true`; kept as a named predicate because the corresponding
check on a *graph-wide* polarity is not trivial.
"""
ispartition(::Polarity) = true

# --- Factor interface ------------------------------------------------------

"""
    channels(factor) -> Tuple{Vararg{Channel}}

The named ports of `factor`. Must be implemented by every factor.
"""
function channels end

"""
    supports_polarity(factor, p::Polarity) -> Bool

Whether `factor` can answer the input/output split `p`.

This predicate replaces Lux's acyclicity restriction. A Lux `Chain` is well-formed iff the
wiring is a DAG; a Lenticulum graph is well-formed iff every scheduled message uses a
polarity its factor supports. The default is conservative: a factor supports nothing until
it says otherwise.
"""
supports_polarity(::AbstractLenticulumFactor, ::Polarity) = false


"""
    supported_polarities(factor) -> Tuple{Vararg{Polarity}}

The polarities `factor` can answer, enumerated.

[`supports_polarity`](@ref) is the predicate form; this is the *listable* form, which a
scheduler needs in order to plan messages without guessing. The default is empty, matching
`supports_polarity`'s conservative `false`.

A factor with exactly one supported polarity is **unidirectional**: it can be evaluated in
one direction only, like an ordinary Lux layer. See [`isunidirectional`](@ref).
"""
supported_polarities(::AbstractLenticulumFactor) = ()

"""
    isunidirectional(factor) -> Bool

Whether `factor` admits exactly one polarity, i.e. has a fixed input/output split and is
therefore an explicit rather than an implicit factor.

Non-learnable structural nodes are usually unidirectional: a data source only emits, a loss
only absorbs.
"""
isunidirectional(f::AbstractLenticulumFactor) = length(supported_polarities(f)) == 1

"""
    assemble(factor, p::Polarity, ps, st) -> (lens, st)

Produce the [`AbstractBayesianLens`](@ref) realising `factor` under polarity `p`.

This is the operation that has no counterpart in Lux, and the reason a factor cannot be a
Lux layer: a Lux layer *is* a lens, whereas a factor only *becomes* one once a direction is
chosen. See `Lux as a Parametric Lens.md`.
"""
function assemble end
