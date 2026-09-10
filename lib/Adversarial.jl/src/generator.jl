# ---------------------------------------------------------------------------
# The implicit generative model of Mohamed & Lakshminarayanan (arXiv:1610.03483):
#
#     z ~ q(z),   x = G_θ(z)
#
# You can SAMPLE x. You cannot EVALUATE p_θ(x). That is the paper's definition of implicit,
# and it is a different sense from the one `README.md` means — see `Three Senses of Implicit.md`.
#
# For this project the operational content is one line:
#
#     this is the first factor that produces a `SampleBelief`.
#
# `LenticulumCore` has declared `SampleBelief` since the beginning and nothing has ever
# constructed one. A generator is what it was for.
#
# See `generator.md` and `Implicit Generative Models.md`.
# ---------------------------------------------------------------------------

"""
    NoiseSource(channel, dim; nsamples = 64, rng = Random.default_rng(), dist = randn)

The latent prior ``q(z)`` as an emitting factor: draws `nsamples` particles and hands them on
as a [`LenticulumCore.SampleBelief`](@ref).

This is the ``p(z)`` node of every GAN diagram, and it is a factor rather than a property of
the variable because of [[Everything is a Factor]] — a variable is a wire and has no content
of its own. Non-learnable, arity one, emitting only, exactly like `Mycelium.DataFactor`; the
difference is that a `DataFactor` clamps a point and this scatters a cloud.

`dist(rng, dim, n)` may be replaced to sample from something other than a standard normal.
"""
struct NoiseSource{R<:AbstractRNG,D} <: LenticulumCore.AbstractLenticulumFactor
    channel::Symbol
    dim::Int
    nsamples::Int
    rng::R
    dist::D
end

_std_normal(rng, d, n) = [randn(rng, d) for _ in 1:n]

function NoiseSource(channel::Symbol, dim::Int; nsamples::Int = 64,
                     rng::AbstractRNG = Random.default_rng(), dist = _std_normal)
    dim > 0 || throw(ArgumentError("dim must be positive; got $dim"))
    nsamples > 0 || throw(ArgumentError("nsamples must be positive; got $nsamples"))
    return NoiseSource(channel, dim, nsamples, rng, dist)
end

LenticulumCore.channels(f::NoiseSource) = (LenticulumCore.Channel(f.channel, f.dim),)
LenticulumCore.supported_polarities(f::NoiseSource) =
    (LenticulumCore.Polarity(NamedTuple{(f.channel,)}((LenticulumCore.Unobserved(),))),)
LenticulumCore.supports_polarity(f::NoiseSource, p::LenticulumCore.Polarity) =
    keys(p) == (f.channel,) && p[f.channel] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(::NoiseSource) = false
LuxCore.initialparameters(::AbstractRNG, ::NoiseSource) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::NoiseSource) = NamedTuple()

Mycelium.factor_message(f::NoiseSource, ::Symbol, _, _, _, ps, st) =
    (LenticulumCore.SampleBelief(f.dist(f.rng, f.dim, f.nsamples)), st)

# A prior over a latent nobody observes charges nothing: the entropy of q(z) is a constant of
# the model, not a function of anything being inferred.
Mycelium.local_free_energy(::NoiseSource, _, ps, st) = (0.0, st)

# =========================================================================== #

"""
    GeneratorFactor(net, dims::NamedTuple; nsamples = 64, rng = Random.default_rng(),
                    channels = (:z, :x), input = identity)

``x = G_\\theta(z)`` — an **implicit generative model**: samples come out, densities do not.

`net` is any `LuxCore.AbstractLuxLayer` mapping a latent to a sample. `dims` names the two
channels and their dimensions. Parameters and state are the net's, untouched.

```julia
g = GeneratorFactor(my_decoder, (z = 8, x = 2); nsamples = 512)
```

!!! warning "Unidirectional, and honestly so"
    `supported_polarities` returns **one** element. Inverting ``G_\\theta`` — recovering the
    latent that produced a given sample — is the "GAN inversion" problem and is generally
    intractable; there is no residual to solve, because a generator is a *function* with no
    relation behind it. Contrast `ImplicitLayers.DEQFactor`, which has two.

    That is not a limitation of this wrapper. It is what "implicit" means in Mohamed &
    Lakshminarayanan's sense as opposed to `README.md`'s sense — see
    `Three Senses of Implicit.md`.

!!! note "It emits particles, not a point"
    The message is a [`LenticulumCore.SampleBelief`](@ref) — the first one anything in this project has ever
    produced. And `Mycelium.combine` throws on two of those, which is `messages.md` §1's
    recorded main gap. [`RatioFactor`](@ref) is the route around it.
"""
struct GeneratorFactor{names,D<:Tuple,N,R<:AbstractRNG,F} <:
       LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names,D}
    net::N
    nsamples::Int
    rng::R
    input::F
end

function GeneratorFactor(
    net, dims::NamedTuple;
    nsamples::Int = 64,
    rng::AbstractRNG = Random.default_rng(),
    channels::Tuple{Symbol,Symbol} = (:z, :x),
    input = identity,
)
    length(dims) == 2 || throw(ArgumentError(
        "a GeneratorFactor has exactly two channels (latent and sample); got $(keys(dims))"))
    keys(dims) == channels || throw(ArgumentError(
        "dims keys $(keys(dims)) must match channels $channels, in order"))
    all(d -> d isa Int && d > 0, values(dims)) ||
        throw(ArgumentError("channel dimensions must be positive Ints; got $dims"))
    nsamples > 0 || throw(ArgumentError("nsamples must be positive; got $nsamples"))
    return GeneratorFactor(dims, net, nsamples, rng, input)
end

latentchannel(f::GeneratorFactor{names}) where {names} = names[1]
samplechannel(f::GeneratorFactor{names}) where {names} = names[2]
latentdim(f::GeneratorFactor) = f.dims[latentchannel(f)]
sampledim(f::GeneratorFactor) = f.dims[samplechannel(f)]

LuxCore.initialparameters(rng::AbstractRNG, f::GeneratorFactor) =
    LuxCore.initialparameters(rng, f.net)
LuxCore.initialstates(rng::AbstractRNG, f::GeneratorFactor) =
    LuxCore.initialstates(rng, f.net)
LuxCore.parameterlength(f::GeneratorFactor) = LuxCore.parameterlength(f.net)
LuxCore.statelength(f::GeneratorFactor) = LuxCore.statelength(f.net)

LenticulumCore.channels(f::GeneratorFactor{names}) where {names} =
    map((n, d) -> LenticulumCore.Channel(n, d), names, values(f.dims))
LenticulumCore.islearnable(::GeneratorFactor) = true

LenticulumCore.supported_polarities(f::GeneratorFactor{names}) where {names} = (
    LenticulumCore.Polarity(NamedTuple{names}(
        (LenticulumCore.Observed(), LenticulumCore.Unobserved()))),
)
function LenticulumCore.supports_polarity(
    f::GeneratorFactor{names}, p::LenticulumCore.Polarity
) where {names}
    keys(p) == names || return false
    return p[latentchannel(f)] isa LenticulumCore.Observed &&
           p[samplechannel(f)] isa LenticulumCore.Unobserved
end

# A generator has no residual: it is a function, and every (z, G(z)) pair satisfies it
# exactly. The energy that matters lives on the RatioFactor, which is the whole point of
# `Implicit Generative Models.md` — learning by comparison rather than by likelihood.
LenticulumCore.energyspace(::GeneratorFactor) = LenticulumCore.ScalarEnergySpace()
LenticulumCore.energy(::GeneratorFactor, x, a, y, ps, st) = (0.0, st)

"""
    generate(f::GeneratorFactor, z, ps, st) -> (x, st)

One forward pass: ``G_\\theta(z)`` for a single latent.
"""
generate(f::GeneratorFactor, z, ps, st) = LuxCore.apply(f.net, f.input(z), ps, st)

"""
    LenticulumCore.pushforward(f::GeneratorFactor, belief, ps, st) -> (belief, st)
    LenticulumCore.pushforward(m::GeneratorModel, belief, ps, st) -> (belief, st)

Push a belief on the latent channel through ``G_\\theta`` — AutoBayes' ``c_*\\pi``.

> [!note] This is the first implementation of `pushforward` in the project
> `LenticulumCore` has declared `forward`, `logdensity` and `pushforward` since the beginning
> and nothing has implemented the last one. A generator is the case where it is *easy*: the
> map is deterministic, so pushing a particle set forward is mapping over it, with no
> marginalisation integral at all. `open_model.md` warns that `pushforward` is "one of the two
> expensive operations"; for a sampler it is the cheap one, which is the whole appeal of
> implicit generative models.

| in | out | why |
|---|---|---|
| `DiracBelief` | `DiracBelief` | a function of a point is a point |
| `SampleBelief` | `SampleBelief` | map each particle; **weights are carried through unchanged** |
| `TrivialBelief` | `TrivialBelief` | no latent, no samples |

The middle row is the interesting one and the weight-preservation is not a detail: pushing a
*weighted* particle set through a deterministic map leaves the weights alone, because the map
is a bijection on particle indices. That is why importance weights survive a generator and why
[`RatioFactor`](@ref) can be applied downstream of one.
"""
function LenticulumCore.pushforward(f::GeneratorFactor, b::LenticulumCore.SampleBelief, ps, st)
    out = similar(b.samples, Any)
    for i in eachindex(b.samples)
        y, st = generate(f, b.samples[i], ps, st)
        out[i] = y
    end
    return (LenticulumCore.SampleBelief([out[i] for i in eachindex(out)], b.weights), st)
end
function LenticulumCore.pushforward(f::GeneratorFactor, b::LenticulumCore.DiracBelief, ps, st)
    y, st = generate(f, _vec(b.value), ps, st)
    return (LenticulumCore.DiracBelief(y), st)
end
LenticulumCore.pushforward(::GeneratorFactor, ::LenticulumCore.TrivialBelief, ps, st) =
    (LenticulumCore.TrivialBelief(), st)
LenticulumCore.pushforward(::GeneratorFactor, ::Nothing, ps, st) =
    (LenticulumCore.TrivialBelief(), st)

"""
    GeneratorModel(factor)

The open model: a deterministic map with the latent as its input. Pure — a generator hides
nothing *given* `z`; what it hides is the density of its own output, and that is not what
``\\llbracket c\\rrbracket`` means.
"""
struct GeneratorModel{F<:GeneratorFactor} <: LenticulumCore.AbstractOpenModel
    factor::F
end
LenticulumCore.ispure(::GeneratorModel) = true
LenticulumCore.latentspace(::GeneratorModel) = nothing

# The canonical interface signature, `pushforward(model, π, ps, st)`.
LenticulumCore.pushforward(m::GeneratorModel, π, ps, st) =
    LenticulumCore.pushforward(m.factor, π, ps, st)
# Deterministic map, exact particle transport — no approximation anywhere.
LenticulumCore.isexact(::GeneratorModel) = true

# `TrivialInversion` for the same reason `LuxFactor` gets it: there is nothing to invert.
LenticulumCore.assemble(f::GeneratorFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(GeneratorModel(f), LenticulumCore.TrivialInversion()), st)

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:GeneratorModel,LenticulumCore.TrivialInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    return LenticulumCore.pushforward(f, _get(inputs, latentchannel(f)), ps, st)
end

"""
    Mycelium.factor_message(f::GeneratorFactor, target, polarity, inputs, prior, ps, st)

The pushforward of the latent belief. Asking for a message on the **latent** channel throws:
that is GAN inversion, and this factor cannot do it.
"""
function Mycelium.factor_message(
    f::GeneratorFactor{names}, target::Symbol, polarity, inputs, prior, ps, st
) where {names}
    if target === latentchannel(f)
        throw(ArgumentError(
            "a GeneratorFactor cannot emit on its latent channel :$(latentchannel(f)) — \
             inverting G_θ is the GAN-inversion problem and there is no residual to solve. \
             See `Three Senses of Implicit.md`."))
    end
    target === samplechannel(f) || throw(ArgumentError(
        "channel :$target is not a channel of this GeneratorFactor (has $names)"))
    return LenticulumCore.pushforward(f, _get(inputs, latentchannel(f)), ps, st)
end

Mycelium.local_free_energy(::GeneratorFactor, _, ps, st) = (0.0, st)
