# Incoming factor games and solved-point lens games deliberately have different domains.
function _game_point(b, dim, who)
    b isa LenticulumCore.DiracBelief || throw(ArgumentError(
        "$who requires a Dirac belief: no distributional residual expectation or entropy is defined"))
    v = _point(b)
    length(v) == dim || throw(ArgumentError("$who needs a point of dimension $dim"))
    return v
end

@inline function _incoming_point_messages(f, π, y)
    cs = LenticulumCore.channels(f)
    _game_point(π, LenticulumCore.channelspace(cs[1]), "factor game prior")
    _game_point(y, LenticulumCore.channelspace(cs[2]), "factor game observation")
    names = map(LenticulumCore.channelname, cs)
    return NamedTuple{names}((π, y))
end

const _PointGameLens = Union{
    LenticulumCore.BayesianLens{<:DEQModel,<:LenticulumCore.SolverInversion},
    _NeuralODELens,
}

LenticulumCore.energyspace(::_PointGameLens) = LenticulumCore.ScalarEnergySpace()
LenticulumCore.scalarisation(::_PointGameLens) = LenticulumCore.IdentityScalarisation()

function _lens_game_inputs(lens, π, y)
    f, p = lens.model.factor, lens.model.polarity
    LenticulumCore.supports_polarity(f, p) || throw(ArgumentError(
        "point lens game requires a supported observed/unobserved polarity"))
    src = only(LenticulumCore.observed_channels(p))
    dst = only(LenticulumCore.unobserved_channels(p))
    LenticulumCore.channel_precision(p, src) == Inf || throw(ArgumentError(
        "point lens game requires a hard Dirac observation; soft observed-channel inversion is not implemented"))
    inputs = y isa NamedTuple ? y : NamedTuple{(src,)}((y,))
    keys(inputs) == (src,) || throw(ArgumentError(
        "point lens game observations must contain only :$src"))
    cs = LenticulumCore.channels(f)
    srcdim = only(LenticulumCore.channelspace(c) for c in cs if LenticulumCore.channelname(c) == src)
    dstdim = only(LenticulumCore.channelspace(c) for c in cs if LenticulumCore.channelname(c) == dst)
    _game_point(inputs[src], srcdim, "lens game observation")
    if π isa LenticulumCore.DiracBelief
        _game_point(π, dstdim, "lens game prior")
    elseif π isa LenticulumCore.GaussianBelief
        LenticulumCore.dimension(π) == dstdim || throw(ArgumentError("lens game prior dimension mismatch"))
        LenticulumCore.isproper(π) || throw(ArgumentError(
            "lens game Gaussian anchor needs positive definite precision; no singular-anchor law is defined"))
    elseif !(π isa LenticulumCore.TrivialBelief)
        throw(ArgumentError("lens game prior must be Dirac (hard), proper Gaussian (quadratic anchor), or trivial"))
    end
    return inputs, src, dst
end

_anchor_cost(::LenticulumCore.TrivialBelief, u) = zero(eltype(u))
_anchor_cost(π::LenticulumCore.DiracBelief, u) = u == _point(π) ? zero(eltype(u)) : Inf
function _anchor_cost(π::LenticulumCore.GaussianBelief, u)
    δ = u .- LenticulumCore.belief_mean(π)
    return dot(δ, π.Λ * δ) / 2
end

"""
    LenticulumCore.free_energy(lens, π, inputs, ps, st)

For an assembled DEQ/NeuralODE lens, evaluate the residual energy at `invert`'s point,
then add the prior's anchor cost on the unobserved channel. `inputs` is a Dirac observation
or a NamedTuple containing that observation on the polarity's observed channel.

The anchor is zero for a trivial prior; zero at an exactly equal Dirac point and infinite
elsewhere; or `½(u-μ)'Λ(u-μ)` for a proper Gaussian. The Gaussian cost is relative to its
mean (no log-normalization constant). Channel precisions do not soften a Dirac prior.
Soft observations and other beliefs throw. Entropy uses the zero point-belief convention.

This is Definition 20's expectation under the inversion, with this specified residual and
anchor energy. `assemble(f, p)` restricts the relation to one polarity; it does not turn the
root/flow inversion into a posterior conditioned on the anchor. See `point_game.md`.
"""
function LenticulumCore.free_energy(lens::_PointGameLens, π, y, ps, st)
    inputs, src, dst = _lens_game_inputs(lens, π, y)
    solved, st = LenticulumCore.invert(lens, π, inputs, ps, st)
    f = lens.model.factor
    u = _point(solved)
    u === nothing && throw(ArgumentError("point lens inversion did not return a point"))
    names = map(LenticulumCore.channelname, LenticulumCore.channels(f))
    msgs = NamedTuple{names}(map(n -> n == dst ? solved : inputs[src], names))
    fit, st = Mycelium.local_free_energy(f, msgs, ps, st)
    return (fit + _anchor_cost(π, u), st)
end

function LenticulumCore.entropy(lens::_PointGameLens, π, y, ps, st)
    _lens_game_inputs(lens, π, y)
    return (0.0, st)
end
