# ---------------------------------------------------------------------------
# Passing points instead of beliefs: a Rao-Blackwellised Gibbs sampler, and a buffer that keeps
# the samples, weighs them by a filter, forgets old ones, and averages energies and gradients.
#
# The sampled variables (typically the discrete switches) are drawn one at a time from their
# exact conditional; the other variables are never sampled but integrated out by belief
# propagation with the sampled ones pinned (`pin!`). With every sampled variable pinned and the
# rest a tree with closed-form messages, the pinned graph's Bethe free energy is
# −log p(data, pins), so averaging its gradient over posterior samples is Monte Carlo EM.
#
# The averages obey the requirement of `Extending Mycelium.md` §6b: energies are scalarised per
# sample and averaged per graded component, and every factor's residual keeps its mean and full
# covariance (law of total covariance across samples).
#
# See `sampling.md`.
# ---------------------------------------------------------------------------

"""
    sample_belief(rng, belief)

One draw from a belief: the value of a `DiracBelief`, a state (its label, or its index) of a
`CategoricalBelief`, a vector from a proper `GaussianBelief`, and from a `MixtureBelief` a component
by its weight, then a draw from it.
"""
sample_belief(rng, b::LenticulumCore.DiracBelief) = b.value
function sample_belief(rng, b::LenticulumCore.CategoricalBelief)
    p = LenticulumCore.probabilities(b)
    k = min(searchsortedfirst(cumsum(p), Random.rand(rng) * sum(p)), length(p))
    return b.labels === nothing ? k : b.labels[k]
end
function sample_belief(rng, b::GaussianBelief)
    isproper(b) || throw(ArgumentError("cannot sample an improper Gaussian belief"))
    μ, Σ = LenticulumCore.belief_mean(b), LenticulumCore.belief_cov(b)
    return μ + LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ)).L * Random.randn(rng, length(μ))
end
function sample_belief(rng, m::LenticulumCore.MixtureBelief)
    w = LenticulumCore.mixture_weights(m)
    k = min(searchsortedfirst(cumsum(w), Random.rand(rng) * sum(w)), length(w))
    return sample_belief(rng, m.components[k])
end
sample_belief(rng, b) = throw(ArgumentError("no sampler for a $(typeof(b)); a sampled variable needs a \
    Dirac, categorical, proper Gaussian or mixture marginal"))

# --- the sampler -------------------------------------------------------------------------------

"""
    GibbsSampler(g, sampled; schedule = tree_schedule(g))

A Gibbs sampler over the variables named in `sampled`. Each update draws one of them from its
conditional given the others: the others are pinned, belief propagation runs `schedule` over the
graph, and the variable's marginal is that conditional. Variables not in `sampled` are integrated
out (Rao-Blackwellisation), which is exact when the graph without the pinned variables is a tree
with closed-form messages, e.g. a conditional-Gaussian graph with its discrete variables sampled.

Blocks of variables are sampled jointly by making them one variable: in a graph of switching
factors, one factor with ``2^B`` outcomes for a block of ``B`` switches. Use [`gibbs_init`](@ref), then
[`gibbs_sweep!`](@ref); keep the samples in a [`SampleBuffer`](@ref).
"""
struct GibbsSampler{G<:FactorGraph,S<:AbstractSchedule}
    graph::G
    sampled::Vector{Int}
    schedule::S
end
function GibbsSampler(g::FactorGraph, sampled; schedule::AbstractSchedule = tree_schedule(g))
    ids = Int[g.variable_index[v] for v in sampled]
    allunique(ids) || throw(ArgumentError("a variable is listed twice in `sampled`"))
    return GibbsSampler(g, ids, schedule)
end

"""
    GibbsState

The sampler's state: a `MessageStore` whose pins are the current points of the sampled variables,
and the number of sweeps so far. [`gibbs_points`](@ref) reads the points by name.
"""
mutable struct GibbsState
    store::MessageStore
    sweeps::Int
end

"The current points of the sampled variables, as a `NamedTuple` keyed by variable name."
gibbs_points(state::GibbsState, sampler::GibbsSampler) =
    NamedTuple{Tuple(sampler.graph.variables[v].name for v in sampler.sampled)}(
        Tuple(state.store.pins[v] for v in sampler.sampled))

# The conditional of one sampled variable given the pins of the others.
function _conditional!(s::MessageStore, sampler::GibbsSampler, vid::Int, ps, st)
    delete!(s.pins, vid)
    reset!(s)
    _, st = propagate!(s, sampler.graph, sampler.schedule, ps, st)
    return marginal(s, sampler.graph, vid), st
end

"""
    gibbs_init(sampler, ps, st; rng) -> (GibbsState, st)

A first point for every sampled variable, drawn in order, each from its marginal with the earlier
ones pinned and the later ones still free. For the later ones that marginal is the approximate
one belief propagation gives on a graph with mixtures; it only starts the chain.
"""
function gibbs_init(sampler::GibbsSampler, ps, st; rng::AbstractRNG = Random.default_rng())
    s = MessageStore(sampler.graph)
    for vid in sampler.sampled
        b, st = _conditional!(s, sampler, vid, ps, st)
        s.pins[vid] = sample_belief(rng, b)
    end
    return GibbsState(s, 0), st
end

"""
    gibbs_sweep!(state, sampler, ps, st; rng) -> st

One sweep: every sampled variable in turn, drawn from its exact conditional given the current
points of the others.
"""
function gibbs_sweep!(state::GibbsState, sampler::GibbsSampler, ps, st; rng::AbstractRNG = Random.default_rng())
    for vid in sampler.sampled
        b, st = _conditional!(state.store, sampler, vid, ps, st)
        state.store.pins[vid] = sample_belief(rng, b)
    end
    state.sweeps += 1
    return st
end

# --- residual moments: the hook factors implement ------------------------------------------------

"""
    factor_residual_moments(factor, msgs, ps) -> (mean, cov) or nothing

The mean and covariance of the factor's residual vector under its own belief (the factor's
potential times the incoming messages `msgs`). The default is `nothing` (no residual vector).
Lenticulum's `LinearConstraintFactor` and `SwitchingConstraintFactor` implement it. A
[`SampleBuffer`](@ref) combines the per-sample moments across samples.
"""
factor_residual_moments(factor, msgs, ps) = nothing

# --- the buffer ----------------------------------------------------------------------------------

"""
    AbstractSampleFilter

How a [`SampleBuffer`](@ref) weighs its samples: [`WindowFilter`](@ref) or [`DecayFilter`](@ref).
"""
abstract type AbstractSampleFilter end

"""
    WindowFilter(n)

The last `n` samples, with equal weights; older samples are dropped.
"""
struct WindowFilter <: AbstractSampleFilter
    n::Int
    WindowFilter(n::Integer) = n > 0 ? new(n) : throw(ArgumentError("a window needs at least one sample"))
end

"""
    DecayFilter(λ; cutoff = 1e-8)

Weight ``\\lambda^a`` for a sample ``a`` draws old, normalised; samples whose weight falls below
`cutoff` are dropped. With ``\\lambda`` moved towards 1 over the course of learning this is
stochastic-approximation EM (Delyon, Lavielle & Moulines 1999).
"""
struct DecayFilter <: AbstractSampleFilter
    λ::Float64
    cutoff::Float64
    function DecayFilter(λ::Real; cutoff::Real = 1e-8)
        0 < λ <= 1 || throw(ArgumentError("the decay must be in (0, 1]; got $λ"))
        return new(float(λ), float(cutoff))
    end
end

_age_weight(f::WindowFilter, age) = age < f.n ? 1.0 : 0.0
_age_weight(f::DecayFilter, age) = f.λ^age
_keep(f::WindowFilter, age) = age < f.n
_keep(f::DecayFilter, age) = f.λ^age >= f.cutoff

"""
    SampleBuffer(filter; horizon = typemax(Int))

Keeps samples: each one's pinned points, its graded local free energies, its total (the pinned
graph's scalar free energy), the residual moments of every factor that has them, an optional
weight, and the parameter step at which it was drawn. The `filter` weighs the samples by age;
samples drawn more than `horizon` parameter steps ago are forgotten ([`advance!`](@ref)).

Queries: [`sample_weights`](@ref), [`energy_moments`](@ref), [`residual_moments`](@ref),
[`free_energy_mean`](@ref), [`sample_gradient`](@ref).
"""
mutable struct SampleBuffer{F<:AbstractSampleFilter}
    filter::F
    horizon::Int
    entries::Vector{Any}
    step::Int
end
SampleBuffer(filter::AbstractSampleFilter; horizon::Integer = typemax(Int)) =
    horizon >= 0 ? SampleBuffer(filter, Int(horizon), Any[], 0) :
    throw(ArgumentError("the horizon must be nonnegative"))

Base.length(b::SampleBuffer) = length(b.entries)

function _prune!(b::SampleBuffer)
    n = length(b.entries)
    keep = [b.step - e.step <= b.horizon && _keep(b.filter, n - i) for (i, e) in enumerate(b.entries)]
    b.entries = b.entries[keep]
    return b
end

"""
    advance!(buffer) -> buffer

Count one parameter step: samples drawn more than `horizon` steps ago are forgotten. Call it after
every parameter update.
"""
advance!(b::SampleBuffer) = (b.step += 1; _prune!(b))

# A plain record of a (graded) energy: nested NamedTuples of numbers, so it can be averaged.
_energy_record(e::LenticulumCore.GradedEnergy) = map(_energy_record, e.parts)
_energy_record(e::Real) = float(e)
_energy_record(e::AbstractArray{<:Real}) = float.(collect(e))
_energy_record(e) = throw(ArgumentError("cannot record an energy of type $(typeof(e))"))

function _record!(b::SampleBuffer, s::MessageStore, g::FactorGraph, ps, st, weight)
    weight >= 0 || throw(ArgumentError("sample weights must be nonnegative"))
    _check_energy_edges(g; mean_field = false, what = "recording residual moments from samples")
    F, st = factor_free_energies(s, g, ps, st)
    V = variable_corrections(s, g)
    total, st = LenticulumCore.scalar_free_energy(s, g, ps, st)
    names = Tuple(f.name for f in g.factors)
    moments = NamedTuple{names}(Tuple(
        factor_residual_moments(f.factor, messages_into(s, g, f.id), _factor_parameters(f, _subtree(ps, f.name)))
        for f in g.factors))
    push!(b.entries, (pins = copy(s.pins), step = b.step, weight = float(weight), total = float(total),
                      energies = (factors = _energy_record(F), variables = _energy_record(V)), moments))
    _prune!(b)
    return st
end

"""
    record!(buffer, state, sampler, ps, st; weight = 1) -> st
    record!(buffer, sampler, points::NamedTuple, ps, st; weight = 1) -> st

Add a sample: the sampler's current points, or the given points of the sampled variables. All
sampled variables are pinned, belief propagation runs once, and the graph's graded local free
energies, its total and the factors' residual moments are stored with the current parameter
step. The second form, with `weight`, lets a buffer hold a weighted set of configurations, e.g.
every configuration at its exact posterior weight.
"""
function record!(b::SampleBuffer, state::GibbsState, sampler::GibbsSampler, ps, st; weight::Real = 1.0)
    reset!(state.store)
    _, st = propagate!(state.store, sampler.graph, sampler.schedule, ps, st)
    return _record!(b, state.store, sampler.graph, ps, st, weight)
end
function record!(b::SampleBuffer, sampler::GibbsSampler, points::NamedTuple, ps, st; weight::Real = 1.0)
    g = sampler.graph
    s = MessageStore(g)
    all(k -> haskey(g.variable_index, k), keys(points)) &&
        Set(g.variable_index[k] for k in keys(points)) == Set(sampler.sampled) ||
        throw(ArgumentError("give a point for every sampled variable, and only for those"))
    for (k, v) in pairs(points)
        pin!(s, g, k, v)
    end
    _, st = propagate!(s, g, sampler.schedule, ps, st)
    return _record!(b, s, g, ps, st, weight)
end

"""
    sample_weights(buffer) -> Vector

The normalised weights of the stored samples: the filter's weight by age times each sample's own
weight.
"""
function sample_weights(b::SampleBuffer)
    n = length(b.entries)
    n == 0 && throw(ArgumentError("the sample buffer is empty"))
    w = [_age_weight(b.filter, n - i) * e.weight for (i, e) in enumerate(b.entries)]
    sum(w) > 0 || throw(ArgumentError("every stored sample has weight zero"))
    return w ./ sum(w)
end

_wmean(xs, w) = sum(wi .* x for (x, wi) in zip(xs, w))
_wmap(f, xs::AbstractVector{<:NamedTuple}) = NamedTuple{keys(first(xs))}(Tuple(f([getfield(x, k) for x in xs]) for k in keys(first(xs))))

# weighted mean and variance of every leaf of a nested record
_leaf_moments(xs::AbstractVector{<:NamedTuple}, w) = _wmap(ys -> _leaf_moments(ys, w), xs)
function _leaf_moments(xs::AbstractVector, w)
    m = _wmean(xs, w)
    return (mean = m, var = _wmean([(x .- m) .^ 2 for x in xs], w))
end

"""
    energy_moments(buffer) -> NamedTuple

The weighted mean and variance, across the stored samples, of every graded component of every
factor's local free energy and of every variable's correction, keyed as
`(factors = (name = (fit = (mean, var), …), …), variables = …)`. Each sample's energies are
already scalarised per component, so nothing is averaged before its scalarisation.
"""
energy_moments(b::SampleBuffer) = _leaf_moments([e.energies for e in b.entries], sample_weights(b))

"""
    free_energy_mean(buffer) -> Real

The weighted mean of the samples' totals. With every sampled variable pinned and the rest a tree,
each total is ``-\\log p(\\text{data}, \\text{pins})``.
"""
free_energy_mean(b::SampleBuffer) = _wmean([e.total for e in b.entries], sample_weights(b))

"""
    residual_moments(buffer, factor::Symbol) -> (mean, cov)

The mean and the **full** covariance of the factor's residual vector across the stored samples, by
the law of total covariance: the weighted mean of the within-sample covariances plus the weighted
covariance of the within-sample means. For a squared-norm scalarisation these two are exact
sufficient statistics of the expected energy, ``\\mathbb E\\lVert r\\rVert_W^2 = \\lVert \\bar r\\rVert_W^2 + \\operatorname{tr}(W \\operatorname{Cov} r)``.
Throws if the factor reports no residual moments.
"""
function residual_moments(b::SampleBuffer, factor::Symbol)
    w = sample_weights(b)
    ms = [getfield(e.moments, factor) for e in b.entries]
    any(isnothing, ms) && throw(ArgumentError("factor :$factor reports no residual moments (`factor_residual_moments`)"))
    μ = _wmean([first(m) for m in ms], w)
    C = _wmean([last(m) + (first(m) - μ) * (first(m) - μ)' for m in ms], w)
    return μ, C
end

"""
    sample_gradient(buffer, sampler, ps, st; ad = nothing, ε = 1e-6, through_inference = false)
        -> gradient

The weighted average, over the stored samples, of the gradient of each sample's pinned total
``-\\log p(\\text{data}, \\text{pins})`` at the **current** parameters `ps`. With samples from the
posterior and an exact pinned graph (a tree), this is the Monte Carlo EM gradient of
``-\\log p(\\text{data})``. Samples drawn under older parameters bias it, which is what the buffer's
`horizon` limits.

Each sample's points are pinned and belief propagation runs the sampler's schedule at `ps`. Then:

- by default, [`free_energy_gradient`](@ref LenticulumCore.free_energy_gradient) is taken with the
  factor beliefs held fixed (LP-020): one inference per sample, exact when that inference has
  converged, as on a tree. A factor without `factor_belief` gets the fixed-message derivative.
- with `through_inference = true`, the scalar free energy is differentiated *through* the
  inference instead: central differences of step `ε` for `ad = nothing`, or the given AD backend.
  That costs an inference per sample and, with differences, per parameter coordinate, and needs no
  factor hook. It is the reference the default is checked against.

Factors that are not learnable get zero gradients. The result has `ps`'s structure.
"""
function sample_gradient(b::SampleBuffer, sampler::GibbsSampler, ps, st; ad = nothing, ε::Real = 1e-6,
                         through_inference::Bool = false)
    g = sampler.graph
    w = sample_weights(b)
    acc = nothing
    like = nothing
    for (e, wi) in zip(b.entries, w)
        wi == 0 && continue
        grad = through_inference ?
            LenticulumCore._parameter_gradient(ad, _pinned_total, ps, sampler, e.pins, st; ε) :
            _pinned_gradient(ps, sampler, e.pins, st; ad, ε)
        θ = LenticulumCore._flatten(_mask_unlearnable(grad, g))
        acc = acc === nothing ? wi .* θ : acc .+ wi .* θ
        like = grad
    end
    return LenticulumCore._unflatten(acc, like)
end

# Inference with the pins at `ps`, then the graph gradient at the fixed factor beliefs.
function _pinned_gradient(ps, sampler::GibbsSampler, pins, st; ad, ε)
    g = sampler.graph
    s = MessageStore(g)
    merge!(s.pins, pins)
    _, st = propagate!(s, g, sampler.schedule, ps, st)
    return LenticulumCore.free_energy_gradient(s, g, ps, st; ad, ε, schedule = sampler.schedule)
end

# −log p(data, pins): inference with the pins at `ps`, then the scalar free energy.
function _pinned_total(ps, sampler::GibbsSampler, pins, st)
    g = sampler.graph
    s = MessageStore(g)
    merge!(s.pins, pins)
    _, st = propagate!(s, g, sampler.schedule, ps, st)
    return first(LenticulumCore.scalar_free_energy(s, g, ps, st))
end

function _mask_unlearnable(grad, g::FactorGraph)
    names = Tuple(f.name for f in g.factors if haskey(grad, f.name))
    sub = map(names) do n
        f = factornode(g, n)
        p = getfield(grad, n)
        θ = LenticulumCore._flatten(p)
        LenticulumCore._unflatten(_mask_gradient(θ, _active_indices(f, p)), p)
    end
    return merge(grad, NamedTuple{names}(sub))
end
