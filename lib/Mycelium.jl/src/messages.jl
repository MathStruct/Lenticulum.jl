# ---------------------------------------------------------------------------
# Messages and the message store.
#
# A message is an `AbstractBelief` travelling along an edge in one of the two directions.
# The store keeps BOTH directions per edge separately, because belief propagation needs the
# EXCLUSION PRINCIPLE: the message a factor receives from a variable must not contain what
# that same factor told the variable last round.
#
# Keeping both directions is the alternative to "divide out", which is what a
# density-representation would do and which is not available for general beliefs.
#
# See `messages.md`.
# ---------------------------------------------------------------------------

"""
    Message(belief, iteration)

A belief in transit, tagged with the sweep that produced it. The tag is what lets the
scheduler tell a stale message from a fresh one without comparing beliefs.
"""
struct Message{B}
    belief::B
    iteration::Int
end

"""
    MessageStore(g)

Per-edge storage for both message directions. `nothing` means "not yet computed", which is
distinct from "computed and uninformative" (the latter is `TrivialBelief()`).
"""
abstract type AbstractMessageStore end
mutable struct MessageStore <: AbstractMessageStore
    to_variable::Vector{Any}
    to_factor::Vector{Any}
    iteration::Int
end
MessageStore(g::FactorGraph) =
    MessageStore(Vector{Any}(nothing, nedges(g)), Vector{Any}(nothing, nedges(g)), 0)

has_to_variable(s::AbstractMessageStore, ei::Int) = s.to_variable[ei] !== nothing
has_to_factor(s::AbstractMessageStore, ei::Int) = s.to_factor[ei] !== nothing

"""
    reset!(store)

Clear all messages. Used between independent inference problems on the same graph.
"""
function reset!(s::MessageStore)
    fill!(s.to_variable, nothing)
    fill!(s.to_factor, nothing)
    s.iteration = 0
    return s
end

# --- Combining beliefs -----------------------------------------------------

"""
    combine(a::AbstractBelief, b::AbstractBelief) -> AbstractBelief

Pool two beliefs about the same variable — the product of densities, in sum-product terms.

This is **the hardest operation in the package** and is deliberately partial. What is
implemented is what can be done honestly for the belief types `LenticulumCore` currently
provides:

| case | result | why |
|---|---|---|
| `TrivialBelief` with anything | the other one | it is the unit |
| two agreeing `DiracBelief`s | that Dirac | idempotent |
| two disagreeing `DiracBelief`s | **throws** | two hard clamps in contradiction is a modelling error, not a numerical one |
| `DiracBelief` with anything | the Dirac | a hard clamp dominates (`ρ_in = Inf`, per `Channels and Polarity.md`) |
| two Gaussians | Gaussian | add canonical parameters |
| two categoricals on the same support | categorical | normalised pointwise product |
| `SampleBelief` with a density or potential | reweighted samples | importance weighting, no resampling |
| potential with potential | `LogPotentialBelief` | sum the log-potentials |
| Gaussian, categorical, or mixture with a potential | **throws** | arbitrary callables have no exact parametric product rule |
| mixture with other supported beliefs | mixture | component products weighted by overlaps |
| two `SampleBelief`s | **throws** | neither supplies a density |

Everything else throws an informative error. Potential products with categorical beliefs
are also refused: converting labels or state indices into points needs an explicit model.
See `messages.md` §"Implementation difficulties".
"""
function combine end

combine(::LenticulumCore.TrivialBelief, b) = b
combine(a, ::LenticulumCore.TrivialBelief) = a
combine(a::LenticulumCore.TrivialBelief, ::LenticulumCore.TrivialBelief) = a

# The two methods above take an untyped argument, so on a pair of `AbstractBelief`s they are
# no more specific than the catch-all at the bottom of this file and Julia reports an
# ambiguity. These two restate the unit law at the `AbstractBelief` level, which is strictly
# more specific than the catch-all, so the unit always wins.
combine(::LenticulumCore.TrivialBelief, b::LenticulumCore.AbstractBelief) = b
combine(a::LenticulumCore.AbstractBelief, ::LenticulumCore.TrivialBelief) = a

function combine(a::LenticulumCore.DiracBelief, b::LenticulumCore.DiracBelief)
    a.value == b.value && return a
    throw(ArgumentError(
        "contradictory hard clamps on one variable: $(a.value) vs $(b.value). \
         Two Dirac beliefs disagreeing is a wiring error — soften one of them \
         (a finite Observed precision) if disagreement is expected."))
end
combine(a::LenticulumCore.DiracBelief, ::LenticulumCore.AbstractBelief) = a
combine(::LenticulumCore.AbstractBelief, b::LenticulumCore.DiracBelief) = b
combine(a::LenticulumCore.DiracBelief, ::LenticulumCore.TrivialBelief) = a
combine(::LenticulumCore.TrivialBelief, b::LenticulumCore.DiracBelief) = b

# --- categorical: elementwise product, exact and total ---------------------------------
function combine(a::LenticulumCore.CategoricalBelief, b::LenticulumCore.CategoricalBelief)
    _same_support(a, b) || throw(ArgumentError(
        "categorical beliefs over different supports ($(length(a.logp)) vs $(length(b.logp)) states, or different labels)"))
    l = a.logp .+ b.logp
    m = maximum(l)
    m == -Inf && throw(ArgumentError(
        "contradictory categorical beliefs: their supports are disjoint, so the product is zero everywhere"))
    return LenticulumCore.CategoricalBelief(Val(:log), l .- (m + log(sum(exp.(l .- m)))), a.labels)
end
_same_support(a, b) = length(a.logp) == length(b.logp) && a.labels == b.labels

# The catch-all. Specific rules above (and in other packages) win whenever they apply; what
# reaches this method is handled by representation-independent rules where they exist:
#   - a SampleBelief pooled with anything that has a density: importance reweighting;
#   - a MixtureBelief pooled with anything: the product of each component, reweighted by its
#     overlap (needs a `combine_with_overlap` rule for the component types);
# and otherwise refused, informatively.
function combine(a::LenticulumCore.AbstractBelief, b::LenticulumCore.AbstractBelief)
    if a isa LenticulumCore.AbstractPotentialBelief && b isa LenticulumCore.AbstractPotentialBelief
        return LenticulumCore.LogPotentialBelief(x -> belief_logdensity(a, x) + belief_logdensity(b, x))
    end
    if a isa LenticulumCore.SampleBelief && b isa LenticulumCore.SampleBelief
        throw(ArgumentError(
            "cannot pool two sample beliefs: neither has a density to reweight the other by. \
             Project one onto a parametric family first (e.g. a Gaussian by moment matching)."))
    end
    a isa LenticulumCore.SampleBelief && return _reweight(a, b)
    b isa LenticulumCore.SampleBelief && return _reweight(b, a)
    if a isa LenticulumCore.AbstractPotentialBelief || b isa LenticulumCore.AbstractPotentialBelief
        throw(ArgumentError(
            "cannot pool $(typeof(a)) with $(typeof(b)) exactly: an arbitrary log-potential " *
            "has no parametric product rule. Supply explicit particles (SampleBelief) to " *
            "reweight, or implement an exact combine rule for a richer potential type; no resampling is performed."))
    end
    (a isa LenticulumCore.MixtureBelief || b isa LenticulumCore.MixtureBelief) && return _mixture_product(a, b)
    throw(ArgumentError(
        "no `combine` method for $(typeof(a)) and $(typeof(b)). Implement `Mycelium.combine` for \
         this pair, or `belief_logdensity` so that sample beliefs can be reweighted by it."))
end

# --- sample beliefs: importance reweighting by the other belief's density --------------
_points(s::LenticulumCore.SampleBelief) = s.samples isa AbstractMatrix ? collect(eachcol(s.samples)) : s.samples

function _reweight(s::LenticulumCore.SampleBelief, b)
    pts = _points(s)
    hasmethod(belief_logdensity, Tuple{typeof(b),typeof(first(pts))}) || throw(ArgumentError(
        "cannot reweight samples by a $(typeof(b)): it has no `belief_logdensity`"))
    logw0 = s.weights === nothing ? zeros(length(pts)) : log.(s.weights)
    l = logw0 .+ [belief_logdensity(b, x) for x in pts]
    m = maximum(l)
    m == -Inf && throw(ArgumentError("every sample has zero density under the other belief: disjoint supports"))
    w = exp.(l .- m)
    w ./= sum(w)
    ess = 1 / sum(abs2, w)                                    # effective sample size
    ess < max(1.5, min(10, 0.01 * length(w))) && @warn(
        "importance reweighting degenerated: effective sample size $(round(ess; digits = 1)) of $(length(w)) samples. \
         The pooled belief is unreliable; use more samples or a parametric belief.", maxlog = 1)
    return LenticulumCore.SampleBelief(s.samples, w)
end

# --- mixtures: componentwise products, weighted by their overlap -----------------------
"""
    combine_with_overlap(c, d) -> (belief, logZ)

Return the pooled belief and ``\\log Z = \\log \\int p_c\\,p_d``.
For proper beliefs this is the exact log overlap. When `d` is improper,
`logZ` is defined up to an additive constant depending on `d` alone, which
cancels when normalising mixture weights. For Gaussian pairs, `c` must be
proper and `d` may be improper. For disjoint categorical support, return
`(c, -Inf)` since the pooled distribution is undefined.
Implemented for categorical pairs here and Gaussian pairs in Mycelium's `gaussian.jl`.
"""
function combine_with_overlap end

function combine_with_overlap(c::LenticulumCore.CategoricalBelief, d::LenticulumCore.CategoricalBelief)
    l = c.logp .+ d.logp
    m = maximum(l)
    logZ = m == -Inf ? -Inf : m + log(sum(exp.(l .- m)))
    return (logZ == -Inf ? c : combine(c, d)), logZ
end

# Gaussian pairs re-centre an improper second operand at the reference of the other side.
_pair_overlap(c, d, ref) = combine_with_overlap(c, d)
_pair_overlap(c::LenticulumCore.GaussianBelief, d::LenticulumCore.GaussianBelief, ref) =
    _overlap(c, d, LenticulumCore.isproper(d) ? nothing : ref)

_improper_first(c, d) = false
_improper_first(c::LenticulumCore.GaussianBelief, d::LenticulumCore.GaussianBelief) =
    !LenticulumCore.isproper(c) && LenticulumCore.isproper(d)

_parts(m::LenticulumCore.MixtureBelief) = (m.components, m.logw)
_parts(b) = ([b], [0.0])

# Re-centring drops ηᵀref − refᵀΛref/2, a constant of the site. It cancels in the weights only
# when every pair shares it, i.e. when the other side is a single improper Gaussian and
# nothing else; with several sites, or proper components beside the site, relative masses
# would change (LP-019 reviews), so then no reference is used. The point is the mean of this
# side's first proper component.
_isimproper(c) = c isa LenticulumCore.GaussianBelief && !LenticulumCore.isproper(c)
function _reference(cs, others)
    (length(others) == 1 && _isimproper(only(others))) || return nothing
    i = findfirst(c -> c isa LenticulumCore.GaussianBelief && LenticulumCore.isproper(c), cs)
    return i === nothing ? nothing : LenticulumCore.belief_mean(cs[i])
end

function _mixture_product(a, b)
    (ca, la), (cb, lb) = _parts(a), _parts(b)
    ra, rb = _reference(ca, cb), _reference(cb, ca)
    comps, logw = Any[], Float64[]
    for (i, c) in enumerate(ca), (j, d) in enumerate(cb)
        hasmethod(combine_with_overlap, Tuple{typeof(c),typeof(d)}) || throw(ArgumentError(
            "no product rule for mixture components of types $(typeof(c)) and $(typeof(d))"))
        # The Gaussian rule needs a proper first operand; the product itself is symmetric, so
        # put an improper site second whichever side of the product it came from (LP-019).
        cd, logZ = _improper_first(c, d) ? _pair_overlap(d, c, rb) : _pair_overlap(c, d, ra)
        logZ == -Inf && continue                              # disjoint: this pair contributes nothing
        push!(comps, cd); push!(logw, la[i] + lb[j] + logZ)
    end
    isempty(comps) && throw(ArgumentError("contradictory beliefs: every component pair has disjoint support"))
    # Subtract the maximum before adding the correction: at |log w| ~ 1e16, m + log Σ… rounds
    # back to m and the weights would not sum to one.
    shifted = logw .- maximum(logw)
    logw = shifted .- log(sum(exp.(shifted)))
    return LenticulumCore.MixtureBelief(Val(:log), [c for c in comps], logw)
end

"""
    belief_logdensity(b::AbstractBelief, x) -> Real

``\\log p_b(x)``. Implemented for categorical beliefs (``x`` a state index or label), mixtures
(the log of the weighted sum of the components' densities), Gaussians, and callable potentials.
Any belief with a density can reweight a `SampleBelief` in [`combine`](@ref).
For `AbstractPotentialBelief` this is a log-potential defined up to an additive constant,
not a normalised probability density. Subtypes provide their own evaluation method.
"""
function belief_logdensity end

belief_logdensity(b::LenticulumCore.LogPotentialBelief, x) = b.logφ(x)

function belief_logdensity(b::LenticulumCore.CategoricalBelief, x)
    if b.labels === nothing
        x isa Integer || throw(ArgumentError("a categorical belief without labels takes a state index"))
        return 1 ≤ x ≤ length(b.logp) ? b.logp[x] : -Inf
    end
    k = findfirst(==(x), b.labels)
    return k === nothing ? -Inf : b.logp[k]
end

function belief_logdensity(m::LenticulumCore.MixtureBelief, x)
    l = m.logw .+ [belief_logdensity(c, x) for c in m.components]
    mx = maximum(l)
    return mx == -Inf ? -Inf : mx + log(sum(exp.(l .- mx)))
end

"""
    belief_distance(a, b) -> Real

A non-negative measure of how much a message changed, used as the convergence criterion in
[`propagate!`](@ref).

Implemented for the cases that can be answered without densities; otherwise returns `Inf`,
which makes the scheduler run to `maxiters` rather than silently declaring convergence it
cannot verify. **Erring towards `Inf` is deliberate**: a false "converged" is much worse
than a wasted sweep.

`LogPotentialBelief` pairs have distance zero only when their wrapped callables are
identical (`===`), and `Inf` otherwise. This assumes captured mutable data is unchanged;
no numerical comparison up to additive constants is attempted. Other potential subtypes
retain the conservative `Inf` fallback.
"""
belief_distance(::Nothing, ::Any) = Inf
belief_distance(::Any, ::Nothing) = Inf
belief_distance(::Nothing, ::Nothing) = Inf
belief_distance(::LenticulumCore.TrivialBelief, ::LenticulumCore.TrivialBelief) = 0.0
# Callable identity certifies only an unchanged evaluator, not equivalence up to a constant.
# Other potential pairs keep the conservative AbstractBelief fallback (Inf).
belief_distance(a::LenticulumCore.LogPotentialBelief, b::LenticulumCore.LogPotentialBelief) =
    a.logφ === b.logφ ? 0.0 : Inf
function belief_distance(a::LenticulumCore.DiracBelief, b::LenticulumCore.DiracBelief)
    return a.value == b.value ? 0.0 : _numeric_distance(a.value, b.value)
end
belief_distance(a::LenticulumCore.CategoricalBelief, b::LenticulumCore.CategoricalBelief) =
    _same_support(a, b) ? sum(abs, exp.(a.logp) .- exp.(b.logp)) / 2 : Inf   # total variation
function belief_distance(a::LenticulumCore.MixtureBelief, b::LenticulumCore.MixtureBelief)
    length(a.components) == length(b.components) || return Inf
    wdist = sum(abs, exp.(a.logw) .- exp.(b.logw)) / 2
    cdist = maximum(k -> belief_distance(a.components[k], b.components[k]), eachindex(a.components); init = 0.0)
    return max(wdist, cdist)
end
function _sample_weights(b::LenticulumCore.SampleBelief)
    n = length(_points(b))
    b.weights === nothing && return fill(1 / n, n)
    w = float.(b.weights)
    m = maximum(w)
    m > 0 || return fill(1 / n, n)
    w = w ./ m
    return w ./ sum(w)
end
function belief_distance(a::LenticulumCore.SampleBelief, b::LenticulumCore.SampleBelief)
    pa, pb = _points(a), _points(b)
    length(pa) == length(pb) || return Inf
    wa, wb = _sample_weights(a), _sample_weights(b)
    wdist = sum(abs, wa .- wb) / 2
    a.samples == b.samples && return wdist
    pdist = maximum(i -> _numeric_distance(pa[i], pb[i]), eachindex(pa); init = 0.0)
    return max(wdist, pdist)
end
belief_distance(::LenticulumCore.AbstractBelief, ::LenticulumCore.AbstractBelief) = Inf

_numeric_distance(a::Number, b::Number) = abs(float(a - b))
_numeric_distance(a::AbstractArray, b::AbstractArray) = sqrt(sum(abs2, a .- b))
_numeric_distance(a, b) = a == b ? 0.0 : Inf

# --- Marginals and the exclusion principle ---------------------------------

"""
    marginal(store, g, vid) -> AbstractBelief

The belief at a variable: the combination of **all** incoming factor → variable messages.

This is what you read out at the end of inference.
"""
function marginal(s::AbstractMessageStore, g::FactorGraph, vid::Int)
    acc = LenticulumCore.TrivialBelief()
    for ei in g.edges_of_variable[vid]
        m = s.to_variable[ei]
        m === nothing && continue
        acc = combine(acc, g.edges[ei].detach ? LenticulumCore.stop_gradient(m.belief) : m.belief)
    end
    return acc
end

"""
    excluded_marginal(store, g, vid, skip_edge) -> AbstractBelief

The belief at a variable **excluding** the message that arrived along `skip_edge` — the
variable → factor message of belief propagation.

> **This exclusion is not an optimisation; it is what makes BP correct.** Without it a
> factor's own previous message is fed back to it as if it were independent evidence, and
> the belief becomes exponentially over-confident with each sweep. On a tree the exclusion
> is exactly what makes two sweeps give the true marginals.

Note that we *recompute* the product over the other edges rather than dividing the full
marginal by the skipped message. Division requires densities and is numerically fragile;
recomputation costs `O(degree)` per message and is always defined.
"""
function excluded_marginal(s::AbstractMessageStore, g::FactorGraph, vid::Int, skip::Int)
    acc = LenticulumCore.TrivialBelief()
    for ei in g.edges_of_variable[vid]
        ei == skip && continue
        m = s.to_variable[ei]
        m === nothing && continue
        acc = combine(acc, g.edges[ei].detach ? LenticulumCore.stop_gradient(m.belief) : m.belief)
    end
    return acc
end

"""
    damp(new, old, α) -> belief

``\\alpha \\cdot \\text{new} + (1-\\alpha)\\cdot\\text{old}``, the standard remedy for
oscillating loopy BP.

Defined only where a convex combination of beliefs makes sense. For `DiracBelief` with
numeric payloads this is interpolation of the values; for anything else, damping is a no-op
returning `new`, inspectable through [`can_damp`](@ref). Potentials use this fallback:
without a normalisation contract there is no chosen convex-mixture interpretation.
Schedules do not automatically report unsupported damping.
"""
damp(new, old, α::Real) = can_damp(new, old) ? _damp(new, old, α) : new

"""
    can_damp(new, old) -> Bool

Whether a convex combination of these two beliefs is defined. `false` by default, so
callers can inspect whether [`damp`](@ref) will be a no-op.
"""
can_damp(::Any, ::Any) = false
can_damp(a::LenticulumCore.DiracBelief{<:Union{Number,AbstractArray}},
         b::LenticulumCore.DiracBelief{<:Union{Number,AbstractArray}}) = true
_damp(a::LenticulumCore.DiracBelief, b::LenticulumCore.DiracBelief, α::Real) =
    LenticulumCore.DiracBelief(α .* a.value .+ (1 - α) .* b.value)
can_damp(a::LenticulumCore.CategoricalBelief, b::LenticulumCore.CategoricalBelief) = _same_support(a, b)
_damp(a::LenticulumCore.CategoricalBelief, b::LenticulumCore.CategoricalBelief, α::Real) =
    LenticulumCore.CategoricalBelief(α .* exp.(a.logp) .+ (1 - α) .* exp.(b.logp); labels = a.labels)
can_damp(a::LenticulumCore.MixtureBelief, b::LenticulumCore.MixtureBelief) =
    length(a.components) == length(b.components) &&
    all(k -> belief_distance(a.components[k], b.components[k]) == 0, eachindex(a.components))
_damp(a::LenticulumCore.MixtureBelief, b::LenticulumCore.MixtureBelief, α::Real) =
    LenticulumCore.MixtureBelief(a.components, α .* exp.(a.logw) .+ (1 - α) .* exp.(b.logw))
can_damp(a::LenticulumCore.SampleBelief, b::LenticulumCore.SampleBelief) = a.samples == b.samples
_damp(a::LenticulumCore.SampleBelief, b::LenticulumCore.SampleBelief, α::Real) =
    LenticulumCore.SampleBelief(a.samples, α .* _sample_weights(a) .+ (1 - α) .* _sample_weights(b))
