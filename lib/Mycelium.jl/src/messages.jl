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
mutable struct MessageStore
    to_variable::Vector{Any}
    to_factor::Vector{Any}
    iteration::Int
end
MessageStore(g::FactorGraph) =
    MessageStore(Vector{Any}(nothing, nedges(g)), Vector{Any}(nothing, nedges(g)), 0)

has_to_variable(s::MessageStore, ei::Int) = s.to_variable[ei] !== nothing
has_to_factor(s::MessageStore, ei::Int) = s.to_factor[ei] !== nothing

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

Everything else throws an informative error. In particular **two `SampleBelief`s cannot be
combined without densities** — that needs importance reweighting, which needs
[`belief_logdensity`](@ref), which no belief type currently implements. See `messages.md`
§"Implementation difficulties".
"""
function combine end

combine(::LenticulumCore.TrivialBelief, b) = b
combine(a, ::LenticulumCore.TrivialBelief) = a
combine(a::LenticulumCore.TrivialBelief, ::LenticulumCore.TrivialBelief) = a

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

combine(a::LenticulumCore.AbstractBelief, b::LenticulumCore.AbstractBelief) =
    throw(ArgumentError(
        "no `combine` method for $(typeof(a)) and $(typeof(b)). Pooling general beliefs \
         requires densities (importance reweighting); implement `Mycelium.combine` for \
         your belief representation, or `belief_logdensity` and use the generic path."))

"""
    belief_logdensity(b::AbstractBelief, x) -> Real

``\\log p_b(x)``. Not implemented by any `LenticulumCore` belief type yet; it is the missing
piece that would make a generic [`combine`](@ref) possible. Declared here so that downstream
belief representations have a name to extend.
"""
function belief_logdensity end

"""
    belief_distance(a, b) -> Real

A non-negative measure of how much a message changed, used as the convergence criterion in
[`propagate!`](@ref).

Implemented for the cases that can be answered without densities; otherwise returns `Inf`,
which makes the scheduler run to `maxiters` rather than silently declaring convergence it
cannot verify. **Erring towards `Inf` is deliberate**: a false "converged" is much worse
than a wasted sweep.
"""
belief_distance(::Nothing, ::Any) = Inf
belief_distance(::Any, ::Nothing) = Inf
belief_distance(::LenticulumCore.TrivialBelief, ::LenticulumCore.TrivialBelief) = 0.0
function belief_distance(a::LenticulumCore.DiracBelief, b::LenticulumCore.DiracBelief)
    return a.value == b.value ? 0.0 : _numeric_distance(a.value, b.value)
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
function marginal(s::MessageStore, g::FactorGraph, vid::Int)
    acc = LenticulumCore.TrivialBelief()
    for ei in g.edges_of_variable[vid]
        m = s.to_variable[ei]
        m === nothing && continue
        acc = combine(acc, m.belief)
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
function excluded_marginal(s::MessageStore, g::FactorGraph, vid::Int, skip::Int)
    acc = LenticulumCore.TrivialBelief()
    for ei in g.edges_of_variable[vid]
        ei == skip && continue
        m = s.to_variable[ei]
        m === nothing && continue
        acc = combine(acc, m.belief)
    end
    return acc
end

"""
    damp(new, old, α) -> belief

``\\alpha \\cdot \\text{new} + (1-\\alpha)\\cdot\\text{old}``, the standard remedy for
oscillating loopy BP.

Defined only where a convex combination of beliefs makes sense. For `DiracBelief` with
numeric payloads this is interpolation of the values; for anything else, damping is a no-op
returning `new`, **and that is reported** rather than silently ignored — see
[`can_damp`](@ref).
"""
damp(new, old, α::Real) = can_damp(new, old) ? _damp(new, old, α) : new
can_damp(::Any, ::Any) = false
can_damp(a::LenticulumCore.DiracBelief{<:Union{Number,AbstractArray}},
         b::LenticulumCore.DiracBelief{<:Union{Number,AbstractArray}}) = true
_damp(a::LenticulumCore.DiracBelief, b::LenticulumCore.DiracBelief, α::Real) =
    LenticulumCore.DiracBelief(α .* a.value .+ (1 - α) .* b.value)
