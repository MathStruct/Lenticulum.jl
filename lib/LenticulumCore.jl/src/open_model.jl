# ---------------------------------------------------------------------------
# Open models (AutoBayes Def. 1) and beliefs.
#
# The defining feature is the latent space ⟦c⟧: composition FILES the intermediate value
# away instead of integrating it out, so composing open models costs no integral. That is
# the reason the Bayesian chain rule is usable at all. See `open_model.md`.
# ---------------------------------------------------------------------------

"""
    OpenModelResult(observed, latent)

The result of running an open model forward: a value in ``Y`` together with the value in
``\\llbracket c \\rrbracket`` that composition would otherwise have destroyed.

This is the exact analogue of an autodiff tape entry. Reverse-mode AD caches activations so
the backward pass can use them; an open model caches ``\\llbracket c \\rrbracket`` so the
inversion can. Same trade of memory for tractability, under the same chain rule.
"""
struct OpenModelResult{Y,A}
    observed::Y
    latent::A
end

Base.show(io::IO, r::OpenModelResult) = print(io, "OpenModelResult(", r.observed, ", ⟦", r.latent, "⟧)")

"""
    forward(model, x, ps, st) -> (OpenModelResult, st)

Sample the kernel ``c : X \\rightsquigarrow \\llbracket c \\rrbracket \\times Y`` at `x`.
"""
function forward end

"""
    logdensity(model, x, a, y, ps, st) -> (Real, st)

``\\log p_c(a, y \\mid x)``. Needed whenever the energy is a negative log-likelihood, which
is the default choice in every example of AutoBayes' Appendix A.
"""
function logdensity end

"""
    pushforward(model, π::AbstractBelief, ps, st) -> (AbstractBelief, st)

The pushforward prior ``c_*\\pi``.

!!! warning "This is one of the two expensive operations"
    Computing ``c_*\\pi`` is marginalisation, and is about as costly as exact inversion.
    AutoBayes' closing discussion names belief propagation and variational message passing
    as the remedy, and states that they fit in the framework — that is Mycelium.jl's job.
    A `LenticulumCore` implementation may legitimately return an approximate belief, but it
    should say so via [`isexact`](@ref).
"""
function pushforward end

"""
    isexact(x) -> Bool

Whether a pushforward or an inversion is exact rather than approximate. Defaults to `false`
because approximation is the normal case and silence should not imply exactness.
"""
isexact(::Any) = false

"""
    latentspace(model)
    observedspace(model)
    unobservedspace(model)

The three spaces ``\\llbracket c \\rrbracket``, ``Y``, ``X`` of AutoBayes Definition 1.
"""
function latentspace end
function observedspace end
function unobservedspace end

"""
    ispure(model) -> Bool

``\\llbracket c \\rrbracket \\cong 1``: an ordinary kernel ``X \\rightsquigarrow Y``.

Purity is **not preserved by composition** — that is precisely why the latent space exists.
Composing two pure models yields a model whose latent space is the intermediate space.
"""
ispure(::AbstractOpenModel) = false

# --- Minimal concrete beliefs ---------------------------------------------

"""
    DiracBelief(value)

A point mass. What a clamped (observed) channel carries, and what a
categorical *cup* produces (see `Copiers Cups and Caps.md`). Inversions against a Dirac prior typically
trivialise — which is exactly AutoBayes' Example 4: supervised learning is the case where
the cup has collapsed the posterior.
"""
struct DiracBelief{T} <: AbstractBelief
    value::T
end
isexact(::DiracBelief) = true

"""
    SampleBelief(samples, [weights])

A particle representation of ``\\pi \\in \\mathcal{P}X``. The default fallback whenever no
conjugate structure is available, which is most of the time.
"""
struct SampleBelief{S,W} <: AbstractBelief
    samples::S
    weights::W
end
SampleBelief(samples) = SampleBelief(samples, nothing)

"""
    TrivialBelief()

The unique belief on the one-point space ``1``. The prior argument of a factor with
``X \\cong 1`` — a prior distribution, in AutoBayes' terminology, has this as its input.
"""
struct TrivialBelief <: AbstractBelief end
isexact(::TrivialBelief) = true

"""
    CategoricalBelief(p; labels = nothing)

A distribution over a **finite set** ``\\{1, \\dots, K\\}`` (or over `labels`), stored as
normalised log-probabilities `logp`. `p` is any non-negative vector with a positive sum; it is
normalised. Pooling two categorical beliefs is elementwise multiplication and renormalisation:
exact, associative, commutative, and always defined unless the two have disjoint support
(`Belief Algebra.md`, `Categorical Belief.md`). [`bernoulli`](@ref) is the two-state case.
"""
struct CategoricalBelief{T<:Real,L} <: AbstractBelief
    logp::Vector{T}
    labels::L
    # internal: already-normalised log-probabilities (used by the pooling rules)
    CategoricalBelief(::Val{:log}, logp::Vector{T}, labels::L) where {T<:Real,L} = new{T,L}(logp, labels)
end
function CategoricalBelief(p::AbstractVector{<:Real}; labels = nothing)
    any(<(0), p) && throw(ArgumentError("probabilities must be non-negative"))
    s = sum(p)
    s > 0 || throw(ArgumentError("probabilities must not all be zero"))
    labels === nothing || length(labels) == length(p) ||
        throw(ArgumentError("need one label per state: $(length(labels)) labels for $(length(p)) states"))
    return CategoricalBelief(Val(:log), log.(float.(p) ./ s), labels === nothing ? nothing : collect(labels))
end

"""
    probabilities(b::CategoricalBelief) -> Vector

The probabilities of the states, ``\\exp`` of the stored log-probabilities.
"""
probabilities(b::CategoricalBelief) = exp.(b.logp)

"""
    bernoulli(p; labels = [false, true]) -> CategoricalBelief

A two-state belief: probability `p` of the second state (`true`).
"""
bernoulli(p::Real; labels = [false, true]) = CategoricalBelief([1 - p, p]; labels)

"""
    MixtureBelief(components, weights = equal)

A weighted mixture ``\\sum_k w_k\\, b_k`` of beliefs of one type, with normalised log-weights
`logw`. It is how a belief holds **several answers at once**, e.g. both branches of a relation.
Its density is the weighted sum of the components' densities; its product with another belief
is again a mixture, with weights scaled by each component's overlap with that belief
(`Mixture Belief.md`).
"""
struct MixtureBelief{B<:AbstractBelief,T<:Real} <: AbstractBelief
    components::Vector{B}
    logw::Vector{T}
    # internal: already-normalised log-weights (used by the product and reduction rules)
    MixtureBelief(::Val{:log}, components::Vector{B}, logw::Vector{T}) where {B<:AbstractBelief,T<:Real} =
        new{B,T}(components, logw)
end
function MixtureBelief(components::AbstractVector{<:AbstractBelief}, weights::AbstractVector{<:Real} = fill(1.0, length(components)))
    isempty(components) && throw(ArgumentError("a mixture needs at least one component"))
    length(weights) == length(components) || throw(ArgumentError("need one weight per component"))
    any(<(0), weights) && throw(ArgumentError("weights must be non-negative"))
    s = sum(weights)
    s > 0 || throw(ArgumentError("weights must not all be zero"))
    return MixtureBelief(Val(:log), [c for c in components], log.(float.(weights) ./ s))
end

"""
    mixture_weights(m::MixtureBelief) -> Vector

The mixture weights, ``\\exp`` of the stored log-weights.
"""
mixture_weights(m::MixtureBelief) = exp.(m.logw)
