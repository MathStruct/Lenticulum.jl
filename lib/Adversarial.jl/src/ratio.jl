# ---------------------------------------------------------------------------
# The discriminator, read as what it actually is: a DENSITY RATIO ESTIMATOR.
#
# Mohamed & Lakshminarayanan's central identity (arXiv:1610.03483 §3). Train a classifier to
# separate p (label 1) from q (label 0) with equal class priors. At the optimum
#
#     D*(x) = p(x) / (p(x) + q(x))        ⟹        logit D*(x) = log p(x) - log q(x)
#
# **The logit of the optimal discriminator is the log density ratio.** That one line is why
# GANs work, and it is why this file matters to Lenticulum for a reason that has nothing to do
# with generating images:
#
#     `messages.md` §1 records that the package's main gap is `combine` for particle beliefs,
#     which needs densities, which no sample-based belief has. A log-density-RATIO is exactly
#     what importance reweighting needs, and a classifier estimates one without either density.
#
# So the GAN's discriminator is a candidate answer to this project's oldest recorded blocker.
# See `ratio.md` §4 and `Implicit Generative Models.md` §5.
# ---------------------------------------------------------------------------

"""
    RatioFactor(net, dim; channel = :x, input = identity)

A **density-ratio factor**: a unary factor on `channel` whose potential is an estimated
``\\log r(x) = \\log p(x) - \\log q(x)``.

`net` is any `LuxCore.AbstractLuxLayer` mapping a sample to a **logit** (a real number, or a
length-1 vector). It is the GAN discriminator, and it is read as a ratio estimator rather than
as a classifier — the two are the same object under the identity above.

```julia
d = RatioFactor(my_critic, 2)          # a unary factor on :x
```

!!! note "The energy is estimated, not computed"
    ``\\mathbf{l}^c(x) = -\\log r(x)``, so ``\\mathbb{E}_{q}[-\\log r] = \\mathrm{KL}(q\\,\\|\\,p)``
    — the factor's energy is the integrand of a KL divergence. But ``\\log r`` comes from a
    *fitted network*, so unlike every other factor in this project the energy itself is an
    **estimate**.

    `Bayesian Lens.md` licenses inexact *inversions*. This is a different thing: an inexact
    **energy**, and the free energy has no term that accounts for it. See `ratio.md` §5.
"""
struct RatioFactor{N,F} <: LenticulumCore.AbstractLenticulumFactor
    channel::Symbol
    dim::Int
    net::N
    input::F
end

function RatioFactor(net, dim::Int; channel::Symbol = :x, input = identity)
    dim > 0 || throw(ArgumentError("dim must be positive; got $dim"))
    return RatioFactor(channel, dim, net, input)
end

LuxCore.initialparameters(rng::AbstractRNG, f::RatioFactor) =
    LuxCore.initialparameters(rng, f.net)
LuxCore.initialstates(rng::AbstractRNG, f::RatioFactor) = LuxCore.initialstates(rng, f.net)
LuxCore.parameterlength(f::RatioFactor) = LuxCore.parameterlength(f.net)
LuxCore.statelength(f::RatioFactor) = LuxCore.statelength(f.net)

LenticulumCore.channels(f::RatioFactor) = (LenticulumCore.Channel(f.channel, f.dim),)
LenticulumCore.supported_polarities(f::RatioFactor) =
    (LenticulumCore.Polarity(NamedTuple{(f.channel,)}((LenticulumCore.Unobserved(),))),)
LenticulumCore.supports_polarity(f::RatioFactor, p::LenticulumCore.Polarity) =
    keys(p) == (f.channel,) && p[f.channel] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(::RatioFactor) = true

LenticulumCore.energyspace(::RatioFactor) = LenticulumCore.ScalarEnergySpace()

"""
    logratio(f::RatioFactor, x, ps, st) -> (Real, st)

``\\log r(x) = \\log p(x) - \\log q(x)``, i.e. the discriminator's logit.
"""
function logratio(f::RatioFactor, x, ps, st)
    ℓ, st = LuxCore.apply(f.net, f.input(x), ps, st)
    return (_scalar(ℓ), st)
end

_scalar(v::Real) = float(v)
_scalar(v::AbstractArray) = (length(v) == 1 ? float(only(v)) :
    throw(DimensionMismatch("a RatioFactor's net must return one logit; got $(size(v))")))

"""
    discriminator(f::RatioFactor, x, ps, st) -> (Real, st)

``D(x) = \\sigma(\\log r(x)) = p(x)/(p(x)+q(x))`` — the classifier reading of the same number.

Provided so the identity is visible in code: `discriminator` and [`logratio`](@ref) are the
logistic transform of one another, and which one you call is a matter of which paper you are
reading.
"""
function discriminator(f::RatioFactor, x, ps, st)
    ℓ, st = logratio(f, x, ps, st)
    return (_sigmoid(ℓ), st)
end
_sigmoid(t) = t >= 0 ? 1 / (1 + exp(-t)) : exp(t) / (1 + exp(t))

# Energy: -log r. Minimising E_q[-log r] over q's parameters is minimising KL(q‖p).
function LenticulumCore.energy(f::RatioFactor, x, a, y, ps, st)
    ℓ, st = logratio(f, x, ps, st)
    return (-ℓ, st)
end

# --- Reweighting: what the ratio is FOR ------------------------------------

"""
    reweight(f::RatioFactor, b::SampleBelief, ps, st) -> (SampleBelief, st)

Importance-reweight a particle set by the estimated ratio:
``w_i \\propto w_i^{\\text{old}}\\, r(x_i)``.

If `b`'s particles are drawn from ``q`` and ``r = p/q``, the result is a weighted particle
approximation of ``p``. **This is the operation `messages.md` §1 says is missing** — pooling
a sample-based belief with another distribution — performed by the one object in the graph
that knows the ratio.

Weights are normalised and computed in log space (a shift by the maximum before
exponentiating), because raw ratios overflow for any interesting `p`/`q` pair.

> [!warning] Reweighting concentrates
> The variance of the weights grows with the divergence between `p` and `q`, and the particle
> set silently collapses onto a handful of samples. Check [`effective_sample_size`](@ref);
> it is the diagnostic that says whether the answer means anything.
"""
function reweight(f::RatioFactor, b::LenticulumCore.SampleBelief, ps, st)
    n = length(b.samples)
    logw = Vector{Float64}(undef, n)
    for i in 1:n
        ℓ, st = logratio(f, b.samples[i], ps, st)
        logw[i] = ℓ
    end
    prior = b.weights === nothing ? zeros(n) : log.(max.(float.(b.weights), eps()))
    return (LenticulumCore.SampleBelief(b.samples, _normalise_logweights(logw .+ prior)), st)
end

function _normalise_logweights(logw::AbstractVector)
    m = maximum(logw)
    w = exp.(logw .- m)
    s = sum(w)
    s > 0 || throw(ArgumentError(
        "all importance weights underflowed to zero; the ratio estimator and the particle \
         set have disjoint support"))
    return w ./ s
end

"""
    effective_sample_size(b::SampleBelief) -> Real

``\\mathrm{ESS} = 1/\\sum_i w_i^2`` for normalised weights; `length(samples)` when unweighted.

Ranges from 1 (one particle carries everything — the answer is a point estimate wearing a
distribution's clothes) to `n` (uniform). **Report it or the reweighting means nothing**, and
it is the honest reason a ratio-based `combine` is not simply dropped into
`Mycelium.combine`: the operation is defined but its quality is not, and a `combine` that
silently degenerates is worse than one that throws.
"""
function effective_sample_size(b::LenticulumCore.SampleBelief)
    b.weights === nothing && return float(length(b.samples))
    w = float.(b.weights)
    s = sum(w)
    s > 0 || return 0.0
    w = w ./ s
    return 1 / sum(abs2, w)
end

"""
    weighted_mean(b::SampleBelief)
    weighted_cov(b::SampleBelief)

Self-normalised importance-sampling estimates of the first two moments.

These are the only way to read anything out of a `SampleBelief`, and they are how the test
suite checks reweighting against a Gaussian closed form.
"""
function weighted_mean(b::LenticulumCore.SampleBelief)
    w = _weights(b)
    acc = zeros(Float64, length(first(b.samples)))
    for i in eachindex(b.samples)
        acc .+= w[i] .* float.(b.samples[i])
    end
    return acc
end

function weighted_cov(b::LenticulumCore.SampleBelief)
    w = _weights(b)
    m = weighted_mean(b)
    d = length(m)
    acc = zeros(Float64, d, d)
    for i in eachindex(b.samples)
        r = float.(b.samples[i]) .- m
        acc .+= w[i] .* (r * r')
    end
    return acc
end

function _weights(b::LenticulumCore.SampleBelief)
    n = length(b.samples)
    b.weights === nothing && return fill(1 / n, n)
    w = float.(b.weights)
    return w ./ sum(w)
end

# --- The factor interface --------------------------------------------------

"""
    RatioModel(factor)

The open model of a unary evidence factor. There is no forward map — a ratio factor does not
*produce* an `x`, it *scores* one — so the inversion is where all the content is.
"""
struct RatioModel{F<:RatioFactor} <: LenticulumCore.AbstractOpenModel
    factor::F
end
LenticulumCore.ispure(::RatioModel) = true
LenticulumCore.latentspace(::RatioModel) = nothing

LenticulumCore.assemble(f::RatioFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(RatioModel(f), LenticulumCore.AmortisedInversion(f.net)), st)

"""
    LenticulumCore.invert(lens, π, inputs, ps, st)

Reweight the prior `π` by the estimated ratio.

`AmortisedInversion` is the right label: the inversion *is* a learned network with its own
parameters, which `lens.md` says is the structural reason a factor cannot be a Lux layer. A
`RatioFactor` is the first factor in the project where that description is literally true —
the discriminator is trained separately from whatever produced the samples.

> [!warning] The prior is consumed, so this is a posterior
> A unary factor has no other channel to read, so the only thing to reweight is the incoming
> belief — which makes the message a posterior rather than a likelihood, and double-counts on
> a variable of degree > 1. The same wall the diffusion and DEQ factors hit, for the same
> reason: there is nothing to divide out of a neural network. See `ratio.md` §6.
"""
function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:RatioModel,<:LenticulumCore.AmortisedInversion},
    π, inputs, ps, st,
)
    return _score(lens.model.factor, something(_get(inputs, lens.model.factor.channel), π), ps, st)
end

_score(f::RatioFactor, b::LenticulumCore.SampleBelief, ps, st) = reweight(f, b, ps, st)
_score(::RatioFactor, b::LenticulumCore.DiracBelief, ps, st) = (b, st)   # a clamp dominates
_score(::RatioFactor, ::LenticulumCore.TrivialBelief, ps, st) =
    (LenticulumCore.TrivialBelief(), st)
_score(::RatioFactor, ::Nothing, ps, st) = (LenticulumCore.TrivialBelief(), st)
_score(f::RatioFactor, b, ps, st) = throw(ArgumentError(
    "a RatioFactor can only reweight a SampleBelief (or pass a DiracBelief through); got \
     $(typeof(b)). Reweighting a parametric belief would need to resample it first."))

"""
    Mycelium.factor_message(f::RatioFactor, target, polarity, inputs, prior, ps, st)

The reweighted belief on `f`'s single channel. Falls back to the `prior` when no message has
arrived — which for a unary factor is the normal case, since its only neighbour is its target.
"""
function Mycelium.factor_message(
    f::RatioFactor, target::Symbol, polarity, inputs, prior, ps, st
)
    target === f.channel || throw(ArgumentError(
        "channel :$target is not the channel of this RatioFactor (has :$(f.channel))"))
    return _score(f, something(_get(inputs, f.channel), prior), ps, st)
end

"""
    Mycelium.local_free_energy(f::RatioFactor, msgs, ps, st)

``\\mathbb{E}_{b}[-\\log r(x)]`` under the incoming particle set — a Monte-Carlo estimate of
``\\mathrm{KL}(q\\,\\|\\,p)`` when `r` is well fitted.

This is the number a GAN's generator descends, and the sign is worth dwelling on: the
*discriminator* ascends the same quantity. A Bethe free energy has one sign, so the graph can
express half of a GAN. See `GANs as Two Factors.md` §4.
"""
function Mycelium.local_free_energy(f::RatioFactor, msgs, ps, st)
    b = _get(msgs, f.channel)
    b isa LenticulumCore.SampleBelief || return (0.0, st)
    w = _weights(b)
    acc = 0.0
    for i in eachindex(b.samples)
        ℓ, st = logratio(f, b.samples[i], ps, st)
        acc += w[i] * (-ℓ)
    end
    return (acc, st)
end
