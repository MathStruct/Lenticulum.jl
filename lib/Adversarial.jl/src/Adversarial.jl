"""
    Adversarial

Implicit **generative** models as Lenticulum factors — Mohamed & Lakshminarayanan's
*Learning in Implicit Generative Models* ([arXiv:1610.03483](https://arxiv.org/pdf/1610.03483)),
and the GAN diagram read as a factor graph.

## The three factors

| factor | is | polarities |
|---|---|---|
| [`NoiseSource`](@ref) | the latent prior ``q(z)``, as an emitting factor | 1 |
| [`GeneratorFactor`](@ref) | ``x = G_\\theta(z)`` — samples out, no density | 1 |
| [`RatioFactor`](@ref) | ``\\log r(x) = \\log p(x) - \\log q(x)`` — the discriminator | 1 |

Every one of them is unidirectional, and that is the finding rather than a shortcoming: a
generator is a *function* with no residual behind it, so there is nothing to run backwards.
See [[Three Senses of Implicit]] for why a model can be "implicit" and still have exactly one
direction.

## Two things this package is for

**1. It is the first thing in the project that produces a `SampleBelief`.** `LenticulumCore`
has declared that type since the beginning and nothing has ever constructed one. A generator
is what it was for — and the moment one exists, `Mycelium.combine` throws, which is
`messages.md` §1's recorded main gap made concrete.

**2. A discriminator is a way around that gap.** The central identity of the paper is

```math
D^\\ast(x) = \\frac{p(x)}{p(x)+q(x)}
\\qquad\\Longrightarrow\\qquad
\\operatorname{logit} D^\\ast(x) = \\log p(x) - \\log q(x)
```

so a classifier estimates a log-density **ratio** without either density — and a ratio is
precisely what importance reweighting needs. [`reweight`](@ref) is the operation;
[`effective_sample_size`](@ref) is the diagnostic that says whether it meant anything.

## What it cannot do

**Train adversarially.** The generator descends the same quantity the discriminator ascends,
and a Bethe free energy has one sign. The graph expresses the GAN's *wiring* exactly and its
*objective* not at all — see [[GANs as Two Factors]] §4, which argues the missing structure is
Ghani–Hedges–Winschel–Zahn's **open game**, a third lens-shaped object alongside the
parametric lens and the statistical game.

Dependencies are `LuxCore`, `Random`, `LinearAlgebra` — no `Lux`, no AD, no training loop.
This package scores and reweights; fitting the two networks is somebody else's job.

Concept notes: [[Implicit Generative Models]], [[GANs as Two Factors]],
[[Three Senses of Implicit]]. Per-file notes: [[generator]], [[ratio]].
"""
module Adversarial

using DispatchDoctor: @stable
using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG, randn
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium

# Reading a value out of an incoming message, shared by both factors.
_get(inputs, k) = (inputs isa NamedTuple && haskey(inputs, k)) ? getfield(inputs, k) : nothing
_vec(v::AbstractVector) = float.(v)
_vec(v::Real) = [float(v)]

include("generator.jl")
include("ratio.jl")

# --- The latent prior and the generator ------------------------------------
export NoiseSource, GeneratorFactor, GeneratorModel
export generate                      # `pushforward` is LenticulumCore's, extended here
export latentchannel, samplechannel, latentdim, sampledim

# --- The discriminator, read as a ratio estimator --------------------------
export RatioFactor, RatioModel
export logratio, discriminator, reweight
export effective_sample_size, weighted_mean, weighted_cov

end # module
