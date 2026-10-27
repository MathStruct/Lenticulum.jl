# [Adversarial](@id adversarial)

```@meta
CurrentModule = Adversarial
```

**Implicit generative models** as factors — models you can sample from but whose density you
cannot evaluate — and the discriminator read as what it is: a density-ratio estimator.

## Three factors

```julia
src = NoiseSource(:z, 8; nsamples = 512)              # q(z), emits particles
gen = GeneratorFactor(my_decoder, (z = 8, x = 2))     # x = G_θ(z)
rat = RatioFactor(my_critic, 2)                        # log p(x) - log q(x)
```

All three are **unidirectional**. A generator is a function with no residual behind it, so
there is nothing to run backwards — inverting ``G_\theta`` is the GAN-inversion problem, and
`factor_message` on the latent channel says so rather than pretending.

This package is the only one that produces a `SampleBelief`: a weighted particle set rather
than a point or a parametric distribution.

## The ratio identity

Train a classifier to separate ``p`` from ``q`` with equal class priors, and at the optimum

```math
\operatorname{logit} D^\ast(x) = \log p(x) - \log q(x)
```

so the discriminator's logit *is* the log density ratio. `logratio` and `discriminator` are
the two readings of that number.

## Reweighting

The reason a ratio is useful here has little to do with generating images: importance
reweighting needs a *ratio*, not a density, and that is what a classifier gives you.

```julia
pb, st = reweight(rat, particle_belief, ps, st)   # w_i ∝ w_i · r(x_i)
effective_sample_size(pb)                          # ...and check it meant something
weighted_mean(pb), weighted_cov(pb)
```

Always look at `effective_sample_size`. Reweighting concentrates, and the particle set can
collapse onto a handful of samples with no error raised.

## What it cannot do

**Train adversarially.** The generator descends the same quantity the discriminator ascends,
and the graph's free energy has one sign. You can express a GAN's *wiring* here — and it is a
DAG, since every factor is unidirectional — but not its objective. `local_free_energy` on a
`RatioFactor` reports a factor-belief energy contribution, which generally differs from
the generator's expectation under its original sample distribution.

There is no training of either network. Mycelium pools RatioFactor's likelihood potential
with particles; explicit `reweight` applies the same weight update. Applying the same
ratio twice squares it, as expected when evidence is multiplied twice.

## API

```@index
Modules = [Adversarial]
```

```@autodocs
Modules = [Adversarial]
```

## Point games and conformance

Generator has zero game free energy/entropy for a Dirac latent observation `y`.
NoiseSource has the same pair for a fixed Dirac draw `π` with `nsamples=1`, without sampling again.
Ratio has game energy `-logratio` and zero entropy for a Dirac prior. Its non-Dirac game
priors remain unsupported: no exact reweighted density/entropy law is supplied.
Particle clouds have no Lebesgue density (LP-006).

`AffineGenerator(A,b)` supplies an exact Gaussian forward game with a square nonsingular
matrix and identity input adapter: `y` is the latent Gaussian and `π` is absent/trivial.
`pushforward_entropy` computes `H(source)+log|det A|`; `entropy` returns that value and
`free_energy` its negative. Output conditioning, singular/rectangular maps and improper
sources throw. The built-in standard-normal NoiseSource has exact unconditional entropy
for `π=TrivialBelief()` and no observation, without drawing particles. Custom samplers
supply no density contract.

For a caller-certified nonlinear bijection, `InvertibleGenerator(net,logabsdet)` and
`estimate_pushforward_entropy(...;rng,nsamples)` expose an `EntropyEstimate` with a sampling
standard error. This estimate is separate from exact game entropy. Finite supported/Dirac
laws do not acquire a Lebesgue entropy from a finite Jacobian expectation.

Ratio emits a `LogPotentialBelief` for `logratio(x)` without reading the cavity or target
input. `invert` and explicit `reweight` return the posterior. Mycelium pools the potential
with particles by importance weighting. The graph energy contribution averages `-logratio`
under that reweighted cavity; it does not include a continuous particle entropy or generally
equal `KL(q || p)`.

`test/conformance.jl` checks the point games, the two-particle Gaussian ratio closed form,
target exclusion, and a particle prior with two RatioFactors on one variable. Both insertion
orders give weights proportional to the prior times each ratio once (LP-016).
