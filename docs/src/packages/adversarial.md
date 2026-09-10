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
`RatioFactor` computes the number both players care about; nothing acts on it.

Also: no training of either network, and `reweight` is not idempotent (applying the same ratio
twice squares the weights), so it is not wired into `Mycelium.combine`.

## API

```@index
Modules = [Adversarial]
```

```@autodocs
Modules = [Adversarial]
```
