# [VariationalDiffusion](@id variationaldiffusion)

```@meta
CurrentModule = VariationalDiffusion
```

A **diffusion model as a factor**: wrap a trained noise predictor and use it as a prior, with
inference by RED-Diff's proximal solve.

This package *consumes* a trained ``\varepsilon_\theta``; it does not train one. There is no
automatic-differentiation dependency, because RED-Diff's stop-gradient means the network is
only ever evaluated forwards.

## The pieces

```julia
sched = VPSDE()                                   # the forward process
pred  = NoisePredictor(my_unet, sched)            # ε_θ(x, t) — any Lux model
prox  = REDDiff(λ = 0.25, steps = 200)            # the inversion
f     = DiffusionFactor((obs = 4, hidden = 4), pred; prox)
```

### The schedule

`VPSDE` is the variance-preserving SDE of Song et al.: ``x_t = \alpha_t x_0 + \sigma_t
\varepsilon`` with ``\alpha_t^2 + \sigma_t^2 = 1``. Query it with `alpha`, `sigma`, `snr`,
`perturb`, `sample_time`.

### The predictor

`NoisePredictor` wraps any `LuxCore.AbstractLuxLayer` and derives two things from it:

| function | is | note |
|---|---|---|
| `epsilon` | the network itself | the only place it is evaluated |
| `score` | ``-\varepsilon_\theta/\sigma_t`` | ``\approx \nabla_x \log p_t(x)`` |
| `denoise` | ``(x - \sigma_t\varepsilon_\theta)/\alpha_t`` | Tweedie — the MMSE denoiser |

Parameters are the wrapped model's, untouched: `LuxCore.setup(rng, pred) == LuxCore.setup(rng, unet)`.

### The factor

`DiffusionFactor` carves one vector space into named channel blocks. The `Polarity` decides
which blocks are clamped and how hard, which is exactly the selection-matrix formulation
``P = \rho_{in}P_{in} + \rho_{out}P_{out} + \rho_{latent}P_{latent}`` — inpainting, in other
words. `Observed()` defaults to ``\rho = \infty``, a hard clamp applied by projection.

The inversion runs `REDDiff` and returns a `DiracBelief`, because RED-Diff's variational
family is a point mass. That is the paper's choice, not a simplification here.

## Choosing λ

`λ` is presented in the paper as a tuned hyperparameter (they use `0.25`). For a Gaussian data
distribution it is *derivable* — there is exactly one value making the implied prior correct,
and `calibrate_lambda(schedule, v₀)` returns it.

Worth knowing what tuning `λ` actually does: it sets **how strong the learned prior is**.
A single scalar can only calibrate one eigendirection of a correlated prior, so it
over-regularises high-variance directions and under-regularises low-variance ones.

## Known gaps

- Cannot train ``\varepsilon_\theta``.
- The inversion returns a point, so uncertainty does not propagate.
- The message is a posterior rather than a likelihood, so it double-counts on a variable of
  degree greater than one.

## API

```@index
Modules = [VariationalDiffusion]
```

```@autodocs
Modules = [VariationalDiffusion]
```
