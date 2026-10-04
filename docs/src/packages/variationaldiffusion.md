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

## Small networks, any AD backend

A diffusion model here is a prior over the joint space of **one factor**: a few coordinates,
not an image. It is evaluated many times per query, so small models are the point, and the
architecture is free. An MLP with a sinusoidal time embedding is the tested baseline;
`input` adapts ``(x, t)`` to whatever the model expects.

The implicit learner needs two derivatives of the network: the input Jacobian (Newton steps,
adjoint solve) and a parameter VJP (backward pass). The backend is a field of the predictor:

```julia
using DifferentiationInterface, Zygote          # or Enzyme, ForwardDiff, Mooncake, …
pred = NoisePredictor(mlp, VPSDE(); input, ad = AutoZygote())

using Reactant                                  # XLA-compiled forward and Enzyme VJPs
pred = NoisePredictor(mlp, VPSDE(); input, ad = AutoReactant())
```

Both are package extensions; without them the Jacobian falls back to finite differences.
`examples/circle_mlp.jl` trains a 5k-parameter MLP on a circle in about 20 seconds on a CPU,
then infers both branches and checks the adjoint against finite differences.
For Enzyme on Lux use `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`.

## Energy-parametrised models

`NoisePredictor(EnergyNetwork(net), VPSDE(); input, ad = AutoZygote())`: the network outputs a
scalar energy and ``\varepsilon_\theta = \sigma_t\nabla_x E_\theta``, so the score is conservative by
construction. The implicit learner's field is then the gradient of `implicit_energy`, a real
energy for the learned relation. Train with `denoising_gradient` (the loss gradient as one
mixed second derivative, no nested AD in the loop) and any optimiser.

## Proximal diffusion models

ProxDM (Fang et al. 2025) queries the prior through ``\operatorname{prox}_{-\lambda\log p_t}``
instead of its score. `proximal` is the interface, `ProxNetwork` a learned prox
(``v - \sqrt\lambda\,\varepsilon_\theta(v; t, \lambda)``), `MixtureProx` the exact one for a Gaussian
mixture. `proxdm_sample` is the paper's sampler (PDA and PDA-hybrid), `prox_infer` the implicit
learner's query by half-quadratic splitting, `proximal_matching_loss` the training loss.

## Known gaps

- `prox_infer` has no adjoint yet, and ProxDM is not yet a `DiffusionFactor` inversion.
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
