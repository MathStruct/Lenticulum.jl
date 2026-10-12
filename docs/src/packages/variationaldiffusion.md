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

## Many free coordinates

Every solver step, the Laplace covariance and the adjoint need only the Jacobian block on the
**free** coordinates, and that is all they compute. With `ImplicitDiffusion(pred, nodes;
batched = true)` all nodes of one noise level go through the network in one call. When most of
the state is free (generating an image), pass `linsolve = KrylovSolve()` to `implicit_infer`,
`implicit_pullback`, `laplace_belief`, `implicit_mixture` or `ImplicitProx`: Newton steps and
the adjoint are then solved by GMRES on Jacobian-vector products (`prior_jvp`) and
vector-Jacobian products (`prior_vjp`), stability is tested by Lanczos, and the Laplace
covariance is computed only on the coordinates asked for. On a 794-dimensional MLP with 784
free coordinates, `implicit_infer` takes about 1 s; one dense 784 × 784 block takes 33 s.

## Samples, not only modes

The root finders return modes. `implicit_sample(m, z₀, ρ, ps, st; n, steps, resample)` draws
`n` samples of the free coordinates given the hard-clamped ones (`ρ = Inf`) with the reverse
diffusion, replacing the clamped coordinates at every step; `resample > 0` adds RePaint's
harmonisation, which a strongly coupled clamp needs.

## Differentiating through inference

With ChainRulesCore loaded (Zygote loads it), `implicit_infer` has an `rrule` whose pullback is
`implicit_pullback`: `Zygote.gradient(ps -> loss(first(implicit_infer(m, z₀, ρ, ps, st)).z), ps)`
works without unrolling the solver, and gives gradients for `z₀` and `ρ` too.

## Proximal diffusion models

ProxDM (Fang et al. 2025) queries the prior through ``\operatorname{prox}_{-\lambda\log p_t}``
instead of its score. `proximal` is the interface, `ProxNetwork` a learned prox
(``v - \sqrt\lambda\,\varepsilon_\theta(v; t, \lambda)``), `MixtureProx` the exact one for a Gaussian
mixture. `proxdm_sample` is the paper's sampler (PDA and PDA-hybrid), `prox_infer` the implicit
learner's query by half-quadratic splitting, `proximal_matching_loss` the training loss.

## Known gaps

- `prox_infer` has no adjoint yet, and ProxDM is not yet a `DiffusionFactor` inversion.
- RED-Diff's inversion returns a point, so uncertainty does not propagate; the implicit
  solver's `message = :gaussian` does send a Laplace Gaussian.
- `implicit_sample` takes hard clamps only; soft evidence would need likelihood guidance.
- The `KrylovSolve` stability test and log-determinant are estimates once more coordinates are
  free than Lanczos steps and probes.

## API

```@index
Modules = [VariationalDiffusion]
```

```@autodocs
Modules = [VariationalDiffusion]
```
