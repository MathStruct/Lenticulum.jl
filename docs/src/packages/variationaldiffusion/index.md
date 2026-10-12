# [VariationalDiffusion](@id variationaldiffusion)

```@meta
CurrentModule = VariationalDiffusion
```

A **diffusion model as a factor**: wrap a trained noise predictor and use it as a prior over
the joint space of a factor, queried by root finding (the implicit learner), by RED-Diff's
proximal solve, or by ProxDM's learned prox.

This package *consumes* a trained ``\varepsilon_\theta`` (it offers training losses, not a
training loop) and has no automatic-differentiation dependency: derivatives come from closed
forms, finite differences, or an AD backend the user chooses through a package extension.

| page | covers |
|---|---|
| this page | the pieces, the schedule, choosing λ, known gaps |
| [Predictors and backends](@ref vd-predictors) | `NoisePredictor`, closed-form and kernel predictors, AD and Reactant backends, energy networks |
| [Implicit inference](@ref vd-implicit) | the solver, all answers and their uncertainty, many free coordinates, sampling, gradients through inference, products |
| [Factors](@ref vd-factors) | `DiffusionFactor` with RED-Diff or the implicit solver as its inversion |
| [ProxDM](@ref vd-proxdm) | proximal diffusion models: the prox, the sampler, proximal inference |

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

Details, backends and energy networks: [Predictors and backends](@ref vd-predictors).


`NoisePredictor` wraps any `LuxCore.AbstractLuxLayer` and derives two things from it:

| function | is | note |
|---|---|---|
| `epsilon` | the network itself | the only place it is evaluated |
| `score` | ``-\varepsilon_\theta/\sigma_t`` | ``\approx \nabla_x \log p_t(x)`` |
| `denoise` | ``(x - \sigma_t\varepsilon_\theta)/\alpha_t`` | Tweedie — the MMSE denoiser |

Parameters are the wrapped model's, untouched: `LuxCore.setup(rng, pred) == LuxCore.setup(rng, unet)`.

### The factor

Details: [Factors](@ref vd-factors).


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

- ProxDM is not yet a `DiffusionFactor` inversion (`prox_infer` and its adjoint `prox_pullback` exist).
- RED-Diff's inversion returns a point, so uncertainty does not propagate; the implicit
  solver's `message = :gaussian` does send a Laplace Gaussian.
- `implicit_sample` takes hard clamps only; soft evidence would need likelihood guidance.
- The `KrylovSolve` stability test and log-determinant are estimates once more coordinates are
  free than Lanczos steps and probes.

## Index

```@index
Modules = [VariationalDiffusion]
```

## API

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["VariationalDiffusion.jl", "schedule.jl"]
```
