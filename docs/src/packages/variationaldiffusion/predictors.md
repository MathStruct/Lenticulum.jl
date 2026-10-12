# [Predictors and backends](@id vd-predictors)

```@meta
CurrentModule = VariationalDiffusion
```

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

## Closed-form and kernel predictors

`GaussianMixtureEps` is the exact noise predictor of a Gaussian mixture: every derivative the
solvers need is in closed form, which makes it the test oracle. `kde_predictor` turns data
points into such a mixture (a kernel density estimate), and `OnlineKDE` keeps one that learns
from a stream and forgets.

## API

### The predictor

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["predictor.jl"]
```

### Closed-form predictors

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["analytic.jl"]
```

### Kernel predictors

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["kernel.jl"]
```

### Energy networks

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["energy_network.jl"]
```
