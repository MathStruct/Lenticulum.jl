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

## On a GPU

`device = (to = gpu_device(), back = cpu_device())` (MLDataDevices) runs the network and all
its derivatives on the GPU, while the solvers keep their state on the CPU. Move the parameters
yourself (`ps = gpu_device()(ps)`), and build the time features in `input` on the device of
`x`. For an energy network use `ad = SecondOrder(AutoForwardDiff(), AutoZygote())`: the
second derivatives are then taken forward over reverse on the device. At MNIST size (794
inputs, two hidden layers of 512) on an RTX 3080, a training step on 128 points takes 6 ms
against 300 ms on the CPU. A classification query with ten free coordinates takes 1.8 ms per
iteration when 1000 queries are solved together. Results agree with the CPU in Float64 to about
1e-7.

## A noise level per block

`NoisePredictor(net, VPSDE(); input, blocks = [1:784, 785:794])` lets blocks of coordinates sit
at different noise levels ([bao2023unidiffuser](@cite)). Time is then a number, one time per
block, or a matrix with a column per input column, and `input` receives the matrix. Train with
`block_times(rng, pred, B)` (independent times per block; `clean` holds one block at `tmin` in a
share of the columns, as queries hold it) and `denoising_gradient(…; weights)` (a loss weight per
coordinate, or per entry). A block held at a low level is a clean condition, a block at
level 1 is noise: one network gives the joint score, every conditional and every marginal.
The closed-form mixture and energy networks handle per-coordinate noise exactly, also on a
device.

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
