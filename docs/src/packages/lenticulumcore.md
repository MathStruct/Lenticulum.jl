# [LenticulumCore](@id lenticulumcore)

```@meta
CurrentModule = LenticulumCore
```

Defines **what a factor is**. No graphs, no message passing, no concrete factors — just the
interface every factor implements and the types it traffics in.

This is the analogue of `LuxCore`: a small package that other things agree on. If you are
writing a new factor, this is the interface you implement.

## What is in here

| group | types |
|---|---|
| the factor interface | `AbstractLenticulumFactor`, `channels`, `supports_polarity`, `assemble` |
| channels and roles | `Channel`, `Polarity`, `Observed`, `Unobserved`, `Latent` |
| beliefs | `AbstractBelief`, `DiracBelief`, `SampleBelief`, `TrivialBelief` |
| the forward half | `AbstractOpenModel`, `forward`, `pushforward`, `logdensity` |
| the backward half | `BayesianLens`, `invert`, and the inversion kinds below |
| energies | `AbstractEnergySpace`, `GradedEnergy`, scalarisations |

## Writing a factor

A factor must provide `LuxCore.initialparameters` / `initialstates` (exactly as a Lux layer
does), plus:

```julia
LenticulumCore.channels(f)               # the named ports
LenticulumCore.supports_polarity(f, p)   # can it be run this way?
LenticulumCore.supported_polarities(f)   # ...and list the ways
LenticulumCore.assemble(f, p, ps, st)    # build the lens for one polarity
LenticulumCore.energy(f, x, a, y, ps, st)
```

`assemble` is the operation with no Lux counterpart. A Lux layer *is* a forward/backward pair;
a factor only *becomes* one once you have chosen a direction.

## The inversion kinds

`assemble` returns a `BayesianLens` pairing a model with an inversion, and the inversion kind
says how the backward direction is computed:

| kind | means | used by |
|---|---|---|
| `ExactInversion` | closed form | `GaussianFactor`, `NeuralODEFactor` |
| `SolverInversion` | root-finding | `DEQFactor` |
| `ProximalInversion` | a proximal / optimisation step | `DiffusionFactor` |
| `AmortisedInversion` | a separate trained network | `RatioFactor` |
| `TrivialInversion` | there is nothing to invert | `LuxFactor`, priors |

Nothing requires an inversion to be exact. An approximate one is legal; the free energy is
what records the cost.

## Known gaps

- `SampleBelief` has no `belief_logdensity`, so two of them cannot be pooled. See
  `Adversarial` for a route around this.
- `GaussianBelief` lives in `Lenticulum`, not here, so packages under `lib/` cannot produce
  one and fall back to `DiracBelief`.

## API

```@index
Modules = [LenticulumCore]
```

### Abstract types

```@autodocs
Modules = [LenticulumCore]
Pages = ["abstract_types.jl"]
```

### Channels and polarity

```@autodocs
Modules = [LenticulumCore]
Pages = ["channels.jl"]
```

### Beliefs and open models

```@autodocs
Modules = [LenticulumCore]
Pages = ["open_model.jl"]
```

### Bayesian lenses

```@autodocs
Modules = [LenticulumCore]
Pages = ["lens.jl"]
```

### Energies and scalarisations

```@autodocs
Modules = [LenticulumCore]
Pages = ["energy.jl"]
```

### Statistical games

```@autodocs
Modules = [LenticulumCore]
Pages = ["statistical_game.jl"]
```

### Everything else

```@autodocs
Modules = [LenticulumCore]
Pages = ["LenticulumCore.jl"]
```
