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
| beliefs | `AbstractBelief`, `DiracBelief`, `SampleBelief`, `TrivialBelief`, `AbstractPotentialBelief`, `LogPotentialBelief` |
| the forward half | `AbstractOpenModel`, `forward`, `pushforward`, `logdensity` |
| the backward half | `BayesianLens`, `invert`, and the inversion kinds below |
| energies | `AbstractEnergySpace`, `GradedEnergy`, scalarisations |

`LogPotentialBelief(logφ)` is an improper likelihood message with log-density `logφ(x)`
up to an additive constant. It has no sampling, mean, or entropy contract. Mycelium pools
it with particles by importance weighting, and with another potential by adding log-densities.
Richer likelihoods can subtype `AbstractPotentialBelief` to expose gradients, parametric
structure, or batched evaluation.

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
| `ExactInversion` | exact model inversion, up to numerical solve tolerance | `GaussianFactor`; symmetric fixed-step `NeuralODEFactor` on matching grids |
| `SolverInversion` | approximate numerical inversion | `DEQFactor`; explicit/adaptive `NeuralODEFactor` |
| `ProximalInversion` | a proximal / optimisation step | `DiffusionFactor` |
| `AmortisedInversion` | a separate trained network | `RatioFactor` |
| `TrivialInversion` | there is nothing to invert | `LuxFactor`, priors |

Nothing requires an inversion to be exact. An approximate one is legal; the free energy is
what records the cost.

## Parameter derivatives

```julia
using LenticulumCore, DifferentiationInterface, ADTypes, ForwardDiff
ad = AutoForwardDiff()     # or load Zygote and use AutoZygote()
J = energy_jacobian(f, x, a, y, ps, st; wrt=:ps, ad)
G = gauss_newton_metric(f, x, a, y, ps, st; wrt=:ps, ad)
grad = energy_gradient(f, x, a, y, ps, st; ad)
grad_free = free_energy_gradient(f, prior, observation, ps, st; ad)
```

`J` has one column per flattened parameter (depth-first fields, column-major arrays).
Scalar gradients have the same numeric tree structure as `ps`. Omitting `ad` keeps
central finite differences. DI is a weak dependency and the user loads the backend.
`parameter_gradient(loss, ps; ad)` supports custom scalar objectives. State/data are
held fixed; no estimator for stochastic expectations is supplied. ForwardDiff,
Zygote and Enzyme are tested against closed forms; other backends depend on their support for
the chosen model. AD through finite-difference Sampson localisation is not covered.

## Gradient semantics and remaining gaps

- All four built-in gradient couplings are accepted. Gradient evaluation requires an
  implemented family/domain. Mycelium supplies finite static graph replay and terminal
  score estimates; Lenticulum supplies collapsed Gaussian composite gradients.
- `isfrozen=true` pins parameter updates while preserving input derivatives. Graph-local
  subset selection is `factor!(...; freeze)`. `islearnable=false` is the intrinsic
  optimiser-candidate trait; it also preserves input derivatives.
- `stop_gradient` has ForwardDiff/ChainRules zero-derivative rules.
  `straight_through(forward,surrogate,x)` uses the forward value and surrogate input
  Jacobian with those backends; differences of the forward function do not implement it.

- `SampleBelief` has no `belief_logdensity`, so two of them cannot be pooled. See
  `Adversarial` for a route around this.
- `GaussianBelief` lives here too (canonical form, `gaussian_belief.jl`); its message rules
  are in `Mycelium`. So far only `VariationalDiffusion`'s implicit solver emits one
  (`ImplicitProx(…; message = :gaussian)`); the equilibrium factors still return points.

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
Pages = ["open_model.jl", "gaussian_belief.jl", "interop.jl"]
```

### Bayesian lenses

```@autodocs
Modules = [LenticulumCore]
Pages = ["lens.jl"]
```

### Energies and scalarisations

`AbstractPointScalarisation` represents a field of metrics. `scalar_energy` calls
`localise(σ, f, x, a, y, ps, st)` before applying it. `SampsonNorm` localises to a
`SquaredNorm` from the pushed-forward noise covariance; factors with a residual gauge
supply `sampson_damping` transforming as `B*R*B'`. Singular covariance throws instead
of using a pseudoinverse. Graded scalarisations containing point-dependent parts must
have their parts localised in their own contexts first. `jensen_gap` requires a fixed
scalarisation; the factor form of `gauss_newton_metric` freezes the localised metric.


```@autodocs
Modules = [LenticulumCore]
Pages = ["energy.jl", "parameter_tree.jl"]
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

Lenticulum extends `ComposedFactor` for a pair of GaussianFactors by integrating out the
shared channel. It exposes the endpoint channels and exact collapsed Gaussian messages,
entropy and game/graph free energy. Other compositions throw with an exact-pushforward reason.
`TensorFactor` combines independent linear-Gaussian games: `π` and `y` must be NamedTuples
keyed exactly like `parts`; their entries describe each component's complete channel game.
Results retain the branch grading and scalarise to the sum. Correlated joint beliefs and
unsupported components throw. Tensor graph wiring/inversion remains absent.

The helpers `compose_energy`, `compose_entropy` and `compose_free_energy` retain their
value-level path-energy semantics, distinct from a collapsed Gaussian marginal game.
`FreeEnergyUpperBound` and `FreeEnergyBoundEstimate` are separate wrappers for variational
bounds and estimates; neither enters exact `free_energy` or `scalarise`. Diffusion supplies
the explicitly normalised finite-chain instance ([Factors](@ref vd-factors)).
