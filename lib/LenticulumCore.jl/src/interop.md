#implementation

> Sources: code: `interop.jl`, `ext/LenticulumCoreExponentialFamilyExt.jl`; Bagaev, Podusenko & de Vries, *RxInfer: A Julia package for reactive real-time Bayesian inference*, JOSS 2023
>
> Bibliography: [[Bibliography#^bagaev2023rxinfer|Bagaev et al. 2023]]
>
> Theory: [[RxInfer as a Backend]] · [[Belief Algebra]] · [[Beliefs]]

Implements: step 1 of [[RxInfer as a Backend]] §4. Every belief converts to the distribution
type RxInfer.jl passes as a message, and back.

## 1. The functions

`as_distribution(b)` and `as_belief(d)`. The stubs live in `interop.jl`; the methods are in a
package extension that loads when ExponentialFamily.jl, BayesBase.jl and Distributions.jl are
loaded, which `using RxInfer` does. `LenticulumCore` itself gains no dependency.

| belief | distribution | exact? |
|---|---|---|
| `GaussianBelief(η, Λ)` | `MvNormalWeightedMeanPrecision(η, Λ)` (`univariate = true`: `NormalWeightedMeanPrecision`) | yes: both canonical |
| `DiracBelief` | `PointMass` | yes |
| `CategoricalBelief` | `Categorical` | probabilities yes; labels are dropped |
| `MixtureBelief` | `MixtureModel` | yes, componentwise |
| `SampleBelief` | `SampleList` | yes; weights normalised |
| `TrivialBelief` | — | throws |

In the other direction, **any normal** (Distributions' `Normal` and `MvNormal`, every
parametrisation in ExponentialFamily) goes through BayesBase's `weightedmean_precision`, so it
lands in canonical form without inverting a covariance. `Bernoulli` becomes a two-state
`CategoricalBelief`.

## 2. Design decisions

- **Canonical to canonical.** RxInfer's `MvNormalWeightedMeanPrecision` is exactly $(\eta, \Lambda)$.
  Converting through the moment form would fail for the improper Gaussians that likelihood
  messages are, and would cost an inversion each way.
- **An extension on ExponentialFamily, not on RxInfer.** The message types live in
  ExponentialFamily and BayesBase; RxInfer is the engine. Depending on the smaller packages
  means the conversion works for anyone using those types, RxInfer or not.
- **Functions, not `convert` methods.** The conversions are not always lossless (labels,
  `TrivialBelief`), and `convert` is called implicitly.

## 3. How it is tested

In the top-level `Lenticulum` test suite ("interop"), because the checks need `Mycelium`'s
`combine`:

- round trips of every type, an improper Gaussian included;
- Lenticulum's pooling (`combine`) against BayesBase's `prod` for Gaussians and categoricals;
- the posterior of a one-step conjugate model, $x \sim \mathcal N(0, 4)$, $y \mid x \sim \mathcal N(2x + 1, 0.5)$,
  $y = 3$, computed by RxInfer's `infer` and by a `GaussianFactor` message pooled with the
  prior: identical.

## 4. What it does not do

BayesBase's `prod` of a mixture and a Gaussian is a lazy `ProductOf`, while `combine` computes
the product mixture in closed form; the conversion makes the two interchangeable but does not
add rules to RxInfer. Step 3 of [[RxInfer as a Backend]] §4, a learned relation as an RxInfer
node, builds on this and is not done.

Related: [[gaussian_belief]], [[open_model]], [[messages]]
