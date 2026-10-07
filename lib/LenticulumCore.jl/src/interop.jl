# ---------------------------------------------------------------------------
# Converting beliefs to and from the Julia distribution ecosystem.
#
# Only the function stubs live here; the methods are in package extensions, so LenticulumCore
# itself gains no dependency. With ExponentialFamily.jl loaded (as RxInfer.jl does), every
# belief type converts to the distribution RxInfer uses for it and back. See `interop.md`.
# ---------------------------------------------------------------------------

"""
    as_distribution(b::AbstractBelief; univariate = false)

The belief as a distribution of the Julia ecosystem, available when ExponentialFamily.jl is
loaded (RxInfer.jl loads it):

| belief | distribution |
|---|---|
| `GaussianBelief(η, Λ)` | `MvNormalWeightedMeanPrecision(η, Λ)`, exactly; with `univariate = true` and dimension 1, `NormalWeightedMeanPrecision` |
| `DiracBelief` | `PointMass` |
| `CategoricalBelief` | `Categorical` (labels are dropped) |
| `MixtureBelief` | `MixtureModel` of the converted components |
| `SampleBelief` | `SampleList` |

An improper Gaussian (a likelihood message) converts too: RxInfer's canonical form allows a
singular precision. `TrivialBelief` has no counterpart and throws.
"""
function as_distribution end

"""
    as_belief(d) -> AbstractBelief

The inverse of [`as_distribution`](@ref), available when ExponentialFamily.jl is loaded. Any
univariate or multivariate normal (from Distributions.jl or ExponentialFamily.jl, in any
parametrisation) becomes a `GaussianBelief` through its weighted mean and precision; a
`PointMass` a `DiracBelief`; `Categorical` and `Bernoulli` a `CategoricalBelief`; a
`MixtureModel` a `MixtureBelief`; a `SampleList` a `SampleBelief`.
"""
function as_belief end
