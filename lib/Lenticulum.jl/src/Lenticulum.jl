"""
    Lenticulum

Implicit machine learning on factor graphs: learning **relations**
``R_\\theta \\subseteq X_1\\times\\cdots\\times X_n`` rather than functions.

This package is the user-facing layer, mirroring `Lux.jl`. It sits on

- `LenticulumCore` — what a **factor** is (a parameterized statistical game);
- `Mycelium` — how factors are **wired** and in what **order** they talk.

## What is here so far

The **linear-Gaussian factor** and its beliefs: the first factor in the project that
implements the entire interface with nothing stubbed — a real open model, an *exact* Bayesian
inversion, a genuine vector energy and entropy, bidirectional polarity, and learnable
parameters.

Its purpose is to be a **test oracle**. Every quantity it produces has a closed form, so the
framework's claims can be checked against arithmetic instead of against itself. Two of them
turned out to be wrong; see `The Gaussian Factor.md` §"What this caught".

- `beliefs.jl` — projections onto [`GaussianBelief`](@ref) (`moment_match`; `reduce_mixture` is re-exported from Mycelium)
  and `kl_divergence`. The type itself is in `LenticulumCore` and its message rules (pooling is
  **addition** of canonical parameters) in `Mycelium`; both are re-exported here.
- `gaussian.jl` — [`GaussianFactor`](@ref) and [`GaussianPrior`](@ref).
- `constraint.jl` — [`LinearConstraintFactor`](@ref), the **acausal** n-ary sibling of
  `GaussianFactor`: one equation ``0 = \\sum_i A_i x_i - c + \\varepsilon`` over any number of
  channels, none of them distinguished. This is ModelingToolkit's `0 ~ ...` equation as a
  statistical game; see `ModelingToolkit as an Acausal Relation.md`.
- `nonlinear.jl` — [`NonlinearFactor`](@ref): a given ``f`` with Gaussian noise, by linearisation
  or the unscented transform.
- `likelihood.jl` — [`LikelihoodFactor`](@ref): a Gaussian latent observed through a
  non-Gaussian likelihood (counts, ordered categories), with Laplace and EP messages and a
  predictive over discrete outcomes; see `likelihood.md`.
- `switching.jl` — [`SwitchingConstraintFactor`](@ref): a linear constraint whose right-hand
  side is chosen by a discrete outcome, closed form in every direction; see `switching.md`.
- `coupling.jl` — [`coupling_weights`](@ref): each factor's coupling strength, as default weights
  for `spanning_tree_schedule`; see `coupling.md`.
"""
module Lenticulum

using LinearAlgebra: LinearAlgebra, Diagonal, I, Symmetric, cholesky, isposdef, logdet, tr, diag, isdiag, issymmetric, eigen, eigmin
using Random: Random, AbstractRNG
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium, reduce_mixture          # re-exported (moved to Mycelium, P17)
using MyceliumLearning: MyceliumLearning
using SpecialFunctions: SpecialFunctions
using LenticulumCore: GaussianBelief, Gaussian, uninformative, belief_mean, belief_cov, isproper,
    logpartition, dimension, _chol, _spd_inv, _improper_check, ExtendedGaussian

include("beliefs.jl")
include("noise.jl")
include("gaussian.jl")
include("exposure.jl")
include("composition.jl")
include("nonlinear.jl")
include("constraint.jl")
include("hard_constraint.jl")
include("likelihood.jl")
include("switching.jl")
include("regime.jl")
include("m_steps.jl")
include("coupling.jl")
include("split.jl")

# The factors whose belief-propagation messages are exact for point, categorical and Gaussian
# inputs (also flat ones), each tested against a closed form (`test/gaussian_oracle.jl`,
# `test/hard_constraint.jl`, the switching testsets): a tree of them, one sweep, is exact unless a
# mixture reaches a factor that collapses it (Mycelium's `exact_messages`, P13). Not declared:
# `NonlinearFactor` (a local linear model), `LikelihoodFactor` (non-conjugate), `RegimeFactor`
# (exact only inside a cluster node), and the exposed and overlay wrappers.
for T in (GaussianFactor, GaussianPrior, LinearConstraintFactor, HardConstraintFactor, SwitchingConstraintFactor)
    @eval Mycelium.exact_messages(::$T) = true
end

export GaussianBelief, Gaussian, uninformative
export belief_mean, belief_cov, isproper, logpartition, kl_divergence
export moment_match, reduce_mixture
export GaussianFactor, GaussianPrior, LinearGaussianModel
export WishartPrecisionPrior, LogNormalVariancePrior, noise_covariance
export NonlinearFactor, NonlinearGaussianModel, Linearization, Unscented, linearize
export LinearConstraintFactor, LinearConstraintModel
export HardConstraintFactor, ExtendedGaussian
export SplitFactor
export residual, residual_statistics
export SwitchingConstraintFactor, SwitchingConstraintModel, switch_statistics, switch_overlaps, partial_sums
export RegimeFactor, regime_covariances
export FrozenGaussianBelief, FrozenSwitchingBeliefs
export LikelihoodFactor, LikelihoodModel, Laplace, ExpectationPropagation, predictive, log_evidence
export AbstractLikelihood, PoissonLikelihood, BivariatePoissonLikelihood, OrdinalProbitLikelihood
export coupling_precision, coupling_weights
export npredictors, obsdim, loglikelihood, loglikelihood_derivatives, outcomes, outcome_logprobs

end # module
