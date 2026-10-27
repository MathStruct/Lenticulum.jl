"""
    LenticulumCore

Core abstractions for Lenticulum.jl: **factors as parameterized statistical games**.

`LuxCore.jl` is to `Lux.jl` as `LenticulumCore.jl` is to `Lenticulum.jl`, and the
parameter/state interface is deliberately identical. The difference is what a node *is*:

| | Lux.jl | Lenticulum.jl |
|---|---|---|
| node | parametric lens (Cruttwell et al., Def. 2.5) | parameterized statistical game (AutoBayes, Def. 27) |
| category | `Para(Lens(C))` | `Para(StatGame)` |
| direction | fixed at construction | chosen per call, via a `Polarity` |
| wiring | directed **acyclic** graph | any weakly-connected digraph |
| backward pass | reverse derivative | Bayesian inversion + free-energy accumulation |

A factor carries four pieces (AutoBayes Definition 20): a forward kernel `c`, an inversion
`c'`, an energy and an entropy. Lenticulum keeps the energy and entropy **vector-valued**,
with a separate scalarisation, so that Jacobians — and hence the Gauss–Newton and Fisher
metrics, and the implicit function theorem — survive composition. See
`Scalar and Multivariate Energy.md`.

Concept notes live in `vault/`; per-file implementation notes sit next to each source
file.
"""
module LenticulumCore

using DispatchDoctor: @stable
using Random: Random, AbstractRNG
using LuxCore: LuxCore
using LinearAlgebra: LinearAlgebra, Diagonal, Hermitian, Symmetric, cholesky, dot, eigvals, isposdef, issymmetric, issuccess, logdet

include("abstract_types.jl")
include("channels.jl")
include("parameter_tree.jl")
include("energy.jl")
include("open_model.jl")
include("gaussian_belief.jl")
include("interop.jl")
include("lens.jl")
include("statistical_game.jl")

# --- Types -----------------------------------------------------------------
export AbstractLenticulumFactor,
    AbstractLenticulumContainerFactor,
    AbstractLenticulumWrapperFactor,
    AbstractOpenModel,
    AbstractBelief,
    AbstractInversion,
    AbstractBayesianLens,
    AbstractEnergySpace,
    AbstractScalarisation,
    AbstractChannelPolarity,
    AbstractGradientCoupling

# --- Polarity (note: `Channel` is intentionally NOT exported; it shadows Base.Channel) ---
export Observed, Unobserved, Latent, Polarity
export channels, channelname, channelspace, supports_polarity, supported_polarities, assemble
export polarities, hasnopolarity, isunidirectional, islearnable, isfrozen
export default_precision
export select, observed_channels, unobserved_channels, latent_channels, channel_precision, ispartition

# --- Energy ----------------------------------------------------------------
export ScalarEnergySpace, EuclideanEnergySpace, GradedEnergySpace, GradedEnergy
export AbstractPointScalarisation, IdentityScalarisation, WeightedSum, SquaredNorm, GradedScalarisation, SampsonNorm
export localise, sampson_damping
export energyspace, scalarisation, pointwise_energyspace, pointwise_scalarisation, scalarise, islinear, jensen_gap, oplus, ⊕, dimension
export energy, entropy, scalar_energy, energy_jacobian, gauss_newton_metric
export energy_gradient, free_energy_gradient, parameter_gradient

# --- Models and lenses -----------------------------------------------------
export OpenModelResult, DiracBelief, SampleBelief, TrivialBelief
export AbstractPotentialBelief, LogPotentialBelief
export CategoricalBelief, MixtureBelief, probabilities, bernoulli, mixture_weights
export GaussianBelief, Gaussian, uninformative, isuninformative, belief_mean, belief_cov, isproper, logpartition
export as_distribution, as_belief
export forward, logdensity, pushforward, isexact, ispure
export latentspace, observedspace, unobservedspace
export BayesianLens, ComposedLens, TensorLens, invert, likelihood_message, compose
export ExactInversion, AmortisedInversion, SolverInversion, ProximalInversion, TrivialInversion

# --- Games -----------------------------------------------------------------
export ComposedFactor, TensorFactor, setup
export free_energy, scalar_free_energy
export FreeEnergyUpperBound, FreeEnergyBoundEstimate, free_energy_upper_bound, estimate_free_energy_upper_bound
export compose_energy, compose_entropy, compose_free_energy, chain_rule_defect
export DiagonalCoupling, PathwiseCoupling, ScoreFunctionCoupling, ExactCoupling

end # module
