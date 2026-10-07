# Beliefs ↔ the distributions of BayesBase / ExponentialFamily / Distributions, the types
# RxInfer.jl passes as messages. See `interop.md` in LenticulumCore's src/.
module LenticulumCoreExponentialFamilyExt

using LenticulumCore: LenticulumCore, GaussianBelief, DiracBelief, CategoricalBelief, MixtureBelief,
    SampleBelief, TrivialBelief, probabilities, mixture_weights, bernoulli
using ExponentialFamily: ExponentialFamily, MvNormalWeightedMeanPrecision, NormalWeightedMeanPrecision,
    UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily
using BayesBase: BayesBase, PointMass, SampleList, weightedmean_precision
using Distributions: Distributions, Normal, MvNormal, Categorical, Bernoulli, MixtureModel

# --- to distributions --------------------------------------------------------------------

function LenticulumCore.as_distribution(b::GaussianBelief; univariate::Bool = false)
    if univariate
        length(b.η) == 1 || throw(ArgumentError("univariate = true needs a one-dimensional belief"))
        return NormalWeightedMeanPrecision(b.η[1], b.Λ[1, 1])
    end
    return MvNormalWeightedMeanPrecision(collect(b.η), Matrix(b.Λ))
end
LenticulumCore.as_distribution(b::DiracBelief; kw...) = PointMass(b.value)
LenticulumCore.as_distribution(b::CategoricalBelief; kw...) = Categorical(probabilities(b))
LenticulumCore.as_distribution(b::MixtureBelief; kw...) =
    MixtureModel([LenticulumCore.as_distribution(c; kw...) for c in b.components], mixture_weights(b))
function LenticulumCore.as_distribution(b::SampleBelief; kw...)
    xs = b.samples isa AbstractMatrix ? [collect(c) for c in eachcol(b.samples)] : collect(b.samples)
    w = b.weights === nothing ? fill(1 / length(xs), length(xs)) : collect(float.(b.weights)) ./ sum(b.weights)
    return SampleList(xs, w)
end
LenticulumCore.as_distribution(::TrivialBelief; kw...) = throw(ArgumentError(
    "a TrivialBelief has no distribution counterpart; leave the message out instead"))

# --- from distributions ------------------------------------------------------------------

const _Univariate = Union{Normal,UnivariateNormalDistributionsFamily}
const _Multivariate = Union{MvNormal,MultivariateNormalDistributionsFamily}

function LenticulumCore.as_belief(d::_Univariate)
    ξ, w = weightedmean_precision(d)
    return GaussianBelief([float(ξ)], fill(float(w), 1, 1))
end
function LenticulumCore.as_belief(d::_Multivariate)
    ξ, W = weightedmean_precision(d)
    return GaussianBelief(collect(float.(ξ)), Matrix{Float64}(W))
end
LenticulumCore.as_belief(d::PointMass) = DiracBelief(BayesBase.getpointmass(d))
LenticulumCore.as_belief(d::Categorical) = CategoricalBelief(Distributions.probs(d))
LenticulumCore.as_belief(d::Bernoulli) = bernoulli(Distributions.succprob(d))
LenticulumCore.as_belief(d::MixtureModel) =
    MixtureBelief([LenticulumCore.as_belief(c) for c in Distributions.components(d)], Distributions.probs(d))
LenticulumCore.as_belief(d::SampleList) =
    SampleBelief([collect(x) for x in BayesBase.get_samples(d)], collect(BayesBase.get_weights(d)))

end # module
