# Loaded last: the main belief tests also run without the distribution extension.
using ExponentialFamily: MvNormalWeightedMeanPrecision, NormalWeightedMeanPrecision,
    MvNormalMeanCovariance, NormalMeanVariance
using BayesBase: BayesBase, PointMass, SampleList, GenericProd
using Distributions: Distributions, Categorical, Bernoulli, MixtureModel

@testset "P36 interop: beliefs and independent distribution representations" begin
    g = Gaussian([1.0, -2.0], [2.0 0.5; 0.5 1.0])
    d = as_distribution(g)
    @test d isa MvNormalWeightedMeanPrecision
    @test BayesBase.weightedmean_precision(d)[1] == g.η && BayesBase.weightedmean_precision(d)[2] == g.Λ
    @test Distributions.mean(d) ≈ belief_mean(g) && Distributions.cov(d) ≈ belief_cov(g)
    @test as_belief(d).η ≈ g.η && as_belief(d).Λ ≈ g.Λ                         # exact round trip
    @test as_distribution(Gaussian(0.5, 2.0); univariate = true) isa NormalWeightedMeanPrecision
    @test belief_cov(as_belief(NormalMeanVariance(0.5, 2.0))) ≈ fill(2.0, 1, 1)  # any parametrisation
    @test belief_mean(as_belief(MvNormalMeanCovariance([1.0, 0.0], [1.0 0.0; 0.0 4.0]))) ≈ [1.0, 0.0]
    @test belief_mean(as_belief(Distributions.MvNormal([1.0, 2.0], [1.0 0.0; 0.0 1.0]))) ≈ [1.0, 2.0]
    # an improper Gaussian (a likelihood on one direction) converts too
    lik = GaussianBelief([1.0, 0.0], [1.0 0.0; 0.0 0.0])
    @test as_belief(as_distribution(lik)).Λ == lik.Λ
    # pooling agrees with RxInfer's product
    h = Gaussian([0.0, 1.0], [1.0 0.0; 0.0 3.0])
    p = prod(GenericProd(), as_distribution(g), as_distribution(h))
    @test as_belief(p).η ≈ (g.η + h.η) && as_belief(p).Λ ≈ (g.Λ + h.Λ)
    # the discrete and particle beliefs
    c = LenticulumCore.CategoricalBelief([0.2, 0.3, 0.5])
    @test as_distribution(c) isa Categorical && Distributions.probs(as_distribution(c)) ≈ [0.2, 0.3, 0.5]
    @test probabilities(as_belief(as_distribution(c))) ≈ [0.2, 0.3, 0.5]
    @test probabilities(as_belief(Bernoulli(0.3))) ≈ [0.7, 0.3]
    c2 = LenticulumCore.CategoricalBelief([0.5, 0.25, 0.25])
    @test ([0.2,0.3,0.5] .* [0.5,0.25,0.25]) / sum([0.2,0.3,0.5] .* [0.5,0.25,0.25]) ≈ Distributions.probs(prod(GenericProd(), as_distribution(c), as_distribution(c2)))
    x = as_distribution(DiracBelief([1.0, 2.0]))
    @test x isa PointMass && as_belief(x).value == [1.0, 2.0]
    m = LenticulumCore.MixtureBelief([Gaussian(-1.0, 0.5), Gaussian(2.0, 1.0)], [0.25, 0.75])
    md = as_distribution(m)
    @test md isa MixtureModel && Distributions.probs(md) ≈ [0.25, 0.75]
    @test mixture_weights(as_belief(md)) ≈ [0.25, 0.75]
    @test belief_mean(as_belief(md).components[2]) ≈ [2.0]
    s = LenticulumCore.SampleBelief([[1.0], [2.0], [4.0]], [1.0, 1.0, 2.0])
    sl = as_distribution(s)
    @test sl isa SampleList && Distributions.mean(sl) ≈ [2.75]
    @test as_belief(sl).weights ≈ [0.25, 0.25, 0.5]
    @test_throws ArgumentError as_distribution(TrivialBelief())

    @test_throws ArgumentError as_distribution(g; univariate = true)
    @test belief_mean(as_belief(Distributions.Normal(1.2, 0.7))) ≈ [1.2]
    @test belief_cov(as_belief(Distributions.Normal(1.2, 0.7))) ≈ fill(0.49, 1, 1)
    for samples in ([1.0, 3.0, 5.0], [[1.0, 2.0], [3.0, 4.0]], [1.0 3.0; 2.0 4.0])
        n = samples isa AbstractMatrix ? size(samples, 2) : length(samples)
        for weights in (nothing, collect(1.0:n))
            b = SampleBelief(samples, weights)
            d = as_distribution(b)
            r = as_belief(d)
            expected = samples isa AbstractMatrix ? collect(eachcol(samples)) : samples
            @test r.samples == expected
            @test r.weights ≈ (weights === nothing ? fill(1/n, n) : weights / sum(weights))
        end
    end
    @test_throws ArgumentError as_distribution(SampleBelief([1.0, 2.0], [0.0, 0.0]))
    # Construct the external mixture independently, rather than only testing a round trip.
    r = as_belief(MixtureModel([Distributions.Normal(-2, 0.5), Distributions.Normal(3, 2)], [0.4, 0.6]))
    @test mixture_weights(r) ≈ [0.4, 0.6]
    @test belief_mean(r.components[1]) ≈ [-2]
    @test belief_cov(r.components[2]) ≈ fill(4.0, 1, 1)
end
