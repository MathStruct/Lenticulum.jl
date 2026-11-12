# P18: `belief_mean` and `belief_cov` of a `SampleBelief` are LenticulumCore's (they were defined
# in Adversarial, type piracy). Oracle: the weighted sums written out, and Statistics' mean and
# uncorrected covariance for uniform weights.
using LenticulumCore, LinearAlgebra, Random, Statistics, Test

@testset "P18: sample moments in LenticulumCore" begin
    rng = Xoshiro(2618)
    for trial in 1:40
        d, n = rand(rng, 1:4), rand(rng, 1:30)
        X = randn(rng, d, n)
        w = rand(rng, n) .* (rand(rng, n) .> 0.3); w[rand(rng, 1:n)] = rand(rng) + 0.1   # some zero, not all
        wn = w ./ sum(w)
        m = X * wn
        C = sum(wn[i] .* (X[:, i] .- m) * (X[:, i] .- m)' for i in 1:n)
        for b in (SampleBelief(X, w), SampleBelief([X[:, i] for i in 1:n], w))
            @test belief_mean(b) ≈ m atol = 1e-12
            @test belief_cov(b) ≈ C atol = 1e-12
        end
        u = SampleBelief(X)                                       # no weights: uniform
        @test belief_mean(u) ≈ vec(mean(X; dims = 2)) atol = 1e-12
        @test belief_cov(u) ≈ cov(X; dims = 2, corrected = false) atol = 1e-12
        # weights are relative: scaling them, also by 1e300, changes nothing
        @test belief_mean(SampleBelief(X, w .* (1e300 / maximum(w)))) ≈ m atol = 1e-12
    end
    # scalar samples give vectors of length one
    s = SampleBelief([1.0, 2.0, 4.0], [1.0, 1.0, 2.0])
    @test belief_mean(s) ≈ [2.75]
    @test belief_cov(s) ≈ fill((1.75^2 + 0.75^2 + 2 * 1.25^2) / 4, 1, 1)
    @test belief_mean(SampleBelief([3, 5])) == [4.0]              # integer samples are floated
    # a single sample: its value, zero covariance
    @test belief_mean(SampleBelief(reshape([1.0, 2.0], 2, 1))) == [1.0, 2.0]
    @test belief_cov(SampleBelief(reshape([1.0, 2.0], 2, 1))) == zeros(2, 2)
end
