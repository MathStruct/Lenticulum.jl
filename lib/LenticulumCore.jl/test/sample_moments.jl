# P18: `belief_mean` and `belief_cov` of a `SampleBelief` are LenticulumCore's (they were defined
# in AdversarialFactors, type piracy). Oracle: the weighted sums written out, and Statistics' mean and
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

# GH-21: the sample moments are pure reductions — no Float64 accumulator, no mutation — so they
# differentiate under ForwardDiff and Zygote and keep the samples' scalar type. Oracles: the
# weighted sums written out, and the derivative of the closed form.
using ForwardDiff, Zygote

@testset "GH-21: sample moments differentiate and preserve the scalar type" begin
    # matrix columns, unweighted: ForwardDiff through the mean and the covariance
    d = ForwardDiff.derivative(t -> sum(belief_mean(SampleBelief(reshape([t, 2t], 1, 2)))), 1.0)
    @test d ≈ 1.5
    X = [1.0 2.0 4.0; 0.5 -1.0 2.0]
    m = X * [1 / 3, 1 / 3, 1 / 3]
    C = sum((1 / 3) .* (X[:, i] .- m) * (X[:, i] .- m)' for i in 1:3)
    g = ForwardDiff.gradient(x -> tr(belief_cov(SampleBelief(x))), X)
    @test g ≈ 2 .* (X .- m) ./ 3 rtol = 1e-10            # d tr(C)/d X[:,i] = 2 w_i (X[:,i] - m)
    # vector of vectors, weighted: Zygote through mean and covariance against the closed form
    pts = [[1.0, 0.5], [2.0, -1.0], [4.0, 2.0]]
    w = [0.2, 0.3, 0.5]
    b = SampleBelief(pts, w)
    wn = w ./ sum(w)
    m0 = sum(wn[i] .* pts[i] for i in 1:3)
    C0 = sum(wn[i] .* (pts[i] .- m0) * (pts[i] .- m0)' for i in 1:3)
    @test belief_mean(b) ≈ m0 && belief_cov(b) ≈ C0
    gm, = Zygote.gradient(x -> sum(belief_mean(SampleBelief(x, w))), pts)
    @test gm ≈ [wn[i] .* ones(2) for i in 1:3]
    gC, = Zygote.gradient(x -> tr(belief_cov(SampleBelief(x, w))), pts)
    @test gC ≈ [2 .* wn[i] .* (pts[i] .- m0) for i in 1:3]
    # scalar samples: Zygote, and a length-one vector out
    gs, = Zygote.gradient(x -> sum(belief_mean(SampleBelief(x))), [1.0, 2.0])
    @test gs ≈ [0.5, 0.5]
    @test belief_mean(SampleBelief([1.0, 2.0])) isa Vector{<:Real} && length(belief_mean(SampleBelief([1.0, 2.0]))) == 1
    # scalar type preservation: Float32 samples give Float32 moments, also unweighted
    X32 = Float32[1 2 4; 0.5 -1 2]
    @test belief_mean(SampleBelief(X32)) isa Vector{Float32}
    @test belief_cov(SampleBelief(X32)) isa Matrix{Float32}
    @test belief_mean(SampleBelief(X32, Float32[0.2, 0.3, 0.5])) isa Vector{Float32}
    @test belief_mean(SampleBelief([Float32(1), Float32(2)])) == [1.5f0]
    # and the values still match the Float64 oracle
    @test belief_mean(SampleBelief(X32)) ≈ vec(mean(Float64.(X32); dims = 2)) rtol = 1e-6
    # weights given as a tuple, as AdversarialFactors passes them (regression of GH-21: the matrix
    # product refused a tuple), and vectors of vectors against the matrix form
    pts2 = [[1.0, 0.0], [3.0, 2.0]]
    @test belief_mean(SampleBelief(pts2, (0.25, 0.75))) ≈ [2.5, 1.5]
    @test belief_cov(SampleBelief(pts2, (0.25, 0.75))) ≈ 0.1875 .* [4.0 4.0; 4.0 4.0]
    @test belief_cov(SampleBelief(pts2, [0.25, 0.75])) ≈ belief_cov(SampleBelief([1.0 3.0; 0.0 2.0], [0.25, 0.75]))
end
