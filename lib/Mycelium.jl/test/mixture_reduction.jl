# P17: Mycelium reduces mixtures of Gaussians itself (Runnalls' merge moved here from Lenticulum's
# `beliefs.jl`, where `reduce_message` was type piracy). This file loads Mycelium and LenticulumCore
# only. Oracle: a merge keeps the mixture's total mean and covariance exactly, and a reduction to
# one component is the moment-matched Gaussian.
using Mycelium, LenticulumCore, LinearAlgebra, Random, Test

@testset "P17: mixture reduction in Mycelium, without Lenticulum" begin
    LC = LenticulumCore
    @test parentmodule(which(reduce_message, (LC.MixtureBelief{LC.GaussianBelief{Vector{Float64},Matrix{Float64}}}, Int))) === Mycelium
    spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))
    moments(m) = begin
        w = LC.mixture_weights(m)
        μs = [belief_mean(c) for c in m.components]
        μ = sum(w[k] .* μs[k] for k in eachindex(w))
        Σ = sum(w[k] .* (belief_cov(m.components[k]) .+ (μs[k] .- μ) * (μs[k] .- μ)') for k in eachindex(w))
        μ, Σ
    end
    rng = Xoshiro(2617)
    for trial in 1:30
        n, K = rand(rng, 1:3), rand(rng, 2:6)
        m = LC.MixtureBelief([LC.Gaussian(randn(rng, n), spd(rng, n)) for _ in 1:K], rand(rng, K) .+ 0.1)
        μ, Σ = moments(m)
        for M in 1:(K + 1)
            r = reduce_message(m, M)
            if M >= K
                @test r === m                                   # within budget: unchanged
            elseif M == 1
                @test r isa GaussianBelief                      # one component: the component
                @test belief_mean(r) ≈ μ atol = 1e-10
                @test belief_cov(r) ≈ Σ atol = 1e-10
            else
                @test r isa LC.MixtureBelief && length(r.components) == M
                μr, Σr = moments(r)
                @test μr ≈ μ atol = 1e-10
                @test Σr ≈ Σ atol = 1e-10
            end
        end
    end
    # an improper (likelihood) component over the budget is refused; within it, passed through
    m = LC.MixtureBelief([LC.Gaussian([0.0], fill(1.0, 1, 1)), GaussianBelief([1.0], fill(0.0, 1, 1), true)], [0.5, 0.5])
    @test_throws ArgumentError reduce_message(m, 1)
    @test reduce_message(m, 2) === m
    @test_throws ArgumentError reduce_mixture(m; max_components = 0)
end
