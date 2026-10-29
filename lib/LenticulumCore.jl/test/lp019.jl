using LenticulumCore, LinearAlgebra, Test

# Regression tests for LP-019 (Codex's review of the core packages).

@testset "LP-019.3 Gaussian validation reads both triangles" begin
    M = [1.0 0.2; 10.0 1.0]                    # symmetric part has eigenvalues −4.1 and 6.1
    @test_throws ArgumentError Gaussian(zeros(2), M)
    b = GaussianBelief(zeros(2), M)            # a likelihood may be anything; it is not proper
    @test !isproper(b)
    @test_throws ArgumentError belief_mean(b)
    @test_throws ArgumentError logpartition(b)
    # Rounding-level asymmetry, as left by A Λ Aᵀ, is still accepted.
    S = [2.0 0.5; 0.5 1.0]
    S[1, 2] += 1e-15
    @test belief_cov(Gaussian(zeros(2), S)) ≈ S
    @test isproper(GaussianBelief(zeros(2), S))
    # Symmetric singular precisions are improper, not invalid.
    @test !isproper(GaussianBelief([1.0, 0.0], [1.0 0.0; 0.0 0.0]))
end

# LP-019.7 (the weight-normalisation pullback) lives in the top-level test/lp019_interop.jl:
# the ChainRulesCore extension does not load in this environment.
