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

@testset "LP-019 review: a small asymmetry cannot hide an indefinite form" begin
    M = [1.0 0.9999; 1.0002 1.0]          # upper mirror definite, symmetric part indefinite
    @test eigmin(Symmetric((M + M') / 2)) < 0
    @test !isproper(GaussianBelief(zeros(2), M))
    @test_throws ArgumentError Gaussian(zeros(2), M)
    # the symmetric part is what is factorised: one ulp of asymmetry, same moments either way
    S = [2.0 0.5; 0.5 1.0]; S2 = copy(S); S2[1, 2] = nextfloat(S2[1, 2])
    @test belief_cov(Gaussian(zeros(2), S2)) ≈ S
end

import ForwardDiff
@testset "LP-019 second review: the symmetric part keeps subnormals, duals agree" begin
    b = GaussianBelief([0.0], fill(nextfloat(0.0), 1, 1))
    @test isproper(b)                                   # A/2 + Aᵀ/2 had rounded it to 0
    # Float32 near-symmetry is judged the same with and without ForwardDiff
    M = Float32[1 0.1; 0.100001 1]
    @test isproper(GaussianBelief(zeros(Float32, 2), M))
    g = ForwardDiff.derivative(t -> sum(belief_mean(GaussianBelief(ones(Float32, 2), t .* M))), 1.0f0)
    @test isfinite(g)
end

@testset "LP-019 third review: display follows isproper" begin
    s = sprint(show, GaussianBelief(zeros(2), [1.0 0.9999; 1.0002 1.0]))
    @test occursin("improper", s)
end
