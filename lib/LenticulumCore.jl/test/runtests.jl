using LenticulumCore
using LenticulumCore: Channel
using Statistics: mean, cov
using Test

diag_of(C) = [C[i, i] for i in axes(C, 1)]

@testset "LenticulumCore" begin
    @testset "channels and polarity" begin
        # P_in + P_out + P_latent = Id, pairwise products zero.
        p = Polarity(; x = Observed(), y = Unobserved(), h = Latent())
        @test observed_channels(p) == (:x,)
        @test unobserved_channels(p) == (:y,)
        @test latent_channels(p) == (:h,)
        @test length(observed_channels(p)) + length(unobserved_channels(p)) +
              length(latent_channels(p)) == length(keys(p))
        @test ispartition(p)

        # ρ_in defaults to a hard clamp (a categorical cup); the rest are free.
        @test channel_precision(p, :x) == Inf
        @test channel_precision(p, :y) == 1.0

        c = Channel(:x, 3)
        @test channelname(c) === :x
        @test channelspace(c) == 3
    end

    @testset "energy spaces compose by direct sum" begin
        Ec = EuclideanEnergySpace(2)
        Ed = ScalarEnergySpace()
        E = GradedEnergySpace((; first = Ec, second = Ed))
        @test dimension(E) == 3
        @test dimension(Ec ⊕ Ed) == 3
    end

    @testset "graded energy is a vector space" begin
        a = GradedEnergy((; first = [1.0, 2.0], second = 3.0))
        b = GradedEnergy((; first = [0.5, 0.5], second = 1.0))
        @test (a + b)[:first] == [1.5, 2.5]
        @test (a - b)[:second] == 2.0
        @test (2.0 * a)[:first] == [2.0, 4.0]
        @test zero(a)[:second] == 0.0
    end

    @testset "scalarisation recovers AutoBayes Definition 22" begin
        # σ_dc(e_c, e_d) = σ_c(e_c) + σ_d(e_d): the paper's `+` is the scalarisation of ⊕.
        σ = GradedScalarisation((; first = SquaredNorm(), second = IdentityScalarisation()))
        e = compose_energy([3.0, 4.0], 2.0)
        @test scalarise(σ, e) ≈ 0.5 * 25 + 2.0
        @test !islinear(σ)

        σlin = GradedScalarisation((;
            first = WeightedSum([1.0, 2.0]), second = IdentityScalarisation()
        ))
        @test islinear(σlin)
        @test scalarise(σlin, e) ≈ (3.0 + 8.0) + 2.0
    end

    @testset "linear σ ⇒ strict chain rule; convex σ ⇒ lax by the Jensen gap" begin
        # Samples of 𝐅^c(π, y) at (y,b) drawn from the downstream inversion.
        Fc = [[1.0, 0.0], [-1.0, 2.0], [0.0, 1.0], [2.0, -1.0]]

        # Linear scalarisation: σ(E[·]) == E[σ(·)], so the two chain rules agree exactly.
        σl = WeightedSum([1.0, -0.5])
        @test islinear(σl)
        @test jensen_gap(σl, Fc) == 0.0
        @test scalarise(σl, mean(Fc)) ≈ mean(scalarise.(Ref(σl), Fc))

        # Convex scalarisation: the gap is exactly ½ tr Cov (population covariance).
        σc = SquaredNorm()
        gap = jensen_gap(σc, Fc)
        M = reduce(hcat, Fc)'                       # 4×2, samples in rows
        C = cov(M; corrected = false)
        @test gap ≈ 0.5 * sum(diag_of(C)) atol = 1e-12
        @test gap > 0

        # AutoBayes Thm 23 (scalar) = multivariate chain rule + defect.
        Fd = 7.0
        σ = GradedScalarisation((; first = σc, second = IdentityScalarisation()))
        F_multi = scalarise(σ, compose_free_energy(Fc, Fd))
        F_scalar = mean(scalarise.(Ref(σc), Fc)) + Fd
        @test F_scalar ≈ F_multi + chain_rule_defect(σc, Fc)
        @test F_multi <= F_scalar        # Jensen: the multivariate composite is a lower bound
    end

    @testset "entropy composition averages under the downstream inversion" begin
        Hc = [1.0, 2.0, 3.0]
        H = compose_entropy(Hc, 10.0)
        @test H[:first] ≈ 2.0
        @test H[:second] == 10.0
    end
end
