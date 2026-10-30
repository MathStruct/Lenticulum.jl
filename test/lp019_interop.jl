using Lenticulum, LenticulumCore, Test, ExponentialFamily, BayesBase, Distributions
import Zygote
using Mycelium: combine

# Regression tests for LP-019 (Codex's review of the core packages) that need LenticulumCore's
# ChainRulesCore and ExponentialFamily extensions.

@testset "LP-019.7 the weight-normalisation pullback does not overflow" begin
    for a in (1.0, 1e308, 1e-307, 1e-308, 1e-320)
        g = Zygote.gradient(w -> sum(CategoricalBelief(w).logp), fill(a, 2))[1]
        @test g == [0.0, 0.0]                  # was [-Inf, -Inf] at 1e-308, NaN at 1e-320
    end
    # ∂ log(wᵢ/Σw)/∂wⱼ = δᵢⱼ/wᵢ − 1/Σw, against the closed form
    f(w) = CategoricalBelief(w).logp[1]
    @test Zygote.gradient(f, [1.0, 3.0])[1] ≈ [1 - 1 / 4, -1 / 4]
    @test Zygote.gradient(f, [1e-308, 3e-308])[1] ≈ [0.75e308, -0.25e308]
    # disparate weights (LP-019 review): p₁ underflows, ∂logp₂/∂w₁ = −1/Σw = −1e-200
    @test Zygote.gradient(w -> CategoricalBelief(w).logp[2], [1e-200, 1e200])[1] ≈ [-1e-200, 0.0]
    # and the shared term must not overflow: ∂logp₂/∂w = (−1/Σw, 1/w₂ − 1/Σw, −1/Σw), Σw = 2e308
    @test Zygote.gradient(w -> CategoricalBelief(w).logp[2], [1e-200, 1e308, 1e308])[1] ≈ [-5e-309, 5e-309, -5e-309]
    # a finite upstream cotangent near floatmax: 1e308 · Σ logp is constant in w at equal weights
    @test Zygote.gradient(w -> 1e308 * sum(CategoricalBelief(w).logp), [1e308, 1e308])[1] == [0.0, 0.0]
end

@testset "LP-019 third review: the covariance pullback is symmetric" begin
    # Σ = sym(M)⁻¹, so ∂Σ₁₂/∂M₁₂ = ∂Σ₁₂/∂M₂₁; against ForwardDiff on the forward map
    import ForwardDiff
    M = [2.0 0.3; 0.3 1.0]
    f(M) = belief_cov(GaussianBelief(zeros(2), M))[1, 2]
    gz = Zygote.gradient(f, M)[1]
    gf = ForwardDiff.gradient(f, M)
    @test gz ≈ gf
    @test gz[1, 2] ≈ gz[2, 1]
end

@testset "LP-019.8 scalar particles survive a SampleList round trip" begin
    s = LenticulumCore.SampleBelief([1, 3], [0.5, 0.5])
    back = as_belief(as_distribution(s))
    @test back.samples == [1, 3] && eltype(back.samples) <: Real   # was [fill(1), fill(3)]
    @test back.weights ≈ [0.5, 0.5]
    # Pooling with a categorical reweights by p[1] = 0.2 and p[3] = 0.5, as before the trip.
    c = LenticulumCore.CategoricalBelief([0.2, 0.3, 0.5])
    @test combine(back, c).weights ≈ [2 / 7, 5 / 7]
    # Vector particles still come back as vectors.
    v = as_belief(as_distribution(LenticulumCore.SampleBelief([[1.0], [2.0]], [1.0, 1.0])))
    @test v.samples == [[1.0], [2.0]]
end

@testset "LP-019 follow-up: LP-018.7 validation admits rounding-level asymmetry" begin
    # J S Jᵀ in floating point need not be exactly symmetric; AlgebraicFactors builds such
    # noise covariances. One ulp off the diagonal must pass, a real asymmetry must not.
    Q = [2.0 0.5; 0.5 1.0]
    Q[1, 2] = nextfloat(Q[1, 2])
    @test LinearConstraintFactor((a = 1, b = 1), 2; noise = Q) isa LinearConstraintFactor
    @test NonlinearFactor(x -> x, 2 => 2; noise = Q) isa NonlinearFactor
    @test_throws ArgumentError LinearConstraintFactor((a = 1, b = 1), 2; noise = [1.0 0.2; 10.0 1.0])
    @test_throws ArgumentError NonlinearFactor(x -> x, 2 => 2; noise = [1.0 0.2; 10.0 1.0])
    # LP-019 review: within √eps(Float64) is fine, beyond it (and indefinite) is not
    M = [1.0 0.9999; 1.0002 1.0]
    @test_throws ArgumentError LinearConstraintFactor((a = 1, b = 1), 2; noise = M)
    @test_throws ArgumentError NonlinearFactor(x -> x, 2 => 2; noise = M)
    @test LinearConstraintFactor((a = 1, b = 1), 2; noise = Q).Q == LinearConstraintFactor((a = 1, b = 1), 2; noise = Q).Q'
end
