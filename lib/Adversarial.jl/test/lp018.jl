using Adversarial, LenticulumCore, Mycelium, LuxCore, Test
struct LP018ConstantRatio <: LuxCore.AbstractLuxLayer end
(::LP018ConstantRatio)(x, ps, st) = (2., st)
@testset "LP-018.9 finite overflowing particle weights" begin
    for weights in ([1e308, 1e308], (1e308, 1e308))
        b = SampleBelief([[1.], [3.]], weights)
        @test weighted_mean(b) ≈ [2.]
        @test weighted_cov(b) ≈ ones(1, 1)
        @test effective_sample_size(b) ≈ 2.
        f = RatioFactor(LP018ConstantRatio(), 1)
        @test first(Mycelium.local_free_energy(f, (x = b,), (;), (;))) ≈ -2.
    end
end
