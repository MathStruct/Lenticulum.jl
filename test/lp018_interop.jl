using Lenticulum, LenticulumCore, Test, ExponentialFamily, BayesBase, Distributions
import Zygote
@testset "LP-018.9 finite particle weights across interop and AD" begin
    b = SampleBelief([[1.], [3.]], [1e308, 1e308])
    @test Distributions.mean(as_distribution(b)) ≈ [2.]
    g = Zygote.gradient(w -> sum(LenticulumCore._normalised_logweights(w)), [1e308, 1e308])[1]
    @test maximum(abs, g .* 1e308) < 1e-14
end
