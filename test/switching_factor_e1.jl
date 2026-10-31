using Test
isdefined(@__MODULE__, :SwitchingE1) ||
    include(joinpath(@__DIR__, "..", "examples", "switching_factor_e1.jl"))

@testset "E1: switching constraint factor on non-orthogonal lines (examples/switching_factor_e1.jl)" begin
    r = SwitchingE1.run(; n = 8_000, ntest = 3_000)
    # the general-form switching factor recovers both lines at 0° and 60° ...
    @test maximum(r.err) < 0.5
    @test r.q ≈ [0.05^2, 0.05^2] rtol = 0.15            # the across-line variance is the noise σ²
    # ... and its conditional is as good as the true model's and the free mixture's,
    @test abs(r.cond.switching - r.cond.oracle) < 0.02
    @test abs(r.cond.switching - r.cond.gmm) < 0.02
    # where the spectral measurement and a single Gaussian are far worse
    @test r.cond.switching > r.cond.spectral + 1 && r.cond.switching > r.cond.gauss + 1
    @test abs(r.joint.switching - r.joint.oracle) < 0.02
    # message passing in a Mycelium graph reproduces the direct conditional (Z_k and masses kept)
    @test r.graph_err < 1e-10
end
