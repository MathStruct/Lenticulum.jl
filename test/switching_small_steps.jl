using Test
isdefined(@__MODULE__, :SwitchingSmallSteps) ||
    include(joinpath(@__DIR__, "..", "examples", "switching_small_steps.jl"))

@testset "Small-step test: conditional per-step switching kernels (examples/switching_small_steps.jl)" begin
    M = SwitchingSmallSteps
    # a cheap version of the sweep: few steps, few pairs, short refits (the full sweep is the script)
    r = M.run(; Ts = (4,), Ks = (3, 1), ntest = 300, ngen = 400, n = 1500, iters = 30)
    k3 = only(row for row in r.rows if row.K == 3)
    k1 = only(row for row in r.rows if row.K == 1)
    @test k3.gain > 0                          # the conditional refit improves on the joint EM fit
    @test k3.bound > k1.bound + 0.5            # linear-Gaussian steps stay Gaussian (Proposition 2)
    @test k3.bound <= r.truth + 0.05           # a bound, up to sampling error
    @test M.estimation_cost(100, 3, 6000) ≈ 100 * 44 / 12000
end
