using Test
isdefined(@__MODULE__, :SwitchingE4a) ||
    include(joinpath(@__DIR__, "..", "examples", "switching_reverse_chain_e4a.jl"))

@testset "E4(a): switching reverse chain on mixture data (examples/switching_reverse_chain_e4a.jl)" begin
    M = SwitchingE4a
    # the cheap parts: the oracle and the K = 3 vs K = 1 comparison (the shared fit and the
    # 100-step variant are in the script)
    r = M.run(; T = 10, β = 1 - 0.8^2, ntrain = 1500, ntest = 300, ngen = 400, Ks = (3, 1), shared = false)
    # Proposition 8: exact kernels + exact prior make every path's bound equal log q0(z0)
    @test r.oracle_gap < 1e-10
    rows = Dict(row.name => row for row in r.rows)
    k3 = only(v for (k, v) in rows if startswith(k, "free per step, K=3"))
    k1 = only(v for (k, v) in rows if startswith(k, "free per step, K=1"))
    # linear-Gaussian steps (K = 1) cannot be multimodal (Proposition 2); K = 3 per step can
    @test k3.elbo > k1.elbo + 0.5
    @test k3.ed < k1.ed / 3
    @test k3.elbo <= r.truth + 0.05            # a bound, up to sampling error
end
