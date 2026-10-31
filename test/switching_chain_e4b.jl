using Test
isdefined(@__MODULE__, :SwitchingChainE4b) ||
    include(joinpath(@__DIR__, "..", "examples", "switching_chain_e4b.jl"))

@testset "E4(b): switching chains on two moons and a ring (examples/switching_chain_e4b.jl)" begin
    M = SwitchingChainE4b
    # the exact densities integrate to one
    h = 0.04; g = -4:h:4
    for name in (:moons, :ring)
        @test sum(exp(M.data_logpdf(name, [x, y]; m = 800)) for x in g for y in g) * h^2 ≈ 1 atol = 1e-6
    end
    # a cheap version of the comparison (the full one is the script): a chain with K outcomes per
    # step scores like a flat K-component mixture
    r = M.run(:moons; ntrain = 1000, ntest = 200, ngen = 300, Ks_flat = (2,), chains = [(5, 2)], S = 8, iters = 20)
    flat = only(row for row in r.rows if startswith(row.model, "flat"))
    chain = only(row for row in r.rows if startswith(row.model, "chain"))
    @test abs(chain.ll - flat.ll) < 0.1
    @test chain.ll >= chain.elbo - 0.02        # the importance-weighted bound is the tighter one
end
