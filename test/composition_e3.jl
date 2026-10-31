using Test
isdefined(@__MODULE__, :CompositionE3) ||
    include(joinpath(@__DIR__, "..", "examples", "composition_e3.jl"))

@testset "E3: composition of binary switches vs a flat mixture (examples/composition_e3.jl)" begin
    M = CompositionE3
    using Random: Xoshiro
    # the coupling normaliser Z̄ of the rank-one product of experts: the density has mass one
    X = M.E.sample_gmm(M.DATA.grid3.π, M.DATA.grid3.μ, M.DATA.grid3.Σ, 300, Xoshiro(1))
    θ = M.rank1_init(3, X, Xoshiro(3))
    @test M.rank1_mass(θ, 3; R = 15.0) ≈ 1 atol = 1e-8
    # a cheap version of the comparison (the full one is the script)
    gap(r, prefix) = r.truth - only(row.test for row in r.rows if startswith(row.model, prefix))
    g = M.run(:grid3; ntrains = (400,), ntest = 500, Ks = (8,), starts = 8, iters = 600)
    @test gap(g, "additive") < 0.1             # product structure: the additive composition fits
    @test gap(g, "rank-one") > 0.4             # rank-one constraints on a shared z do not
    r = M.run(:ring; ntrains = (400,), ntest = 500, Ks = (8,), starts = 8, iters = 600)
    @test gap(r, "additive") > 0.5             # no product structure: it does not
    @test gap(r, "additive") > gap(r, "flat") + 0.3
end
