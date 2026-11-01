using Test, Random
isdefined(@__MODULE__, :SwitchingBlocksGraph) ||
    include(joinpath(@__DIR__, "..", "examples", "switching_blocks_graph.jl"))

@testset "Blocks of switches as SwitchingConstraintFactors in Mycelium (examples/switching_blocks_graph.jl)" begin
    # a cheap version of the run: one seed, L = 8
    rows = SwitchingBlocksGraph.run(; seeds = (1,), Ls = (8,), ntrain = 1000, ntest = 1000, iters = 100, ncheck = 3)
    r(m) = only(x for x in rows if x.method == m)
    # Mycelium's infer! on the block graph agrees with the vectorised E-step, and blocks of one are E6's ADF
    @test r("exact").graph_err < 1e-9
    @test r("exact").e6_err < 1e-12
    # exact blocks of overlapping switches remove part of the bias in one pass, iterating removes more
    @test r("graph, overlap blocks of ≤ 4").gap < r("graph, blocks of 1 (E6's ADF)").gap - 0.2
    @test r("structured MF, overlap blocks of ≤ 4 (E7)").gap < r("graph, overlap blocks of ≤ 4").gap - 0.1
    @test r("exact").gap < 0.1
    # the hybrid graph (MeanField sum node, BP block factors) is E7's structured mean field: per point...
    @test r("exact").hybrid_err < 1e-6
    # ...and as an E-step: EM through the graph lands where E7's code does
    S = SwitchingBlocksGraph; F = S.F
    tm = F.truth(8, 8; seed = 1); X = F.sample(tm, 150, Xoshiro(10))
    blocking = G -> S.overlap_blocks(G, 4)
    mh = S.fit_hybrid(tm, X, blocking; iters = 8)
    ms = S.fit_structured(tm, X, blocking; iters = 8)
    @test maximum(abs.(mh.W - ms.W)) < 1e-6 && maximum(abs.(mh.Σ - ms.Σ)) < 1e-6 && maximum(abs.(mh.p - ms.p)) < 1e-6
end
