using Test
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
end
