using Test
isdefined(@__MODULE__, :StructureSwitches) ||
    include(joinpath(@__DIR__, "..", "examples", "structure_switches.jl"))

@testset "Structure learning for switch wirings (examples/structure_switches.jl)" begin
    # a cheap version of the script's run, on the tree data (true structure J = 3, L = 2)
    r = only(StructureSwitches.run(; names = (:tree,), ntrain = 800, ntest = 500, starts = 8, maxJ = 4, maxL = 3))
    @test r.found == (3, 2) == r.exhaustive
    @test r.gap < 0.05
    @test r.fits < r.nstruct + 4                # greedy fits fewer models than the exhaustive grid (16) needs
    @test all(diff([p.score for p in r.path]) .< 0)   # every accepted move lowers BIC
end
