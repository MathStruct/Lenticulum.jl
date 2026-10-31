using Test
isdefined(@__MODULE__, :HierarchyE3b) ||
    include(joinpath(@__DIR__, "..", "examples", "hierarchy_e3b.jl"))

@testset "E3b: a switch over additive models on tree-structured data (examples/hierarchy_e3b.jl)" begin
    # a cheap version of the script's run
    r = HierarchyE3b.run(; ntrains = (400,), ntest = 500, starts = 32, Ls = (4,), Ks = (12,))
    gap(prefix) = r.truth - only(row.test for row in r.rows if startswith(row.model, prefix))
    hier = only(row for row in r.rows if startswith(row.model, "hierarchical"))
    @test hier.mono                            # exact EM never decreases the log-likelihood
    @test gap("hierarchical") < 0.15
    @test gap("additive") > gap("hierarchical") + 0.4   # one sum node cannot rotate its local grids
end
