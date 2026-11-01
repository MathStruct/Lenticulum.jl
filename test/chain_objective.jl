using Test
isdefined(@__MODULE__, :ChainObjective) ||
    include(joinpath(@__DIR__, "..", "examples", "chain_objective.jl"))

@testset "Propositions 9 and 10: the objective, not the class (examples/chain_objective.jl)" begin
    # a cheap version of the script's run
    r = ChainObjective.run(; ntrain = 1000, ntest = 200, chains = [(5, 2)], Ks_flat = (2,), S = 8, iters = 20, em_starts = 32)
    gap(prefix) = r.truth - only(row.ll for row in r.rows if startswith(row.model, prefix))
    @test r.err < 1e-10                        # Proposition 10: the additive model is a 2-outcome chain
    @test gap("additive") < 0.05
    @test gap("ELBO-trained") > gap("additive") + 0.5   # the same kind of chain, trained by the bound
    # the control: one family (constant gates), both scored exactly; only the objective differs
    c = ChainObjective.objective_control(; ntrain = 1000, ntest = 200, chains = [(4, 2)], em_starts = 32, S = 8)
    ex(prefix) = c.truth - only(row.exact for row in c.rows if startswith(row.objective, prefix))
    @test ex("exact") < 0.05
    @test ex("diffusion") > ex("exact") + 0.5
end
