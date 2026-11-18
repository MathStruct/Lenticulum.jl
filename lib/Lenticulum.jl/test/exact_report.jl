# P13: a tree schedule's single sweep is a fixed point; the report calls it exact (`rep.exact`) only
# when nothing approximated (Mycelium `passing.jl`, `exact_messages`). Each claim is checked against
# a closed form: where the report says exact the marginals are the enumerated ones, and where it
# says not exact they differ from them (so the report is not merely cautious there).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using LenticulumCore: MixtureBelief, CategoricalBelief
isdefined(@__MODULE__, :mr_chain) || include(joinpath(@__DIR__, "mixture_rules.jl"))
isdefined(@__MODULE__, :cl_additive) || include(joinpath(@__DIR__, "cluster.jl"))

@testset "P13: the exactness report of a tree sweep" begin
    μe, ve = mr_chain_exact()
    chain_err(mg) = maximum(t -> (q = moment_match(getfield(mg, Symbol(:x, t)));
        max(abs(only(belief_mean(q)) - μe[t]), abs(only(belief_cov(q)) - ve[t]))), 1:MR_T)

    @testset "a switching chain on belief propagation: mixtures collapsed, not exact" begin
        g = mr_chain(BeliefPropagation())
        mg, rep, _, _ = infer!(g, tree_schedule(g), MR_CHAIN_PS, NamedTuple())
        @test rep.converged && rep.sweeps == 1
        @test !rep.exact
        @test occursin("not exact", rep.reason) && occursin("collapses", rep.reason)
        @test !occursin("does not declare", rep.reason)        # every factor here is declared
        @test chain_err(mg) > 0.005                             # and indeed not the enumerated marginals (0.011)
    end

    @testset "MixtureReduction nodes: not claimed exact" begin
        for M in (1, 16)
            g = mr_chain(MixtureReduction(M))
            mg, rep, _, _ = infer!(g, tree_schedule(g), MR_CHAIN_PS, NamedTuple())
            @test !rep.exact && occursin("MixtureReduction", rep.reason)
            # with M = 16 nothing is merged and the marginals are exact: the report is cautious there
            M == 1 && @test chain_err(mg) > 0.1
        end
    end

    @testset "the same chain in one cluster node: exact, and the enumerated marginals" begin
        g = mr_chain(BeliefPropagation())
        gc = cluster(g, :all, [f.name for f in g.factors if !(f.factor isa DataFactor)])
        psc = cluster_parameters(gc, :all, MR_CHAIN_PS)
        _, rep, _, s = infer!(gc, tree_schedule(gc), psc, NamedTuple())
        @test rep.exact && rep.reason == "tree schedule: one sweep is exact"
        for t in 1:MR_T
            q = moment_match(cluster_marginal(s, gc, :all, Symbol(:x, t), psc))
            @test only(belief_mean(q)) ≈ μe[t] atol = 1e-12
        end
    end

    @testset "the additive wiring: a mixture into a sum factor is collapsed; clustered, exact" begin
        g = cl_additive()
        ps = (data = NamedTuple(), sum = CL_SUM, sw1 = cl_sw(1), sw2 = cl_sw(2), sw3 = cl_sw(3))
        _, rep, _, _ = infer!(g, tree_schedule(g), ps, NamedTuple())
        @test !rep.exact && occursin("node :sum receives a mixture", rep.reason)
        gc = cluster(g, :all, [:sum, :sw1, :sw2, :sw3])
        _, repc, _, _ = infer!(gc, tree_schedule(gc), cluster_parameters(gc, :all, ps), NamedTuple())
        @test repc.exact
    end

    @testset "a mixture that only reaches a leaf is exact (random instances against the closed form)" begin
        # prior on u, a switching factor on (u, k) with no message on k: u's marginal is the prior
        # times the mixture Σₖ wₖ N(u; bₖ, q), a mixture whose moments are known in closed form. The
        # prior receives the mixture but sends on no other edge, so nothing collapses.
        rng = Xoshiro(1313)
        for trial in 1:20
            K = rand(rng, 2:4)
            μ0, v0, q = randn(rng), 0.5 + rand(rng), 0.1 + rand(rng)
            bs, logw = 2 .* randn(rng, K), randn(rng, K)
            gb = GraphBuilder(); variable!(gb, :u, 1); variable!(gb, :k, K)
            factor!(gb, :prior, GaussianPrior(:u, μ0, v0)); connect!(gb, :prior, :u, :u)
            factor!(gb, :sw, SwitchingConstraintFactor((u = 1,), K, 1; noise = q))
            connect!(gb, :sw, :u, :u); connect!(gb, :sw, :k, :k)
            g = build(gb)
            ps = (prior = NamedTuple(), sw = (A = (u = fill(1.0, 1, 1),), b = reshape(bs, 1, K), logw = logw))
            mg, rep, _, _ = infer!(g, tree_schedule(g), ps, NamedTuple())
            @test rep.exact
            # closed form: weight ∝ wₖ N(bₖ; μ0, v0 + q), component N((μ0/v0 + bₖ/q)/(1/v0 + 1/q), 1/(1/v0 + 1/q))
            lw = logw .- (bs .- μ0) .^ 2 ./ (2 * (v0 + q)); W = exp.(lw .- maximum(lw)); W ./= sum(W)
            v = 1 / (1 / v0 + 1 / q); ms = v .* (μ0 / v0 .+ bs ./ q)
            mean = sum(W .* ms); var = sum(W .* (v .+ (ms .- mean) .^ 2))
            qu = moment_match(mg.u)
            @test only(belief_mean(qu)) ≈ mean atol = 1e-10
            @test only(belief_cov(qu)) ≈ var atol = 1e-10
            @test exp.(mg.k.logp) ≈ W atol = 1e-10
        end
    end

    @testset "a factor whose messages read the cavity: NonlinearFactor (GH-15)" begin
        gb = GraphBuilder(); variable!(gb, :x, 2); variable!(gb, :y, 1)
        factor!(gb, :prior, GaussianPrior(:x, zeros(2), Matrix(1.0I, 2, 2))); connect!(gb, :prior, :x, :x)
        factor!(gb, :nl, NonlinearFactor(x -> tanh.([1.0 2.0] * x), 2 => 1; noise = fill(0.3, 1, 1)))
        connect!(gb, :nl, :x, :x); connect!(gb, :nl, :y, :y)
        factor!(gb, :d, DataFactor(:y, [0.4])); connect!(gb, :d, :y, :y; direction = Emitting())
        g = build(gb)
        _, rep, _, _ = infer!(g, tree_schedule(g), LuxCore.initialparameters(Xoshiro(1), g), NamedTuple())
        # its backward message linearises at the target's cavity, so the tree is iterated and
        # the residual checked: the report never claims exactness, nor a one-sweep fixed point
        @test !rep.exact
        @test !(rep.converged && rep.sweeps == 1)
    end

    @testset "iterated schedules never claim exactness" begin
        g = mr_chain(BeliefPropagation())
        _, rep, _, _ = infer!(g, SequentialSchedule(all_tasks(g)), MR_CHAIN_PS, NamedTuple())
        @test !rep.exact
        _, rep, _, _ = infer!(g, tree_schedule(g), MR_CHAIN_PS, NamedTuple(); damping = 0.3)
        @test !rep.exact
    end
end
