# Cluster nodes on Mycelium's own factors (`cluster.md`); Lenticulum's `test/cluster.jl` has the
# end-to-end tests (a switching chain against enumeration, E7's block factor).
@testset "cluster: Mycelium's side" begin
    G(μ, v) = LenticulumCore.Gaussian([μ], fill(v, 1, 1))
    function g0()
        b = GraphBuilder(); variable!(b, :x, 1); variable!(b, :k, 2)
        factor!(b, :p1, PriorFactor(:x, G(0.5, 2.0), _ -> 0.0)); connect!(b, :p1, :x, :x)
        factor!(b, :p2, PriorFactor(:x, G(-1.0, 0.5), _ -> 0.0)); connect!(b, :p2, :x, :x)
        factor!(b, :pk, PriorFactor(:k, LenticulumCore.CategoricalBelief([0.3, 0.7]), k -> 0.0)); connect!(b, :pk, :k, :k)
        factor!(b, :out, PriorFactor(:x, G(0.0, 4.0), _ -> 0.0)); connect!(b, :out, :x, :x)
        return build(b)
    end
    g = g0()
    gc = cluster(g, :c, [:p1, :p2, :pk])
    c = gc.factors[gc.factor_index[:c]].factor
    @test c.boundary == [:x] && :k in c.discrete && nvariables(gc) == 1
    # the message to x is the product of the two priors (k is internal and does not touch x)
    m, _ = factor_message(c, :x, nothing, NamedTuple(), nothing, cluster_parameters(gc, :c, NamedTuple()).c, NamedTuple())
    p = Mycelium.combine(G(0.5, 2.0), G(-1.0, 0.5))
    m = m isa LenticulumCore.MixtureBelief ? ep_project(m) : m
    @test LenticulumCore.belief_mean(m) ≈ LenticulumCore.belief_mean(p) && LenticulumCore.belief_cov(m) ≈ LenticulumCore.belief_cov(p)
    @test cluster_marginal(MessageStore(gc), gc, :c, :k, NamedTuple()).logp ≈ log.([0.3, 0.7])
    @test_throws ArgumentError cluster(g, :c, [:p1, :zz])
    @test_throws ArgumentError cluster(g, :out, [:p1])
end
