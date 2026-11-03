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

# A TableFactor ties discrete variables; inside a cluster its joint state is the outcome (a tuple
# of switch channels), and the assignments it rules out (-Inf) are dropped before the Gaussian part.
@testset "TableFactor: exact in a cluster and by belief propagation" begin
    Cat = LenticulumCore.CategoricalBelief
    pa, pb, pc = [0.2, 0.5, 0.3], [0.6, 0.4], [0.1, 0.9]
    Tab = [0.3 -Inf; 1.0 0.2; -0.5 0.7]           # a × b
    Tbc = [0.0 0.4; -Inf 1.2]                      # b × c
    function gt()
        b = GraphBuilder(); variable!(b, :a, 3); variable!(b, :b, 2); variable!(b, :c, 2)
        for (n, v, p) in ((:pa, :a, pa), (:pb, :b, pb), (:pc, :c, pc))
            factor!(b, n, PriorFactor(v, Cat(p), _ -> 0.0)); connect!(b, n, v, v)
        end
        factor!(b, :tab, TableFactor((a = 3, b = 2), Tab)); connect!(b, :tab, :a, :a); connect!(b, :tab, :b, :b)
        factor!(b, :tbc, TableFactor((b = 2, c = 2), Tbc)); connect!(b, :tbc, :b, :b); connect!(b, :tbc, :c, :c)
        return build(b)
    end
    # exact joint by enumeration
    P = [pa[i] * pb[j] * pc[k] * exp(Tab[i, j] + Tbc[j, k]) for i in 1:3, j in 1:2, k in 1:2]
    Z = sum(P)
    g = gt()
    gc = cluster(g, :all, [:pa, :pb, :pc, :tab, :tbc])
    @test Set(gc.factors[gc.factor_index[:all]].factor.discrete) == Set([:a, :b, :c])
    ps = cluster_parameters(gc, NamedTuple())
    s = MessageStore(gc); sweep!(s, gc, SequentialSchedule(all_tasks(gc)), ps, NamedTuple())
    @test first(scalar_free_energy(s, gc, ps, NamedTuple())) ≈ -log(Z) atol = 1e-12
    @test exp.(cluster_marginal(s, gc, :all, :b, ps).logp) ≈ vec(sum(P; dims = (1, 3))) ./ Z atol = 1e-12
    @test exp.(cluster_marginal(s, gc, :all, :a, ps).logp) ≈ vec(sum(P; dims = (2, 3))) ./ Z atol = 1e-12
    # the graph is a tree, so belief propagation through the tables is exact too
    s2 = MessageStore(g)
    for _ in 1:10; sweep!(s2, g, SequentialSchedule(all_tasks(g)), NamedTuple(), NamedTuple()); end
    @test exp.(marginal(s2, g, g.variable_index[:a]).logp) ≈ vec(sum(P; dims = (2, 3))) ./ Z atol = 1e-12
    @test exp.(marginal(s2, g, g.variable_index[:c]).logp) ≈ vec(sum(P; dims = (1, 2))) ./ Z atol = 1e-12
    # an observed state enters as a point
    m, _ = factor_message(TableFactor((a = 3, b = 2), Tab), :a, nothing, (b = LenticulumCore.DiracBelief(2),), nothing, NamedTuple(), NamedTuple())
    @test exp.(m.logp) ≈ exp.(Tab[:, 2]) ./ sum(exp.(Tab[:, 2]))
    # the column order of a tuple switch is column-major (the first channel fastest)
    @test cluster_form(TableFactor((a = 3, b = 2), Tab), NamedTuple()).logw == vec(Tab)
    @test_throws DimensionMismatch TableFactor((a = 2, b = 2), Tab)
    @test_throws ArgumentError TableFactor((a = 1,), [-Inf])
end
