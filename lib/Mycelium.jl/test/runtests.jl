using Mycelium
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, Polarity
using Test

# ---------------------------------------------------------------------------
# A small supervised-learning graph, built the Lenticulum way: data, model and loss are
# ALL factors.  x --[relay/"model"]-- y --[loss]  , with y also observed via a target.
#
#     data_x --- x --- model --- y --- loss --- t --- data_t
#
# is a tree; adding an edge closes a loop.
# ---------------------------------------------------------------------------
function chain_graph()
    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y); variable!(b, :t)
    factor!(b, :data_x, DataFactor(:x, 2.0))
    factor!(b, :model, RelayFactor(:in, :out))
    factor!(b, :data_t, DataFactor(:t, 5.0))
    factor!(b, :loss, LossFactor((:pred, :target), (p, t) -> (p - t)^2 / 2))
    connect!(b, :data_x, :x, :x; direction = Emitting())
    connect!(b, :model, :in, :x; direction = Bidirectional())
    connect!(b, :model, :out, :y; direction = Bidirectional())
    connect!(b, :data_t, :t, :t; direction = Emitting())
    connect!(b, :loss, :pred, :y; direction = Absorbing())
    connect!(b, :loss, :target, :t; direction = Absorbing())
    return build(b)
end

@testset "Mycelium" begin

@testset "graph structure" begin
    g = chain_graph()
    @test nvariables(g) == 3
    @test nfactors(g) == 4
    @test nedges(g) == 6
    @test isconnected(g)
    # |F| + |V| - |E| = 4 + 3 - 6 = 1  =>  a tree
    @test euler_characteristic(g) == 1
    @test istree(g)
    validate(g)

    @test variable_degree(g, g.variable_index[:x]) == 2
    @test variable_degree(g, g.variable_index[:y]) == 2
    @test variable_degree(g, g.variable_index[:t]) == 2
    @test sort(collect(channels_of(g, g.factor_index[:loss]))) == [:pred, :target]
end

@testset "two acyclicity notions are independent" begin
    g = chain_graph()
    @test istree(g)             # undirected: no loops
    @test !isdag(g)             # directed: the Bidirectional model edges are 2-cycles
    @test_throws ArgumentError topological_order(g)

    # the same wiring with every edge unidirectional IS a DAG
    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y)
    factor!(b, :d, DataFactor(:x, 1.0))
    factor!(b, :m, RelayFactor(:in, :out))
    connect!(b, :d, :x, :x; direction = Emitting())
    connect!(b, :m, :in, :x; direction = Absorbing())
    connect!(b, :m, :out, :y; direction = Emitting())
    dag = build(b)
    @test isdag(dag)
    @test istree(dag)
    order = topological_order(dag)
    @test length(order) == nvariables(dag) + nfactors(dag)
    # the data factor must come before the variable it writes
    @test findfirst(==((:factor, dag.factor_index[:d])), order) <
          findfirst(==((:variable, dag.variable_index[:x])), order)
end

@testset "loops are detected" begin
    b = GraphBuilder()
    variable!(b, :a); variable!(b, :c)
    factor!(b, :f, RelayFactor(:p, :q))
    factor!(b, :g, RelayFactor(:p, :q))
    connect!(b, :f, :p, :a); connect!(b, :f, :q, :c)
    connect!(b, :g, :p, :a); connect!(b, :g, :q, :c)
    loopy = build(b)
    @test isconnected(loopy)
    @test euler_characteristic(loopy) == 2 + 2 - 4 == 0   # one independent loop
    @test !istree(loopy)
    @test_throws ArgumentError tree_schedule(loopy)
end

@testset "edge directions constrain polarity" begin
    g = chain_graph()
    fid = g.factor_index[:model]
    @test sort(legal_targets(g, fid)) == [:in, :out]
    @test sort(legal_sources(g, fid)) == [:in, :out]

    # a DataFactor only emits
    d = g.factor_index[:data_x]
    @test legal_targets(g, d) == [:x]
    @test isempty(legal_sources(g, d))

    # a LossFactor only absorbs, and is a sink
    l = g.factor_index[:loss]
    @test isempty(legal_targets(g, l))
    @test sort(legal_sources(g, l)) == [:pred, :target]
    @test issink(factornode(g, l).factor)
end

@testset "polarity resolution" begin
    g = chain_graph()
    fid = g.factor_index[:model]
    p = resolve_polarity(g, fid, :out, (:in,))
    @test p[:in] isa Observed
    @test p[:out] isa Unobserved
    @test observed_channels(p) == (:in,)
    @test unobserved_channels(p) == (:out,)
    check_legal(g, fid, p)                      # legal: both edges bidirectional

    # the reverse polarity is equally legal — this is what "implicit" buys
    q = resolve_polarity(g, fid, :in, (:out,))
    @test q[:in] isa Unobserved
    check_legal(g, fid, q)

    # a channel with no message becomes Latent
    r = resolve_polarity(g, fid, :out, ())
    @test r[:in] isa Latent

    # target cannot also be observed
    @test_throws Mycelium.PolarityError resolve_polarity(g, fid, :out, (:in, :out))
    # unknown channel
    @test_throws Mycelium.PolarityError resolve_polarity(g, fid, :nope, ())
end

@testset "illegal polarities are rejected by direction, not by the factor" begin
    g = chain_graph()
    l = g.factor_index[:loss]
    # the factor would refuse anyway, but the EDGE refuses first and names the channel
    p = resolve_polarity(g, l, :pred, (:target,))
    err = try; check_legal(g, l, p); nothing; catch e; e; end
    @test err isa Mycelium.PolarityError
    @test err.channel === :pred
    @test occursin("Absorbing", err.reason)
end

@testset "tree schedule: two sweeps, minus what the edge directions forbid" begin
    # all-bidirectional tree: the full 2|E| messages, each edge once per sweep,
    # in opposite directions
    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y)
    factor!(b, :f, RelayFactor(:p, :q)); factor!(b, :g, RelayFactor(:p, :q))
    connect!(b, :f, :p, :x); connect!(b, :f, :q, :y)
    connect!(b, :g, :p, :y); connect!(b, :g, :q, :x)   # NB: makes a loop
    # use a genuine bidirectional TREE instead
    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y)
    factor!(b, :f, RelayFactor(:p, :q)); factor!(b, :g, RelayFactor(:p, :q))
    connect!(b, :f, :p, :x); connect!(b, :f, :q, :y); connect!(b, :g, :p, :y)
    bt = build(b)
    @test istree(bt)
    s = tree_schedule(bt)
    @test s.pruned == 0
    @test length(s.inward) == nedges(bt)
    @test length(s.outward) == nedges(bt)
    @test length(tasks(s)) == 2 * nedges(bt)
    @test sort([t.edge for t in s.inward]) == collect(1:nedges(bt))
    inkind = Dict(t.edge => t.kind for t in s.inward)
    outkind = Dict(t.edge => t.kind for t in s.outward)
    @test all(inkind[e] != outkind[e] for e in 1:nedges(bt))

    # the supervised graph has unidirectional edges, so some messages are pruned:
    # 2 Emitting edges + 2 Absorbing edges each lose one direction
    g2 = chain_graph()
    s2 = tree_schedule(g2)
    @test s2.pruned == 4
    @test length(tasks(s2)) == 2 * nedges(g2) - 4
    @test all(islegal(g2, t) for t in tasks(s2))
end

@testset "forward-backward on a strict DAG has an empty backward sweep" begin
    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y)
    factor!(b, :d, DataFactor(:x, 3.0))
    factor!(b, :m, RelayFactor(:in, :out))
    connect!(b, :d, :x, :x; direction = Emitting())
    connect!(b, :m, :in, :x; direction = Absorbing())
    connect!(b, :m, :out, :y; direction = Emitting())
    g = build(b)
    s = forward_backward_schedule(g)
    @test !isempty(s.forward)
    # no Bidirectional edges => no backward BELIEF flow. This graph is explicit (Lux-like).
    @test isempty(s.backward)
end

@testset "messages: combine, exclusion, damping" begin
    @test combine(TrivialBelief(), DiracBelief(1)) === DiracBelief(1)
    @test combine(DiracBelief(2), TrivialBelief()) === DiracBelief(2)
    @test combine(DiracBelief(2), DiracBelief(2)) === DiracBelief(2)
    @test_throws ArgumentError combine(DiracBelief(2), DiracBelief(3))

    @test belief_distance(DiracBelief(1.0), DiracBelief(1.0)) == 0.0
    @test belief_distance(DiracBelief(1.0), DiracBelief(4.0)) == 3.0
    @test belief_distance(nothing, DiracBelief(1.0)) == Inf

    @test can_damp(DiracBelief(1.0), DiracBelief(3.0))
    @test damp(DiracBelief(1.0), DiracBelief(3.0), 0.5).value ≈ 2.0
    @test !can_damp(TrivialBelief(), TrivialBelief())

    # exclusion: the message a factor gets back must not contain its own contribution
    g = chain_graph()
    s = MessageStore(g)
    ex = g.edges_of_variable[g.variable_index[:x]]
    s.to_variable[ex[1]] = Message(DiracBelief(7.0), 1)
    @test marginal(s, g, g.variable_index[:x]) === DiracBelief(7.0)
    @test excluded_marginal(s, g, g.variable_index[:x], ex[1]) isa TrivialBelief
end

@testset "message passing on the tree gives the right answer" begin
    g = chain_graph()
    sched = tree_schedule(g)
    marg, report, st, store = infer!(g, sched, NamedTuple(), NamedTuple())
    @test report.converged
    @test report.sweeps == 1
    @test occursin("exact", report.reason)
    @test marg.x === DiracBelief(2.0)          # from data_x
    @test marg.t === DiracBelief(5.0)          # from data_t
    @test marg.y === DiracBelief(2.0)          # relayed through the model
end

@testset "counting numbers and the Euler characteristic" begin
    g = chain_graph()
    cn = counting_numbers(g)
    @test cn.x == -1 && cn.y == -1 && cn.t == -1      # all degree 2
    # Σ_f 1 + Σ_v (1 - d_v)  ==  |F| + |V| - |E|  ==  χ
    @test total_counting_number(g) == euler_characteristic(g) == 1

    b = GraphBuilder()
    variable!(b, :a); variable!(b, :c)
    factor!(b, :f, RelayFactor(:p, :q)); factor!(b, :g, RelayFactor(:p, :q))
    connect!(b, :f, :p, :a); connect!(b, :f, :q, :c)
    connect!(b, :g, :p, :a); connect!(b, :g, :q, :c)
    loopy = build(b)
    @test total_counting_number(loopy) == euler_characteristic(loopy) == 0   # 1 loop
end

@testset "free energy: graded, Bethe, and the chain form agree when deterministic" begin
    g = chain_graph()
    sched = tree_schedule(g)
    _, _, st, store = infer!(g, sched, NamedTuple(), NamedTuple())

    F, st = factor_free_energies(store, g, NamedTuple(), st)
    # E_G = ⊕_f E_f : the composite energy IS the per-factor breakdown
    @test sort(collect(keys(F))) == [:data_t, :data_x, :loss, :model]
    @test F[:data_x] == 0.0
    @test F[:model] == 0.0
    @test F[:loss] ≈ (2.0 - 5.0)^2 / 2          # y = 2 relayed, t = 5
    @test F[:loss] ≈ 4.5

    # every variable is a Dirac, so all counting corrections vanish
    V = variable_corrections(store, g)
    @test all(==(0.0), values(V.parts))

    B, st = bethe_free_energy(store, g, NamedTuple(), st)
    @test B[:factors][:loss] ≈ 4.5
    total, st = LenticulumCore.scalar_free_energy(store, g, NamedTuple(), st)
    @test total ≈ 4.5

    # AutoBayes Theorem 23 folded along an order agrees in the deterministic regime
    Fs = [F[n] for n in (:data_x, :model, :data_t, :loss)]
    chain = chain_free_energy(Fs)
    flat(x) = x isa LenticulumCore.GradedEnergy ? sum(flat, values(x.parts)) : x
    @test flat(chain) ≈ total
end

@testset "optimisers are Cruttwell's Definition 3.14 lenses" begin
    # gradient descent: get = p, put = p - η ḡ
    gd = GradientDescent(0.1)
    @test rule_get(gd, nothing, 1.0) == 1.0
    _, p = rule_put(gd, nothing, 1.0, 2.0)
    @test p ≈ 0.8

    # momentum: s' = -γ s - η ḡ ; p' = p + s'.  γ = 0 recovers gradient descent.
    m0 = Momentum(0.1, 0.0)
    s′, p′ = rule_put(m0, 0.0, 1.0, 2.0)
    @test p′ ≈ 0.8
    m = Momentum(0.1, 0.9)
    s1, p1 = rule_put(m, 0.0, 1.0, 2.0)
    @test s1 ≈ -0.2 && p1 ≈ 0.8
    s2, p2 = rule_put(m, s1, p1, 2.0)
    @test s2 ≈ -0.9 * (-0.2) - 0.2            # = -0.02
    @test p2 ≈ p1 + s2

    # Nesterov is the one with a NON-TRIVIAL get — the whole reason optimisers are lenses
    n = Nesterov(0.1, 0.9)
    @test rule_get(n, -0.2, 1.0) ≈ 1.0 + 0.9 * (-0.2)
    @test rule_get(n, -0.2, 1.0) != 1.0
    @test rule_get(gd, -0.2, 1.0) == 1.0

    # as a factor: non-learnable (it holds state, it does not have parameters)
    of = OptimiserFactor(:θ, gd)
    @test !LenticulumCore.islearnable(of)
    @test length(LenticulumCore.supported_polarities(of)) == 1
    @test LenticulumCore.isunidirectional(of)
end

@testset "factor traits" begin
    @test !LenticulumCore.islearnable(DataFactor(:x, 1.0))
    @test !LenticulumCore.islearnable(LossFactor((:a,), identity))
    @test !LenticulumCore.islearnable(RelayFactor(:a, :b))
    @test LenticulumCore.isunidirectional(DataFactor(:x, 1.0))     # emits only
    @test !LenticulumCore.isunidirectional(RelayFactor(:a, :b))    # two polarities
    @test issink(LossFactor((:a,), identity))                      # zero polarities
    @test !issink(DataFactor(:x, 1.0))
end

@testset "a multivariate loss factor carries a vector energy" begin
    lf = LossFactor((:a, :b), (a, b) -> [a - b, a + b];
                    scalarisation = LenticulumCore.SquaredNorm(),
                    energyspace = LenticulumCore.EuclideanEnergySpace(2))
    @test LenticulumCore.energyspace(lf) isa LenticulumCore.EuclideanEnergySpace
    @test !LenticulumCore.islinear(LenticulumCore.scalarisation(lf))
    e, _ = local_free_energy(lf, (; a = DiracBelief(3.0), b = DiracBelief(1.0)), nothing, nothing)
    @test e == [2.0, 4.0]
    @test LenticulumCore.scalarise(LenticulumCore.scalarisation(lf), e) ≈ (4 + 16) / 2
end

@testset "validation catches wiring errors" begin
    b = GraphBuilder()
    variable!(b, :x)
    factor!(b, :d, DataFactor(:x, 1.0))
    connect!(b, :d, :nosuchchannel, :x)
    @test_throws Mycelium.PolarityError validate(build(b))

    b2 = GraphBuilder()
    variable!(b2, :x); variable!(b2, :orphan)
    factor!(b2, :d, DataFactor(:x, 1.0))
    connect!(b2, :d, :x, :x)
    @test_throws ArgumentError validate(build(b2))

    b3 = GraphBuilder()
    variable!(b3, :x)
    factor!(b3, :d, DataFactor(:x, 1.0))
    connect!(b3, :d, :x, :x)
    @test_throws ArgumentError connect!(b3, :d, :x, :x)   # one channel, one port
end

end
