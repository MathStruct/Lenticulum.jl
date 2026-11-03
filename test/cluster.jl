# Cluster nodes (Extending Mycelium §5, L-SPEC-009): exact inference inside, messages outside.
# Oracles: exact enumeration of a switching chain (marginals and log p(y)), the hand-built block
# factor of E7 (a switching factor with 2^B outcomes), and the factors themselves.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote
using LenticulumCore: MixtureBelief, CategoricalBelief, DiracBelief, Unobserved, Observed, Polarity
isdefined(@__MODULE__, :mr_chain) || include(joinpath(@__DIR__, "mixture_rules.jl"))

# −log p(y) of the switching chain of `mixture_rules.jl`, enumerating the switch sequences:
# given them y is Gaussian, mean m_s and covariance V + rI with V_ij = v₀ + (min(i, j) − 1) q.
function cl_chain_nlp(offs = MR_CHAIN.offs)
    c, T = MR_CHAIN, MR_T
    V = [c.v0 + (min(i, j) - 1) * c.q for i in 1:T, j in 1:T]
    C = V + c.r * I
    lps = map(vec(collect(Iterators.product(ntuple(_ -> 1:2, T - 1)...)))) do seq
        m = cumsum([c.m0; [offs[k] for k in seq]])
        sum(log(c.wk[k]) for k in seq) - (dot(c.ys - m, C \ (c.ys - m)) + logdet(2π * C)) / 2
    end
    return -mr_lse(lps)
end

@testset "A cluster of every factor is exact" begin
    g = mr_chain(BeliefPropagation())
    gc = cluster(g, :all, [f.name for f in g.factors if !(f.factor isa DataFactor)])
    @test nfactors(gc) == MR_T + 1 && nvariables(gc) == MR_T          # the data, one cluster, the y's
    psc = cluster_parameters(gc, :all, MR_CHAIN_PS)
    @test keys(psc.all) == (:obs, :prior, :dyn)                        # tied factors keep one entry each
    _, rep, _, s = infer!(gc, tree_schedule(gc), psc, NamedTuple())
    @test rep.converged
    F = first(scalar_free_energy(s, gc, psc, NamedTuple()))
    @test F ≈ cl_chain_nlp() rtol = 1e-12                              # the free energy is −log p(y)
    μe, ve = mr_chain_exact()
    for t in 1:MR_T
        q = moment_match(cluster_marginal(s, gc, :all, Symbol(:x, t), psc))
        @test only(belief_mean(q)) ≈ μe[t] atol = 1e-12
        @test only(belief_cov(q)) ≈ ve[t] atol = 1e-12
    end
    k = cluster_marginal(s, gc, :all, :k1, psc)
    @test k isa CategoricalBelief && sum(exp, k.logp) ≈ 1
    # the gradient at the exact belief is the gradient of −log p(y), in the tied offsets
    h = 1e-6
    fd = [(cl_chain_nlp(MR_CHAIN.offs .+ h .* (1:2 .== j)) - cl_chain_nlp(MR_CHAIN.offs .- h .* (1:2 .== j))) / 2h for j in 1:2]
    for (ad, tol) in ((nothing, 1e-6), (AutoForwardDiff(), 1e-7), (AutoZygote(), 1e-7))
        G = free_energy_gradient(s, gc, psc, NamedTuple(); ad)
        @test vec(G.all.dyn.b) ≈ fd atol = tol
    end
end

# E7's additive wiring: z = u₁ + u₂ + u₃ + ε observed, each uₗ from a two-outcome switching factor.
const CL_OFFS = [[-1.0 1.0], [0.0 2.0], [-2.0 0.5]]
const CL_W = [[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]]
function cl_additive(; block = false)
    b = GraphBuilder()
    variable!(b, :z, 1)
    factor!(b, :data, DataFactor(:z, [1.3])); connect!(b, :data, :z, :z; direction = Emitting())
    factor!(b, :sum, LinearConstraintFactor((z = 1, u1 = 1, u2 = 1, u3 = 1), 1; noise = 0.2))
    connect!(b, :sum, :z, :z)
    for l in 1:3
        variable!(b, Symbol(:u, l), 1); connect!(b, :sum, Symbol(:u, l), Symbol(:u, l))
    end
    if block
        variable!(b, :kb, 8)
        factor!(b, :blk, SwitchingConstraintFactor((u1 = 1, u2 = 1, u3 = 1), 8, 3; noise = Diagonal([0.05, 0.1, 0.15]) |> Matrix, switch = :k))
        for l in 1:3; connect!(b, :blk, Symbol(:u, l), Symbol(:u, l)); end
        connect!(b, :blk, :k, :kb)
    else
        for l in 1:3
            variable!(b, Symbol(:k, l), 2)
            factor!(b, Symbol(:sw, l), SwitchingConstraintFactor((u = 1,), 2, 1; noise = 0.05 * l))
            connect!(b, Symbol(:sw, l), :u, Symbol(:u, l)); connect!(b, Symbol(:sw, l), :k, Symbol(:k, l))
        end
    end
    return build(b)
end
const CL_SUM = (A = (z = fill(1.0, 1, 1), u1 = fill(-1.0, 1, 1), u2 = fill(-1.0, 1, 1), u3 = fill(-1.0, 1, 1)), c = [0.0])
cl_sw(l) = (A = (u = fill(1.0, 1, 1),), b = CL_OFFS[l], logw = log.(CL_W[l]))
# the block factor: outcome (k₁, k₂, k₃) with k₁ fastest, offsets stacked, weights multiplied
function cl_block()
    combos = vec(collect(Iterators.product(1:2, 1:2, 1:2)))
    A = (u1 = reshape([1.0, 0, 0], 3, 1), u2 = reshape([0, 1.0, 0], 3, 1), u3 = reshape([0, 0, 1.0], 3, 1))
    B = reduce(hcat, [[CL_OFFS[l][ks[l]] for l in 1:3] for ks in combos])
    return (A = A, b = B, logw = [sum(log(CL_W[l][ks[l]]) for l in 1:3) for ks in combos])
end

@testset "A cluster of switches is the block factor" begin
    g, gb = cl_additive(), cl_additive(; block = true)
    gc = cluster(g, :blk, [:sw1, :sw2, :sw3])
    ps = (data = NamedTuple(), sum = CL_SUM, sw1 = cl_sw(1), sw2 = cl_sw(2), sw3 = cl_sw(3))
    psc = cluster_parameters(gc, :blk, ps)
    psb = (data = NamedTuple(), sum = CL_SUM, blk = cl_block())
    @test Set(gc.factors[gc.factor_index[:blk]].factor.boundary) == Set([:u1, :u2, :u3])
    # one message, against the block factor's, at Gaussian inputs
    c = gc.factors[gc.factor_index[:blk]].factor
    bf = gb.factors[gb.factor_index[:blk]].factor
    ins = (u2 = Gaussian(0.3, 0.5), u3 = Gaussian(-0.4, 2.0))
    mc, _ = factor_message(c, :u1, nothing, ins, nothing, psc.blk, NamedTuple())
    mb, _ = factor_message(bf, :u1, nothing, ins, nothing, psb.blk, NamedTuple())
    for x in ([-1.0], [0.4], [2.0])
        @test Mycelium.belief_logdensity(mc, x) ≈ Mycelium.belief_logdensity(mb, x) atol = 1e-12
    end
    # the same loopy schedule on both graphs: the same marginals and free energy
    sc, sb = MessageStore(gc), MessageStore(gb)
    for _ in 1:15
        sweep!(sc, gc, SequentialSchedule(all_tasks(gc)), psc, NamedTuple())
        sweep!(sb, gb, SequentialSchedule(all_tasks(gb)), psb, NamedTuple())
    end
    for v in (:u1, :u2, :u3)                                            # z is observed
        qc, qb = moment_match(marginal(sc, gc, gc.variable_index[v])), moment_match(marginal(sb, gb, gb.variable_index[v]))
        @test belief_mean(qc) ≈ belief_mean(qb) atol = 1e-10
        @test belief_cov(qc) ≈ belief_cov(qb) atol = 1e-10
    end
    # The uₗ have mixture marginals, so these graphs have no Bethe free energy (no mixture entropy);
    # the nodes' local free energies at their messages agree, term by term.
    Fc, _ = local_free_energy(c, messages_into(sc, gc, gc.factor_index[:blk]), psc.blk, NamedTuple())
    Fb, _ = local_free_energy(bf, messages_into(sb, gb, gb.factor_index[:blk]), psb.blk, NamedTuple())
    for part in (:fit, :complexity, :negentropy)
        @test Fc[part] ≈ Fb[part] rtol = 1e-10
    end
    # a cluster of one factor is that factor
    g1 = cluster(g, :one, [:sw1])
    ps1 = cluster_parameters(g1, :one, ps)
    s0, s1 = MessageStore(g), MessageStore(g1)
    for _ in 1:10
        sweep!(s0, g, SequentialSchedule(all_tasks(g)), ps, NamedTuple())
        sweep!(s1, g1, SequentialSchedule(all_tasks(g1)), ps1, NamedTuple())
    end
    q0, q1 = moment_match(marginal(s0, g, g.variable_index[:u1])), moment_match(marginal(s1, g1, g1.variable_index[:u1]))
    @test belief_mean(q0) ≈ belief_mean(q1) atol = 1e-10
    F0, _ = local_free_energy(g.factors[g.factor_index[:sw1]].factor, messages_into(s0, g, g.factor_index[:sw1]), ps.sw1, NamedTuple())
    F1, _ = local_free_energy(g1.factors[g1.factor_index[:one]].factor, messages_into(s1, g1, g1.factor_index[:one]), ps1.one, NamedTuple())
    for part in (:fit, :complexity, :negentropy)
        @test F0[part] ≈ F1[part] rtol = 1e-10
    end
end

@testset "A cluster closes the mixture-entropy gap" begin
    # On belief propagation the uₗ have mixture marginals and the graph no free energy; with the
    # switches and the sum in one cluster they are internal, and the free energy is −log p(z).
    g = cl_additive()
    ps = (data = NamedTuple(), sum = CL_SUM, sw1 = cl_sw(1), sw2 = cl_sw(2), sw3 = cl_sw(3))
    s = MessageStore(g)
    for _ in 1:5; sweep!(s, g, SequentialSchedule(all_tasks(g)), ps, NamedTuple()); end
    @test_throws ArgumentError scalar_free_energy(s, g, ps, NamedTuple())
    gc = cluster(g, :all, [:sum, :sw1, :sw2, :sw3])
    psc = cluster_parameters(gc, :all, ps)
    _, _, _, sc = infer!(gc, tree_schedule(gc), psc, NamedTuple())
    lp = map(vec(collect(Iterators.product(1:2, 1:2, 1:2)))) do ks
        sum(log(CL_W[l][ks[l]]) for l in 1:3) + mr_lognorm(1.3, sum(CL_OFFS[l][ks[l]] for l in 1:3), 0.2 + 0.05 * 6)
    end
    @test first(scalar_free_energy(sc, gc, psc, NamedTuple())) ≈ -mr_lse(lp) rtol = 1e-12
end

# E7's grouping (examples/hierarchy_inference.jl on experiment-qm), verbatim, as the oracle.
function cl_e7_overlap_blocks(G, B)
    L = size(G, 1)
    blocks = [[l] for l in 1:L]
    c = [abs(G[l, m]) / sqrt(G[l, l] * G[m, m]) for l in 1:L, m in 1:L]
    while true
        best, pair = 0.0, nothing
        for a in eachindex(blocks), b in a+1:length(blocks)
            length(blocks[a]) + length(blocks[b]) <= B || continue
            s = maximum(c[l, m] for l in blocks[a] for m in blocks[b])
            s > best && ((best, pair) = (s, (a, b)))
        end
        pair === nothing && break
        a, b = pair
        blocks[a] = sort(vcat(blocks[a], blocks[b])); deleteat!(blocks, b)
    end
    return blocks
end

# The additive wiring z = Σₗ uₗ + ε, uₗ ∈ {0, Wₗ} + εₗ, in d dimensions with L switches.
function cl_additive_W(W, Σ, ε, x)
    d, L = size(W)
    b = GraphBuilder(); variable!(b, :z, d)
    factor!(b, :data, DataFactor(:z, x)); connect!(b, :data, :z, :z; direction = Emitting())
    dims = NamedTuple{(:z, (Symbol(:u, l) for l in 1:L)...)}(ntuple(_ -> d, L + 1))
    factor!(b, :sum, LinearConstraintFactor(dims, d; noise = Σ)); connect!(b, :sum, :z, :z)
    for l in 1:L
        u, k, sw = Symbol(:u, l), Symbol(:k, l), Symbol(:sw, l)
        variable!(b, u, d); variable!(b, k, 2)
        factor!(b, sw, SwitchingConstraintFactor((u = d,), 2, d; noise = ε))
        connect!(b, :sum, u, u); connect!(b, sw, :u, u); connect!(b, sw, :k, k)
    end
    g = build(b)
    E = Matrix(1.0I, d, d)
    ps = merge((data = NamedTuple(), sum = (A = NamedTuple{keys(dims)}((E, ntuple(_ -> -E, L)...)), c = zeros(d))),
               NamedTuple{Tuple(Symbol(:sw, l) for l in 1:L)}(Tuple((A = (u = E,), b = hcat(zeros(d), W[:, l]), logw = zeros(2)) for l in 1:L)))
    return g, ps
end

@testset "Grouping switches by overlap" begin
    rng = Xoshiro(11)
    d, L = 4, 6
    W = randn(rng, d, L); W[:, 4] .= W[:, 1] .+ 0.1 .* randn(rng, d); W[:, 6] .= -W[:, 2] .+ 0.2 .* randn(rng, d)
    Σ = [0.5 0.1 0 0; 0.1 0.4 0 0; 0 0 0.3 0; 0 0 0 0.6]
    g, ps = cl_additive_W(W, Σ, 1e-3, randn(rng, d))
    sws = [Symbol(:sw, l) for l in 1:L]
    O = switch_overlaps(g, ps, :sum, sws)
    R = Σ + L * 1e-3 * I
    G = W' * (R \ W)
    @test O ≈ [abs(G[l, m]) / sqrt(G[l, l] * G[m, m]) for l in 1:L, m in 1:L] atol = 1e-12
    for B in (2, 3, 4)
        @test overlap_groups(O, B) == cl_e7_overlap_blocks(G, B)
    end
    groups = overlap_groups(O, 3)
    @test any(gr -> 1 in gr && 4 in gr, groups) && any(gr -> 2 in gr && 6 in gr, groups)   # the planted pairs
    gc = cluster(g, [Symbol(:blk, i) => sws[gr] for (i, gr) in enumerate(groups)])
    @test count(f -> f.factor isa ClusterFactor, gc.factors) == length(groups)
    psc = cluster_parameters(gc, ps)
    @test Set(keys(psc)) == Set([:data, :sum, (Symbol(:blk, i) for i in eachindex(groups))...])
    # the blocks' messages run, and the overlap clusters are exact when there is one group
    gall = cluster(g, [:one => sws])
    s = MessageStore(gall)
    for _ in 1:5; sweep!(s, gall, SequentialSchedule(all_tasks(gall)), cluster_parameters(gall, ps), NamedTuple()); end
    @test marginal(s, gall, gall.variable_index[:u1]) isa MixtureBelief
end

@testset "partial_sums: E7's block wiring as an edit" begin
    rng = Xoshiro(12)
    d, L = 3, 5
    W = randn(rng, d, L); W[:, 3] .= W[:, 1] .+ 0.1 .* randn(rng, d)
    Σ = Matrix(0.2I, d, d)
    g, ps = cl_additive_W(W, Σ, 1e-3, randn(rng, d))
    sws = [Symbol(:sw, l) for l in 1:L]
    groups = [[1, 3], [2, 4, 5]]
    Fx(g, ps) = (s = MessageStore(g); sweep!(s, g, SequentialSchedule(all_tasks(g)), ps, NamedTuple());
                 first(scalar_free_energy(s, g, ps, NamedTuple())))
    ge = cluster(g, [:all => [:sum; sws]]); F0 = Fx(ge, cluster_parameters(ge, ps))
    ΔF = map((1e-4, 1e-6, 1e-8)) do ν
        g2, ps2, cl = partial_sums(g, ps, :sum, sws, groups; noise = ν)
        @test cl == [:part1_block => [:part1, :sw1, :sw3], :part2_block => [:part2, :sw2, :sw4, :sw5]]
        @test Set(keys(g2.factors[g2.factor_index[:sum]].factor.dims)) == Set([:z, :part1, :part2])
        @test keys(ps2.part2.A) == (:s, :u2, :u4, :u5) && ps2.part2.A.u4 == ps.sum.A.u4
        ge2 = cluster(g2, [:all => [:sum, :part1, :part2, sws...]])
        Fx(ge2, cluster_parameters(ge2, ps2)) - F0
    end
    # the model is unchanged as the partial sums' noise goes to 0 (the change is linear in it)
    @test abs(ΔF[3]) < 1e-5 && abs(ΔF[2] / ΔF[3] - 100) < 1
    # the returned groups cluster directly, and a singleton group is left alone
    g3, ps3, cl3 = partial_sums(g, ps, :sum, sws, [[1, 3], [2], [4, 5]])
    @test first.(cl3) == [:part1_block, :part3_block] && !haskey(g3.factor_index, :part2)
    gc3 = cluster(g3, cl3); s = MessageStore(gc3)
    for _ in 1:5; sweep!(s, gc3, SequentialSchedule(all_tasks(gc3)), cluster_parameters(gc3, ps3), NamedTuple()); end
    # (the unclustered switch's mixture variable has no free energy, so the marginals are checked)
    @test all(isfinite, exp.(cluster_marginal(s, gc3, :part3_block, :k4, cluster_parameters(gc3, ps3)).logp))
    @test all(isfinite, exp.(marginal(s, gc3, gc3.variable_index[:k2]).logp))
    @test_throws ArgumentError partial_sums(g, ps, :sum, sws, [[1, 2], [2, 3]])   # overlapping groups
    @test_throws ArgumentError partial_sums(g, ps, :data, sws, groups)            # not a linear factor
    @test_throws ArgumentError partial_sums(g, ps, :sum, [:sum], [[1]])           # not a switch
end

@testset "cluster: refusals" begin
    g = cl_additive()
    @test_throws ArgumentError cluster(g, :c, [:data, :sw1])            # DataFactor has no cluster_form
    @test_throws ArgumentError cluster(g, :c, [:nope])
    @test_throws ArgumentError cluster(g, :sum, [:sw1])                 # the name is taken
    gt = mr_chain(BeliefPropagation())
    @test_throws ArgumentError cluster(gt, :c, [:o1, :o2])              # :obs is tied to o3…o5 outside
    @test_throws ArgumentError cluster(gt, [:c1 => [:o1, :o2, :o3], :c2 => [:o4, :o5]])   # tied across clusters without params
    @test_throws ArgumentError cluster(g, [:c1 => [:sw1, :sw2], :c2 => [:sw2]])        # overlapping groups
    @test_throws ArgumentError cluster(g, [:c1 => [:sw1, :sw2]]; params = :sum)        # a key an outside node reads
    b = GraphBuilder()
    plate!((p, i) -> (variable!(p, :x, 1); factor!(p, :pr, GaussianPrior(:x, 0.0, 1.0)); connect!(p, :pr, :x, :x)), b, :pl, 2)
    @test_throws ArgumentError cluster(build(b), :c, [Symbol("pr[1]")])
    # a channel left unconnected inside a cluster
    b2 = GraphBuilder(); variable!(b2, :u, 1)
    factor!(b2, :lc, LinearConstraintFactor((u = 1, w = 1), 1; noise = 1.0)); connect!(b2, :lc, :u, :u)
    @test_throws ArgumentError cluster(build(b2), :c, [:lc])
end
