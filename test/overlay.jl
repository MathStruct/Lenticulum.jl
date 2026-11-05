# OverlayFactor (`lib/Mycelium.jl/src/overlay.jl`, RQ-008): several factors on the same variables
# as one node. Oracles: a cluster node of the members as separate nodes (unnormalised), and the
# closed forms of a product of Gaussian experts and of a mixture of experts (normalised).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using ADTypes, DifferentiationInterface, ForwardDiff
using LenticulumCore: DiracBelief, MixtureBelief, CategoricalBelief

ov_lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
ov_logN(y, m, v) = -(y - m)^2 / (2v) - log(2π * v) / 2
const OV_LIN = (A = (y = ones(1, 1), x = fill(-1.5, 1, 1)), c = [0.2])          # y = 1.5x + 0.2 + ε
const OV_SW = (A = (y = ones(1, 1),), b = [-1.0 1.0], logw = log.([0.3, 0.7]))   # y = b_k + ε

@testset "OverlayFactor unnormalised: the product, as a cluster of the members" begin
    lin = LinearConstraintFactor((y = 1, x = 1), 1; noise = 0.3)
    sw = SwitchingConstraintFactor((y = 1,), 2, 1; noise = 0.2)
    function graph(overlaid)
        b = GraphBuilder()
        variable!(b, :y, 1); variable!(b, :x, 1); variable!(b, :k, 2)
        factor!(b, :d, DataFactor(:y, [0.9])); connect!(b, :d, :y, :y; direction = Emitting())
        factor!(b, :pr, GaussianPrior(:x, 0.0, 1.0)); connect!(b, :pr, :x, :x)
        if overlaid
            factor!(b, :ov, OverlayFactor(:lin => lin, :sw => sw))
            connect!(b, :ov, :y, :y); connect!(b, :ov, :x, :x); connect!(b, :ov, :k, :k)
            return cluster(build(b), :c, [:pr, :ov])
        end
        factor!(b, :lin, lin); connect!(b, :lin, :y, :y); connect!(b, :lin, :x, :x)
        factor!(b, :sw, sw); connect!(b, :sw, :y, :y); connect!(b, :sw, :k, :k)
        return cluster(build(b), :c, [:pr, :lin, :sw])
    end
    go, gs = graph(true), graph(false)
    pso = cluster_parameters(go, :c, (d = NamedTuple(), pr = NamedTuple(), ov = (lin = OV_LIN, sw = OV_SW)))
    pss = cluster_parameters(gs, :c, (d = NamedTuple(), pr = NamedTuple(), lin = OV_LIN, sw = OV_SW))
    _, _, _, so = infer!(go, tree_schedule(go), pso, NamedTuple())
    _, _, _, ss = infer!(gs, tree_schedule(gs), pss, NamedTuple())
    @test first(scalar_free_energy(so, go, pso, NamedTuple())) ≈ first(scalar_free_energy(ss, gs, pss, NamedTuple())) rtol = 1e-12
    @test exp.(cluster_marginal(so, go, :c, :k, pso).logp) ≈ exp.(cluster_marginal(ss, gs, :c, :k, pss).logp) atol = 1e-12
    # outside a cluster: its message to the outcome given y and x
    o = OverlayFactor(:lin => lin, :sw => sw)
    @test Set(LenticulumCore.channelname.(LenticulumCore.channels(o))) == Set([:y, :x, :k])
    mk, _ = factor_message(o, :k, nothing, (y = DiracBelief([0.9]), x = DiracBelief([0.4])), nothing, (lin = OV_LIN, sw = OV_SW), NamedTuple())
    lk = [log(w) + ov_logN(0.9, b, 0.2) for (w, b) in zip([0.3, 0.7], [-1.0, 1.0])]
    @test exp.(mk.logp) ≈ exp.(lk .- ov_lse(lk)) atol = 1e-12
end

@testset "OverlayFactor normalised: products and mixtures of experts" begin
    # two linear experts on y given x: N(y; (m₁/q₁ + m₂/q₂)/P, 1/P), P = 1/q₁ + 1/q₂
    e1 = LinearConstraintFactor((y = 1, x = 1), 1; noise = 0.3)
    e2 = LinearConstraintFactor((y = 1, x = 1), 1; noise = 0.5)
    p2 = (A = (y = ones(1, 1), x = fill(0.5, 1, 1)), c = [-0.4])                  # y = −0.5x − 0.4
    o = OverlayFactor(:e1 => e1, :e2 => e2; normalise = :y)
    ps = (e1 = OV_LIN, e2 = p2)
    x, y = 0.4, 0.9
    m1, m2 = 1.5x + 0.2, -0.5x - 0.4
    P = 1 / 0.3 + 1 / 0.5
    F(o, ps, ins) = (b = Mycelium.factor_belief(o, ins, ps, NamedTuple());
                     LenticulumCore.scalarise(LenticulumCore.scalarisation(o), first(Mycelium.belief_free_energy(o, b, ps, NamedTuple()))))
    ins = (y = DiracBelief([y]), x = DiracBelief([x]))
    @test F(o, ps, ins) ≈ -ov_logN(y, (m1 / 0.3 + m2 / 0.5) / P, 1 / P) rtol = 1e-12
    # a linear expert and a switch on y: a mixture of experts Σ w_k N(y; m_k, 1/P), P = 1/q + 1/q_s
    sw = SwitchingConstraintFactor((y = 1,), 2, 1; noise = 0.2)
    om = OverlayFactor(:lin => e1, :sw => sw; normalise = :y)
    psm = (lin = OV_LIN, sw = OV_SW)
    Pm = 1 / 0.3 + 1 / 0.2
    mix = [log(w) + ov_logN(y, (m1 / 0.3 + b / 0.2) / Pm, 1 / Pm) for (w, b) in zip([0.3, 0.7], [-1.0, 1.0])]
    @test F(om, psm, ins) ≈ -ov_lse(mix) rtol = 1e-12
    # members that do not involve y cancel: a prior on x leaves the conditional of y unchanged
    opr = OverlayFactor(:e1 => e1, :e2 => e2, :px => GaussianPrior(:x, 0.0, 1.0); normalise = :y)
    @test F(opr, (e1 = OV_LIN, e2 = p2, px = NamedTuple()), ins) ≈ F(o, ps, ins) rtol = 1e-12
    @test_throws ArgumentError OverlayFactor(:sw => sw; normalise = :k)                  # a discrete channel
end

@testset "OverlayFactor normalised: a learned relation on data does not collapse (LP-022)" begin
    rng = Xoshiro(22)
    N = 100
    xs = randn(rng, N); ys = 2 .* xs .+ 0.5 .+ 0.3 .* randn(rng, N)
    rel = LinearConstraintFactor((y = 1, x = 1), 1; noise = 1.0)                       # every coefficient learned
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 1); variable!(b, r(:y), 1)
        factor!(b, r(:dx), DataFactor(:x, [xs[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
        factor!(b, r(:dy), DataFactor(:y, [ys[i]])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
        factor!(b, r(:ov), OverlayFactor(:rel => rel; normalise = :y); params = :ov)
        connect!(b, r(:ov), :x, r(:x)); connect!(b, r(:ov), :y, r(:y))
    end
    g = build(b)
    @test isempty(Mycelium._unnormalised_nodes(g))                                       # no warning: it is normalised
    data = NamedTuple(s => NamedTuple() for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i)))
    tasks = [t for t in all_tasks(g) if !(t.kind === :to_variable && g.factors[g.edges[t.edge].factor].factor isa OverlayFactor)]
    function F(θ)
        local ps = merge(data, (ov = (rel = (A = (y = fill(θ[1], 1, 1), x = fill(θ[2], 1, 1)), c = [θ[3]]),),))
        local s = MessageStore(g)
        sweep!(s, g, SequentialSchedule(tasks), ps, NamedTuple())
        return first(scalar_free_energy(s, g, ps, NamedTuple())), s, ps
    end
    @test_throws ArgumentError F([0.0, 0.0, 0.0])                                        # A = 0 is not a density of y
    # the normalised relation is N(y; −(A_x x − c)/A_y, Q/A_y²): ML is least squares and its variance
    D = [xs ones(N)]; a, c0 = D \ ys
    v = sum((ys .- a .* xs .- c0) .^ 2) / N
    θml = [1 / sqrt(v), -a / sqrt(v), c0 / sqrt(v)]
    Fml = first(F(θml))
    @test Fml ≈ -sum(ov_logN(ys[i], a * xs[i] + c0, v) for i in 1:N) rtol = 1e-10
    @test Fml < first(F(1.05 .* θml)) && Fml < first(F(0.95 .* θml)) && Fml < first(F(θml .+ [0, 0.05, 0]))
    _, s, ps = F(θml)
    G = free_energy_gradient(s, g, ps, NamedTuple(); ad = AutoForwardDiff()).ov.rel
    @test maximum(abs, [only(G.A.y), only(G.A.x), only(G.c)]) < 1e-8 * N                # stationary at maximum likelihood
end
