# Per-node message rules: BeliefPropagation (the default) and MeanField (`passing.md`).

# A two-channel factor whose mean-field message records what it was given.
struct RecordingFactor <: AbstractLenticulumFactor
    seen::Vector{Any}
end
LenticulumCore.channels(::RecordingFactor) = (LenticulumCore.Channel(:a, nothing), LenticulumCore.Channel(:b, nothing))
LenticulumCore.supported_polarities(::RecordingFactor) =
    (Polarity(; a = Observed(), b = Unobserved()), Polarity(; a = Unobserved(), b = Observed()))
LenticulumCore.supports_polarity(f::RecordingFactor, p::Polarity) = p in LenticulumCore.supported_polarities(f)
function Mycelium.mean_field_message(f::RecordingFactor, target, polarity, beliefs, ps, st)
    push!(f.seen, (target, beliefs))
    return LenticulumCore.Gaussian([1.0], fill(1.0, 1, 1)), st
end

function recording_graph(rule)
    rec = RecordingFactor(Any[])
    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y)
    factor!(b, :px, PriorFactor(:x, LenticulumCore.Gaussian([0.0], fill(4.0, 1, 1)), _ -> 0.0))
    factor!(b, :py, PriorFactor(:y, LenticulumCore.Gaussian([0.0], fill(4.0, 1, 1)), _ -> 0.0))
    factor!(b, :rec, rec; rule)
    connect!(b, :px, :x, :x); connect!(b, :py, :y, :y)
    connect!(b, :rec, :a, :x); connect!(b, :rec, :b, :y)
    return build(b), rec
end

@testset "message rules: mean-field nodes read marginals, not cavities" begin
    g, _ = recording_graph(BeliefPropagation())
    @test factornode(g, :rec).rule isa BeliefPropagation
    @test factornode(g, :px).rule isa BeliefPropagation          # the default
    g, rec = recording_graph(MeanField())
    @test factornode(g, :rec).rule isa MeanField
    sched = SequentialSchedule(all_tasks(g))
    s = MessageStore(g)
    for _ in 1:2
        sweep!(s, g, sched, NamedTuple(), NamedTuple())
    end
    # by the second sweep :rec has sent to x, so x's marginal is the prior pooled with that
    # message (precision 1/4 + 1), while its cavity towards :rec would be the prior alone
    target, beliefs = last(filter(r -> r[1] === :b, rec.seen))
    @test keys(beliefs) == (:a,)
    @test LenticulumCore.belief_cov(beliefs.a)[1] ≈ 1 / (1 / 4 + 1)
    @test LenticulumCore.belief_cov(beliefs.a)[1] ≈ LenticulumCore.belief_cov(marginal(s, g, g.variable_index[:x]))[1]
    # the hybrid's free energy needs the node's `mean_field_energy`, which this factor lacks
    @test_throws ArgumentError bethe_free_energy(s, g, NamedTuple(), NamedTuple())
    @test_throws ArgumentError LenticulumCore.scalar_free_energy(s, g, NamedTuple(), NamedTuple())
    # a factor without a mean-field rule says so
    @test_throws ArgumentError Mycelium.mean_field_message(RelayFactor(:in, :out), :out, nothing,
                                                           (in = DiracBelief(2.0),), NamedTuple(), NamedTuple())
end

# ExpectationPropagationRule and MixtureReduction (`passing.md`, 2026-11-03). The end-to-end tests
# (clutter problem, switching chain) are in Lenticulum's `test/mixture_rules.jl`; these check the
# Mycelium side alone, with a prior factor emitting a fixed likelihood mixture.
@testset "ExpectationPropagationRule and MixtureReduction: Mycelium's side" begin
    G(μ, v) = LenticulumCore.Gaussian([μ], fill(v, 1, 1))
    # a likelihood mixture in x: exp(yx − x²/2) and a flat component, as unnormalised exponentials
    lik(y) = LenticulumCore.MixtureBelief(Val(:log),
        [LenticulumCore.GaussianBelief([y], fill(1.0, 1, 1), true), LenticulumCore.GaussianBelief([0.0], zeros(1, 1), true)],
        log.([0.7, 0.3]))
    function g(rule)
        b = GraphBuilder(); variable!(b, :x, 1)
        factor!(b, :p, PriorFactor(:x, G(0.0, 4.0), _ -> 0.0)); connect!(b, :p, :x, :x)
        for (i, y) in enumerate((1.0, 1.4, -0.3))
            factor!(b, Symbol(:l, i), PriorFactor(:x, lik(y), _ -> 0.0); rule); connect!(b, Symbol(:l, i), :x, :x)
        end
        return build(b)
    end
    # Belief propagation multiplies the likelihood mixtures (improper times improper) into 8 components.
    gb = g(BeliefPropagation())
    mb, rep, _, _ = infer!(gb, tree_schedule(gb), NamedTuple(), NamedTuple())
    @test rep.converged && rep.sweeps == 1 && length(mb.x.components) == 8
    # EP iterates (a tree sweep is not a fixed point) and ends with a Gaussian marginal whose
    # sites times their cavities have the tilted moments.
    ge = g(ExpectationPropagationRule())
    @test !Mycelium._tree_exact(ge) && Mycelium._tree_exact(gb) && Mycelium._tree_exact(g(MixtureReduction(2)))
    me, rep, _, s = infer!(ge, tree_schedule(ge), NamedTuple(), NamedTuple(); maxsweeps = 500, tol = 1e-12)
    @test rep.converged && rep.sweeps > 1 && me.x isa LenticulumCore.GaussianBelief
    for ei in 2:4
        cav = Mycelium.excluded_marginal(s, ge, 1, ei)
        tilted = Mycelium.combine(cav, lik((1.0, 1.4, -0.3)[ei - 1]))
        p = ep_project(tilted)
        q = Mycelium.combine(cav, s.to_variable[ei].belief)
        @test LenticulumCore.belief_mean(q) ≈ LenticulumCore.belief_mean(p) atol = 1e-10
        @test LenticulumCore.belief_cov(q) ≈ LenticulumCore.belief_cov(p) atol = 1e-10
    end
    @test Mycelium.bp_degree(ge, 1) == 4                     # EP edges count like belief propagation
    # MixtureReduction: Mycelium's default reduce_message passes mixtures within the budget and
    # refuses one above it (the Gaussian merge is Lenticulum's); mixture_message needs a method.
    l1 = lik(1.0)
    @test reduce_message(l1, 2) === l1
    @test_throws ArgumentError reduce_message(LenticulumCore.MixtureBelief([LenticulumCore.CategoricalBelief([0.5, 0.5]), LenticulumCore.CategoricalBelief([0.2, 0.8])], [0.5, 0.5]), 1)
    @test_throws ArgumentError mixture_message(PriorFactor(:x, G(0.0, 1.0), _ -> 0.0), :x, nothing, (x = lik(1.0),), nothing, NamedTuple(), NamedTuple())
    @test_throws ArgumentError MixtureReduction(0)
    @test_throws ArgumentError scalar_free_energy(MessageStore(g(MixtureReduction(2))), g(MixtureReduction(2)), NamedTuple(), NamedTuple())
    # ep_message's cases
    g1 = G(0.0, 1.0)
    @test ep_message(g1, G(1.0, 2.0)) === g1
    @test ep_message(l1, LenticulumCore.GaussianBelief([0.0], fill(-1.0, 1, 1), true)) === nothing
    @test ep_message(l1, DiracBelief([0.2])) === l1
    @test ep_divide(g1, TrivialBelief()) === g1
end
