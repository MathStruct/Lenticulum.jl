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
