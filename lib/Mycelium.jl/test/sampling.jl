# Gibbs sampling with pins, and the sample buffer (`sampling.md`).
using Random

const SAMPLE_PRIOR = [0.2, 0.5, 0.3]

function prior_graph()
    b = GraphBuilder()
    variable!(b, :k, 3)
    factor!(b, :pk, PriorFactor(:k, LenticulumCore.CategoricalBelief(SAMPLE_PRIOR), k -> -log(SAMPLE_PRIOR[k])))
    connect!(b, :pk, :k, :k)
    return build(b)
end

@testset "pins: a pinned variable is a point mass for every reader" begin
    g = prior_graph()
    s = MessageStore(g)
    pin!(s, g, :k, 2)
    propagate!(s, g, tree_schedule(g), NamedTuple(), NamedTuple())
    @test marginal(s, g, 1) == DiracBelief(2)
    total, _ = LenticulumCore.scalar_free_energy(s, g, NamedTuple(), NamedTuple())
    @test total ≈ -log(SAMPLE_PRIOR[2])
    reset!(s)
    @test haskey(s.pins, 1)                       # reset! clears messages, keeps pins
    unpin!(s, g, :k)
    propagate!(s, g, tree_schedule(g), NamedTuple(), NamedTuple())
    @test LenticulumCore.probabilities(marginal(s, g, 1)) ≈ SAMPLE_PRIOR
end

@testset "sample_belief draws from the belief" begin
    rng = Xoshiro(1)
    c = LenticulumCore.CategoricalBelief(SAMPLE_PRIOR)
    f = [count(==(k), [sample_belief(rng, c) for _ in 1:20_000]) / 20_000 for k in 1:3]
    @test maximum(abs.(f - SAMPLE_PRIOR)) < 0.015
    x = [sample_belief(rng, LenticulumCore.Gaussian([1.0, -2.0], [1.0 0.5; 0.5 2.0])) for _ in 1:20_000]
    m = sum(x) / length(x)
    @test maximum(abs.(m - [1.0, -2.0])) < 0.05
    @test sample_belief(rng, DiracBelief(4.0)) == 4.0
    @test_throws ArgumentError sample_belief(rng, TrivialBelief())
end

@testset "the Gibbs sampler and the sample buffer" begin
    g = prior_graph()
    sampler = GibbsSampler(g, (:k,))
    rng = Xoshiro(2)
    state, _ = gibbs_init(sampler, NamedTuple(), NamedTuple(); rng)
    draws = Int[]
    for _ in 1:6000
        gibbs_sweep!(state, sampler, NamedTuple(), NamedTuple(); rng)
        push!(draws, gibbs_points(state, sampler).k)
    end
    @test maximum(abs.([count(==(k), draws) / length(draws) for k in 1:3] - SAMPLE_PRIOR)) < 0.03
    @test state.sweeps == 6000

    # a window keeps the last n samples, with equal weights
    buf = SampleBuffer(WindowFilter(3))
    for k in (1, 2, 3, 1, 2)
        record!(buf, sampler, (k = k,), NamedTuple(), NamedTuple())
    end
    @test length(buf) == 3
    @test sample_weights(buf) ≈ fill(1 / 3, 3)
    @test free_energy_mean(buf) ≈ -(log(SAMPLE_PRIOR[3]) + log(SAMPLE_PRIOR[1]) + log(SAMPLE_PRIOR[2])) / 3

    # a decay filter weighs a sample a draws old by λ^a, times its own weight
    buf = SampleBuffer(DecayFilter(0.5))
    for (k, w) in ((1, 1.0), (2, 2.0), (3, 1.0))
        record!(buf, sampler, (k = k,), NamedTuple(), NamedTuple(); weight = w)
    end
    w = [0.25 * 1.0, 0.5 * 2.0, 1.0 * 1.0]
    @test sample_weights(buf) ≈ w ./ sum(w)
    # energy moments: per graded component, the weighted mean and variance across samples
    em = energy_moments(buf)
    e = [-log(SAMPLE_PRIOR[k]) for k in 1:3]
    @test em.factors.pk.mean ≈ sum(w .* e) / sum(w)
    @test em.factors.pk.var ≈ sum(w .* (e .- sum(w .* e) / sum(w)) .^ 2) / sum(w)

    # forgetting: samples drawn more than `horizon` parameter steps ago are dropped
    buf = SampleBuffer(WindowFilter(10); horizon = 1)
    record!(buf, sampler, (k = 1,), NamedTuple(), NamedTuple())
    advance!(buf)
    record!(buf, sampler, (k = 2,), NamedTuple(), NamedTuple())
    @test length(buf) == 2
    advance!(buf)
    @test length(buf) == 1 && only(buf.entries).pins[1] == 2
    advance!(buf)
    @test length(buf) == 0
    @test_throws ArgumentError sample_weights(buf)

    # a factor without residual moments says so
    record!(buf, sampler, (k = 1,), NamedTuple(), NamedTuple())
    @test_throws ArgumentError residual_moments(buf, :pk)
    @test_throws ArgumentError record!(buf, sampler, (j = 1,), NamedTuple(), NamedTuple())
end
