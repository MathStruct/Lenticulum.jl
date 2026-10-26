include(joinpath(@__DIR__, "../../../test/factor_conformance.jl"))
@testset "Adversarial conformance" begin
    nt = (;)
    # Fixed draw: repeated lens and graph calls see identical particles, without RNG resets.
    f = NoiseSource(:z, 1; nsamples=1, dist=(rng,d,n)->[[2.0]])
    check_factor_conformance(f, only(supported_polarities(f)); π=TrivialBelief(), inputs=nt,
        ps=nt, st=nt, pointwise=(nothing,nothing,[2.0]), game_prior=DiracBelief([2.0]),
        game_y=nothing, msgs=(z=DiracBelief([2.0]),),
        expected_site=SampleBelief([[2.0]]), expected_posterior=SampleBelief([[2.0]]))
    @test first(energy(f, nothing, nothing, [2.0], nt, nt)) == 0
    @test first(free_energy(f, DiracBelief([2.0]), nothing, nt, nt)) == 0
    @test first(LenticulumCore.entropy(f, DiracBelief([2.0]), nothing, nt, nt)) == 0
    @test_throws ArgumentError free_energy(f, TrivialBelief(), nothing, nt, nt)
    @test_throws ArgumentError LenticulumCore.entropy(f, TrivialBelief(), nothing, nt, nt)
    cloud_source = NoiseSource(:z, 1; nsamples=2)
    @test_throws ArgumentError free_energy(cloud_source, DiracBelief([2.0]), nothing, nt, nt)
    @test_throws ArgumentError LenticulumCore.entropy(cloud_source, DiracBelief([2.0]), nothing, nt, nt)
    f = GeneratorFactor(LinearGen(fill(3.0,1,1), [1.0]), (z=1,x=1))
    ps, st = LuxCore.setup(Xoshiro(42), f)
    check_factor_conformance(f, only(supported_polarities(f)); π=TrivialBelief(),
        inputs=(z=DiracBelief([2.0]),), ps, st, pointwise=([2.0],nothing,[7.0]),
        game_prior=DiracBelief([7.0]), game_y=DiracBelief([2.0]),
        msgs=(z=DiracBelief([2.0]),x=DiracBelief([7.0])),
        expected_site=DiracBelief([7.0]), expected_posterior=DiracBelief([7.0]))
    @test first(free_energy(f, TrivialBelief(), DiracBelief([2.0]), ps, st)) == 0
    @test first(LenticulumCore.entropy(f, TrivialBelief(), DiracBelief([2.0]), ps, st)) == 0
    @test_throws ArgumentError free_energy(f, TrivialBelief(), SampleBelief([[2.0]]), ps, st)
    @test_throws ArgumentError LenticulumCore.entropy(f, TrivialBelief(), SampleBelief([[2.0]]), ps, st)
    # log(p/q) = 1/2 - x, for p=N(0,1), q=N(1,1).
    f = RatioFactor(AnalyticRatio(0.0,1.0,1.0,1.0), 1)
    ps, st = LuxCore.setup(Xoshiro(42), f)
    π = DiracBelief([2.0])
    check_factor_conformance(f, only(supported_polarities(f)); π, inputs=nt, ps, st,
        pointwise=([2.0],nothing,nothing), game_prior=π, game_y=nothing, msgs=(x=π,),
        expected_posterior=π)
    @test first(free_energy(f, π, nothing, ps, st)) ≈ 1.5
    @test first(LenticulumCore.entropy(f, π, nothing, ps, st)) == 0
    particles = SampleBelief([[0.0],[1.0]], [0.5,0.5])
    @test_throws ArgumentError free_energy(f, particles, nothing, ps, st)
    @test_throws ArgumentError LenticulumCore.entropy(f, particles, nothing, ps, st)
    post, _ = invert(first(assemble(f, only(supported_polarities(f)), ps, st)), particles, nt, ps, st)
    @test post.weights ≈ [exp(1)/(1+exp(1)), 1/(1+exp(1))]
    # LP-016: the likelihood is independent of both the cavity and target input.
    site, _ = factor_message(f, :x, only(supported_polarities(f)), nt, particles, ps, st)
    @test site isa AbstractPotentialBelief
    @test [belief_logdensity(site, [x]) for x in (0.0, 1.0)] ≈ [0.5, -0.5]
    @test belief_distance(combine(particles, site), post) <= 1e-8
    overridden, _ = factor_message(f, :x, only(supported_polarities(f)), (x=DiracBelief([9.0]),), particles, ps, st)
    @test [belief_logdensity(overridden, [x]) for x in (0.0, 1.0)] ≈ [0.5, -0.5]
    without_prior, _ = factor_message(f, :x, only(supported_polarities(f)), nt, nothing, ps, st)
    @test belief_logdensity(without_prior, [2.0]) ≈ -1.5
    lens = first(assemble(f, only(supported_polarities(f)), ps, st))
    likelihood, _ = likelihood_message(lens, nothing, (x=DiracBelief([9.0]),), ps, st)
    @test likelihood isa AbstractPotentialBelief
    @test belief_logdensity(likelihood, [1.0]) ≈ -0.5
    @test first(invert(lens, particles, (x=DiracBelief([9.0]),), ps, st)).weights ≈ post.weights
    posterior_F = first(local_free_energy(f, (x=particles,), ps, st))
    @test posterior_F ≈ 1/(1+exp(1)) - 0.5
    @test posterior_F ≈ sum(post.weights .* [-0.5,0.5])
    @test first(local_free_energy(f, (x=SampleBelief([[0.0], [1.0]], [1.0, 0.0]),), ps, st)) ≈ -0.5
    @test_throws ArgumentError local_free_energy(f, (x=Gaussian([0.0], ones(1,1)),), ps, st)
end

@testset "two ratio factors apply their potentials once" begin
    particles = SampleBelief([[0.0], [1.0]], [0.5, 0.5])
    r1 = RatioFactor(AnalyticRatio(0.0, 1.0, 1.0, 1.0), 1) # 1/2 - x
    r2 = RatioFactor(AnalyticRatio(0.0, 1.0, 2.0, 1.0), 1) # 2 - 2x
    # Exercise both fold orders: particles first, and both potentials before particles.
    for order in ((:prior, :r1, :r2), (:r2, :r1, :prior))
        b = GraphBuilder()
        variable!(b, :x)
        factors = (prior=PriorFactor(:x, particles, x -> 0.0), r1=r1, r2=r2)
        for name in order
            factor!(b, name, getproperty(factors, name))
            connect!(b, name, :x, :x)
        end
        g = build(b)
        ps, st = LuxCore.setup(Xoshiro(42), g)
        store = MessageStore(g)
        _, st = propagate!(store, g, tree_schedule(g), ps, st)
        marginal_b = marginal(store, g, 1)
        @test marginal_b.samples == particles.samples
        @test marginal_b.weights ≈ [exp(3)/(1+exp(3)), 1/(1+exp(3))]
        @test marginal_b.weights ≈ first(reweight(r2, first(reweight(r1, particles, ps.r1, st.r1)), ps.r2, st.r2)).weights
        # A flooding schedule also handles potential residuals/damping without a crash.
        report, _ = propagate!(MessageStore(g), g, flooding_schedule(g), ps, st; maxsweeps=8, damping=0.2)
        @test report.converged
        # Graph free energy receives excluded variable → factor messages.
        for (name, expected_F) in ((:r1, 1/(1+exp(3)) - 0.5), (:r2, 2/(1+exp(3)) - 2))
            fid = findfirst(f -> f.name === name, g.factors)
            msgs = Mycelium.messages_into(store, g, fid)
            F, _ = local_free_energy(getproperty(factors, name), msgs, getproperty(ps, name), getproperty(st, name))
            @test F ≈ expected_F
        end
    end
end
