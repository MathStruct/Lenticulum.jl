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
        expected_site=π, expected_posterior=π)
    @test first(free_energy(f, π, nothing, ps, st)) ≈ 1.5
    @test first(LenticulumCore.entropy(f, π, nothing, ps, st)) == 0
    particles = SampleBelief([[0.0],[1.0]], [0.5,0.5])
    @test_throws ArgumentError free_energy(f, particles, nothing, ps, st)
    @test_throws ArgumentError LenticulumCore.entropy(f, particles, nothing, ps, st)
    post, _ = invert(first(assemble(f, only(supported_polarities(f)), ps, st)), particles, nt, ps, st)
    @test post.weights ≈ [exp(1)/(1+exp(1)), 1/(1+exp(1))]
    # LP-016: emitted particle message already includes the cavity. Pooling two
    # SampleBeliefs throws; exclusion also fails when the target is supplied in inputs.
    site, _ = factor_message(f, :x, only(supported_polarities(f)), nt, particles, ps, st)
    @test_broken belief_distance(combine(particles, site), post) <= 1e-8
    overridden, _ = factor_message(f, :x, only(supported_polarities(f)), (x=DiracBelief([9.0]),), particles, ps, st)
    @test_broken belief_distance(site, overridden) <= 1e-8
    incoming_F = first(local_free_energy(f, (x=particles,), ps, st))
    @test incoming_F ≈ 0.0
    @test_broken isapprox(incoming_F, sum(post.weights .* [-0.5,0.5]))
end
