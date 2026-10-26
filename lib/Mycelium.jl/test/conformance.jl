include(joinpath(@__DIR__, "../../../test/factor_conformance.jl"))
using LuxCore, Random
@testset "Structural factor conformance" begin
    nt = (;)
    for f in (DataFactor(:x, 2.0), OptimiserFactor(:x, GradientDescent(0.1)))
        st = f isa OptimiserFactor ? (p = 2.0, state = 0.0) : nt
        opt = f isa OptimiserFactor
        check_factor_conformance(f, only(supported_polarities(f));
            π = TrivialBelief(), inputs = nt, ps = nt, st,
            pointwise = (2.0, nothing, nothing), game_prior = DiracBelief(2.0),
            game_y = nothing, msgs = (x = DiracBelief(2.0),), direct_message = true,
            expected_site = DiracBelief(2.0), expected_posterior = DiracBelief(2.0),
            pointwise_gap = opt ? "update rule has no pointwise energy" : nothing,
            game_gap = opt ? "update rule is not a statistical game" : nothing)
        if opt
            @test_throws ArgumentError free_energy(f, TrivialBelief(), nothing, nt, st)
            @test_throws ArgumentError LenticulumCore.entropy(f, TrivialBelief(), nothing, nt, st)
        else
            @test first(energy(f, 2.0, nothing, nothing, nt, st)) == 0
            @test first(free_energy(f, TrivialBelief(), nothing, nt, st)) == 0
            @test first(LenticulumCore.entropy(f, TrivialBelief(), nothing, nt, st)) == 0
        end
    end
    f = PriorFactor(:x, CategoricalBelief([0.25, 0.75]), x -> -log((0.25, 0.75)[x]))
    π = CategoricalBelief([0.6, 0.4])
    post = CategoricalBelief([1/3, 2/3])
    check_factor_conformance(f, only(supported_polarities(f)); π, inputs = nt, ps = nt,
        st = nt, pointwise = (2, nothing, nothing), game_prior = π, game_y = nothing,
        msgs = (x = π,), direct_message = true, expected_site = CategoricalBelief([0.25, 0.75]), expected_posterior = post)
    @test first(energy(f, 2, nothing, nothing, nt, nt)) ≈ -log(0.75)
    H = -sum(q * log(q) for q in (1/3, 2/3))
    @test first(free_energy(f, π, nothing, nt, nt)) ≈ -log(0.25)/3 - 2log(0.75)/3 - H
    @test first(LenticulumCore.entropy(f, π, nothing, nt, nt)) ≈ H
    # Also exercise the exact point game with option 1.
    @test first(free_energy(f, DiracBelief(2), nothing, nt, nt)) ≈ -log(0.75)
    @test first(free_energy(f, DiracBelief(2), nothing, nt, nt)) ==
          first(local_free_energy(f, (x = DiracBelief(2),), nt, nt))
    f = RelayFactor(:a, :b)
    for p in supported_polarities(f)
        source = only(observed_channels(p)); target = only(unobserved_channels(p))
        inputs = NamedTuple{(source,)}((DiracBelief(3.0),))
        check_factor_conformance(f, p; π = TrivialBelief(), inputs, ps = nt, st = nt,
            pointwise = (3.0, nothing, 3.0), game_prior = DiracBelief(3.0),
            game_y = DiracBelief(3.0), msgs = merge(inputs, NamedTuple{(target,)}((DiracBelief(3.0),))),
            direct_message = true, expected_site = DiracBelief(3.0), expected_posterior = DiracBelief(3.0))
        @test first(energy(f, 3.0, nothing, 3.0, nt, nt)) == 0
        @test first(free_energy(f, DiracBelief(3.0), DiracBelief(3.0), nt, nt)) == 0
        @test first(LenticulumCore.entropy(f, DiracBelief(3.0), DiracBelief(3.0), nt, nt)) == 0
    end
    # Terminal game: all channels are observed, no posterior variables or polarity.
    for f in (LossFactor((:a, :b), (a,b) -> (a-b)^2/2),
              LossFactor((:a, :b), (a,b) -> [a-b, a+b];
                         energyspace = EuclideanEnergySpace(2), scalarisation = SquaredNorm()))
        msgs = (a = DiracBelief(2.0), b = DiracBelief(5.0))
        check_factor_conformance(f, nothing; π = TrivialBelief(), inputs = (;), ps = nt, st = nt,
            game_prior = TrivialBelief(), game_y = msgs, msgs,
            pointwise_gap = "sink accepts named channel observations rather than an x/a/y split",
            message_gap = "terminal loss has no polarity or outgoing message")
        F = first(free_energy(f, TrivialBelief(), msgs, nt, nt))
        @test F ≈ (f.energyspace isa ScalarEnergySpace ? 4.5 : [-3.0, 7.0])
        @test first(LenticulumCore.entropy(f, TrivialBelief(), msgs, nt, nt)) == zero(F)
    end
end
