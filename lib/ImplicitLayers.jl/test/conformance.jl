include(joinpath(@__DIR__, "../../../test/factor_conformance.jl"))
# Doubler uses a scalar parameter; LuxCore default counting handles arrays only.
LuxCore.parameterlength(::Doubler) = 1

@testset "Point factor conformance and LP-009 conflict" begin
    nt = (;)
    f = LuxFactor(Doubler(), :x => :y)
    ps, st = LuxCore.setup(Xoshiro(42), f)
    check_factor_conformance(f, only(supported_polarities(f)); π = TrivialBelief(),
        inputs = (x = DiracBelief([3.0]),), ps, st, pointwise = ([3.0], nothing, [6.0]),
        game_prior = DiracBelief([6.0]), game_y = DiracBelief([3.0]),
        msgs = (x = DiracBelief([3.0]), y = DiracBelief([6.0])),
        expected_site = DiracBelief([6.0]), expected_posterior = DiracBelief([6.0]))
    @test first(free_energy(f, TrivialBelief(), DiracBelief([3.0]), ps, st)) == 0
    @test first(LenticulumCore.entropy(f, TrivialBelief(), DiracBelief([3.0]), ps, st)) == 0
    @test_throws ArgumentError free_energy(f, TrivialBelief(), nothing, ps, st)
    @test_throws ArgumentError LenticulumCore.entropy(f, TrivialBelief(), nothing, ps, st)
    # g(z,x)=z/2+x: z*=2x; dz/dt=0: z1=z0. Both give exact, nonzero
    # incoming residuals but zero solved residuals, so a solved-point game disagrees.
    deq = DEQFactor(LinearCell(fill(0.5,1,1), ones(1,1), zeros(1)), (x=1,z=1))
    # A latent companion gives no point to solve from; only O/U splits are supported.
    @test !supports_polarity(deq, Polarity((x=LenticulumCore.Latent(),z=Unobserved())))
    @test !supports_polarity(deq, Polarity((x=Unobserved(),z=LenticulumCore.Latent())))
    ode = NeuralODEFactor(LinearDynamics(zeros(1,1)), 1; tspan=(0.0,1.0))
    for f in (deq, ode)
        ps, st = LuxCore.setup(Xoshiro(42), f)
        names = Tuple(channelname(c) for c in LenticulumCore.channels(f))
        xy = ([1.0], nothing, [3.0])
        expected_energy = f isa DEQFactor ? 0.125 : 2.0
        @test first(scalar_energy(f, xy..., ps, st)) ≈ expected_energy
        for p in supported_polarities(f)
            src = only(observed_channels(p)); dst = only(unobserved_channels(p))
            v = src == names[1] ? [1.0] : [3.0]
            solved = f isa DEQFactor ? (dst == names[2] ? [2.0] : [1.5]) : v
            inputs = NamedTuple{(src,)}((DiracBelief(v),))
            msgs = NamedTuple{names}((DiracBelief([1.0]), DiracBelief([3.0])))
            check_factor_conformance(f, p; π = TrivialBelief(), inputs, ps, st,
                pointwise = xy, msgs, expected_site = DiracBelief(solved),
                expected_posterior = DiracBelief(solved),
                game_gap = "LP-009: graph energy uses incoming points, not the solved point")
            @test first(local_free_energy(f, msgs, ps, st)) ≈ expected_energy
            solved_xy = dst == names[2] ? (v, nothing, solved) : (solved, nothing, v)
            @test first(scalar_energy(f, solved_xy..., ps, st)) ≈ 0 atol=1e-16
            # Not a bug to be fixed but the LP-009 gap made explicit: the graph energy is
            # evaluated at the incoming points, which need not be the solved point, so a
            # game-level free energy at the solved point would disagree with it.
            @test !isapprox(first(local_free_energy(f, msgs, ps, st)),
                            first(scalar_energy(f, solved_xy..., ps, st)); atol=1e-12)
        end
        @test !applicable(free_energy, f, TrivialBelief(), nothing, ps, st)
        @test !applicable(LenticulumCore.entropy, f, TrivialBelief(), nothing, ps, st)
    end
end
