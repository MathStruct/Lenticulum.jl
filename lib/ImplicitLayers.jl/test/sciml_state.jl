@testset "SciML CNF: augmented state, reverse time, and Lux state" begin
    A = [-0.4 0.3; -0.2 -0.5]
    u0 = [1.0, -2.0]
    itg = SciMLIntegrator(Tsit5(); abstol=1e-10, reltol=1e-9)
    for (t0,t1) in ((0.0,1.5), (1.5,0.0))
        u, Δ = integrate_with_divergence((u,t) -> A*u, u0, t0, t1, itg)
        @test u ≈ exp(A*(t1-t0))*u0 rtol=1e-8
        @test Δ ≈ tr(A)*(t1-t0) rtol=1e-6 # dense FD trace, ε=1e-7
    end
    for itg in (itg, SciMLIntegrator(ImplicitMidpoint(); adaptive=false, dt=0.02,
                                     abstol=1e-12, reltol=1e-12))
        f = NeuralODEFactor(CountingDynamics(), 1; integrator=itg)
        ps, st = LuxCore.setup(Xoshiro(42), f)
        u, st1 = flow_forward(f, [1.0], ps, st)
        @test st1.calls > st.calls
        @test u ≈ [exp(-0.5)] rtol=1e-4
        _, st2 = flow_reverse(f, u, ps, st1)
        @test st2.calls > st1.calls
        _, Δ, st3 = flow_logdet(f, [1.0], ps, st2)
        @test st3.calls > st2.calls
        @test Δ ≈ -0.5 rtol=1e-6
        # Nonempty state could change the RHS, so assemble conservatively declines exactness.
        lens, _ = assemble(f, first(supported_polarities(f)), ps, st)
        @test lens.inversion isa SolverInversion
        for p in supported_polarities(f)
            src = only(observed_channels(p))
            lens, _ = assemble(f, p, ps, st)
            solved, nextst = invert(lens, TrivialBelief(),
                NamedTuple{(src,)}((DiracBelief([1.0]),)), ps, st)
            expected = src == :z0 ? exp(-0.5) : exp(0.5)
            @test solved.value ≈ [expected] rtol=1e-4
            @test nextst.calls > st.calls
        end
    end
    # Count every evaluation independently, including rejected stages and FD probes.
    counter = Ref(0)
    f = NeuralODEFactor(CountingDynamics(), 1; integrator=itg,
        input=(u,t) -> (counter[] += 1; (u,t)))
    ps, st = LuxCore.setup(Xoshiro(42), f)
    _, _, finalst = flow_logdet(f, [1.0], ps, st)
    @test finalst.calls == counter[] > 0
end

