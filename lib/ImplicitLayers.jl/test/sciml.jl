@testset "SciML extension absent: actionable error" begin
    @test Base.get_extension(ImplicitLayers, :ImplicitLayersOrdinaryDiffEqExt) === nothing
    for solveflow in (integrate, integrate_with_divergence)
        err = try
            solveflow((u,t) -> u, [1.0], 0.0, 1.0, SciMLIntegrator(nothing))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("OrdinaryDiffEqCore", sprint(showerror, err))
    end
end

using OrdinaryDiffEqCore
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqSDIRK

struct CountingDynamics <: LuxCore.AbstractLuxLayer end
LuxCore.initialparameters(::AbstractRNG, ::CountingDynamics) = (; a = -0.5)
LuxCore.initialstates(::AbstractRNG, ::CountingDynamics) = (; calls = 0)
LuxCore.parameterlength(::CountingDynamics) = 1
LuxCore.statelength(::CountingDynamics) = 1
(::CountingDynamics)(inp, ps, st) = (ps.a .* inp[1], (; calls = st.calls + 1))

@testset "SciML integration and discrete reversibility" begin
    @test Base.get_extension(ImplicitLayers, :ImplicitLayersOrdinaryDiffEqExt) !== nothing
    for dt in (0, -0.1, Inf, NaN)
        @test_throws ArgumentError SciMLIntegrator(Tsit5(); dt)
    end
    @test !isreversible(SciMLIntegrator(Tsit5(); adaptive=false, dt=0.1))
    @test !isreversible(SciMLIntegrator(ImplicitMidpoint(); dt=0.1))
    @test !isreversible(SciMLIntegrator(ImplicitMidpoint(); adaptive=false))
    @test !isreversible(SciMLIntegrator(ImplicitMidpoint(); adaptive=false, dt=0.1,
                                      tstops=[0.3]))
    limited = ImplicitMidpoint(step_limiter! = (u, integrator, p, t) -> nothing)
    @test !isreversible(SciMLIntegrator(limited; adaptive=false, dt=0.1))

    for alg in (ImplicitMidpoint(), Trapezoid())
        fixed = SciMLIntegrator(alg; adaptive=false, dt=0.02, abstol=1e-12, reltol=1e-12)
        @test isreversible(fixed)
        @test inversion_kind(fixed, 0.0, 1.0) isa ExactInversion
        @test inversion_kind(fixed, 1.0, 0.0) isa ExactInversion
        @test inversion_kind(fixed, 0.0, 1.01) isa SolverInversion # unmatched final short step
        for a in (0.5, -5.0)
            vf = (u,t) -> a .* u
            z1 = integrate(vf, [3.0], 0.0, 1.0, fixed)
            # Midpoint and trapezoid share the same scalar linear discrete multiplier.
            multiplier = (1 + a*0.02/2)/(1 - a*0.02/2)
            @test only(z1) ≈ 3multiplier^50 rtol=1e-11
            @test only(z1) ≈ 3exp(a) rtol=0.005 # second-order truncation at a=-5
            z0 = integrate(vf, [3.0], 1.0, 0.0, fixed)
            r = abs(only(integrate(vf, z0, 0.0, 1.0, fixed)) - 3)/3
            @test r < 1e-10 # reltol=abstol=1e-12; accommodates accumulated stage/roundoff error
            @info "symmetric round trip" algorithm=typeof(alg) a residual=r
        end
        # Time-dependent nonlinear dynamics catch grid/stage mistakes hidden by Az.
        vf = (u,t) -> -0.1 .* u.^2 .+ sin(t)
        z1 = integrate(vf, [0.7], 0.0, 1.0, fixed)
        @test integrate(vf, z1, 1.0, 0.0, fixed) ≈ [0.7] rtol=1e-10
    end

    for a in (0.5, -5.0)
        vf = (u,t) -> a .* u
        errors = Float64[]
        for tol in (1e-3, 1e-6, 1e-9)
            adaptive = SciMLIntegrator(Tsit5(); reltol=tol, abstol=tol*1e-3)
            @test inversion_kind(adaptive, 0.0, 1.0) isa SolverInversion
            z1 = integrate(vf, [3.0], 0.0, 1.0, adaptive)
            @test z1 ≈ [3exp(a)] rtol=5tol
            z0 = integrate(vf, [3.0], 1.0, 0.0, adaptive)
            push!(errors, abs(only(integrate(vf, z0, 0.0, 1.0, adaptive)) - 3)/3)
        end
        @info "Tsit5 tolerance scaling" a tolerances=(1e-3,1e-6,1e-9) residuals=Tuple(errors)
        # Local control is not a strict global residual bound or an exact power law.
        # Over 3-decade refinements allow slopes 0.5–1.5, away from roundoff.
        @test all(0.5 .< log10.(errors[1:2] ./ errors[2:3]) ./ 3 .< 1.5)
    end

    # Endpoint-only settings override caller storage requests; failures must not look like results.
    itg = SciMLIntegrator(Tsit5(); reltol=1e-10, abstol=1e-12, save_end=false,
                           saveat=[0.2, 0.4], save_everystep=true)
    @test integrate((u,t) -> u, [1.0], 0.0, 1.0, itg) ≈ [exp(1)] rtol=1e-9
    @test integrate((u,t) -> u, [1.0,2.0], 0.0, 1.0,
        SciMLIntegrator(Tsit5(); reltol=1e-10, abstol=1e-12, save_idxs=1)) ≈
        [exp(1),2exp(1)] rtol=1e-9
    @test integrate((u,t) -> error("not called"), [2.0], 1.0, 1.0, itg) == [2.0]
    @test_throws ErrorException integrate((u,t) -> -5u, [1.0], 0.0, 1.0,
        SciMLIntegrator(Tsit5(); maxiters=1))
end

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
    end
    # Count every evaluation independently, including rejected stages and FD probes.
    counter = Ref(0)
    f = NeuralODEFactor(CountingDynamics(), 1; integrator=itg,
        input=(u,t) -> (counter[] += 1; (u,t)))
    ps, st = LuxCore.setup(Xoshiro(42), f)
    _, _, finalst = flow_logdet(f, [1.0], ps, st)
    @test finalst.calls == counter[] > 0
end

@testset "SciML NeuralODE factor conformance and lens games" begin
    for itg in (SciMLIntegrator(Tsit5(); abstol=1e-12, reltol=1e-10),
                SciMLIntegrator(ImplicitMidpoint(); adaptive=false, dt=0.02,
                                abstol=1e-12, reltol=1e-12))
        # Zero dynamics gives independent exact fixtures in both polarities, like conformance.jl.
        f = NeuralODEFactor(LinearDynamics(zeros(1,1)), 1; integrator=itg)
        ps, st = LuxCore.setup(Xoshiro(42), f)
        for p in supported_polarities(f)
            src = only(observed_channels(p))
            v = src == :z0 ? [1.0] : [3.0]
            inputs = NamedTuple{(src,)}((DiracBelief(v),))
            check_factor_conformance(f, p; π=TrivialBelief(), inputs, ps, st,
                pointwise=([1.0],nothing,[3.0]),
                msgs=(z0=DiracBelief([1.0]),z1=DiracBelief([3.0])),
                expected_site=DiracBelief(v), expected_posterior=DiracBelief(v),
                game_prior=DiracBelief([1.0]), game_y=DiracBelief([3.0]))
            lens, _ = assemble(f, p, ps, st)
            @test isexact(lens.inversion) == isreversible(itg)
            @test first(free_energy(lens, TrivialBelief(), inputs, ps, st)) < 1e-20
        end
    end
    # A nonzero contracting flow checks that the lens game retains actual reverse residuals.
    for itg in (SciMLIntegrator(Tsit5(); abstol=1e-6, reltol=1e-3),
                SciMLIntegrator(ImplicitMidpoint(); adaptive=false, dt=0.02,
                                abstol=1e-12, reltol=1e-12))
        f = NeuralODEFactor(LinearDynamics(fill(-5.0,1,1)), 1; integrator=itg)
        ps, st = LuxCore.setup(Xoshiro(42), f)
        lens, _ = assemble(f, Polarity(; z0=Unobserved(), z1=Observed()), ps, st)
        inputs = (; z1=DiracBelief([3.0]))
        solved, _ = invert(lens, TrivialBelief(), inputs, ps, st)
        r, _ = residual(f, solved.value, [3.0], ps, st)
        @test first(free_energy(lens, TrivialBelief(), inputs, ps, st)) ≈ sum(abs2,r)/2
    end
end
