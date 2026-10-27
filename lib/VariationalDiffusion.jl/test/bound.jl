include(joinpath(@__DIR__, "../../../test/conformance_capture.jl"))

@testset "LP-009 diffusion ELBO: bounds and exact Gaussian score" begin
    sched = VPSDE(βmin=0.3,βmax=3.0)
    layer = GaussianMixtureEps(sched,zeros(2,1);s=1.0)
    f = DiffusionFactor((x=2,),NoisePredictor(layer,sched))
    ps,st = LuxCore.setup(Xoshiro(9002),f)
    times = [0.1,0.3,0.6,1.0]
    # For N(0,I) data the analytic score is -z at every t. The reverse chain
    # with variance β_i is stationary N(0,I), including its terminal prior.
    for x in ([0.0,0.0],[0.4,-0.7],[2.0,1.0])
        bound,next = free_energy_upper_bound(f,x,ps,st;times)
        exact = (length(x)*log(2π)+sum(abs2,x))/2
        @test bound isa FreeEnergyUpperBound
        @test !(bound isa Real)
        @test next === st
        @test bound.value >= exact-1e-12
        @test bound.value ≈ exact atol=2e-12
        check_factor_conformance(f,nothing;scope=:bound,ps,st,
            bound=(evidence=x,kwargs=(;times),exact=exact))
        @test_throws MethodError scalarise(IdentityScalarisation(),bound)
        # A displaced analytic Gaussian score approaches the exact data score.
        gaps = Float64[]
        for δ in (1.0,0.5,0.1,0.0)
            pp = (μ=fill(δ,2,1),)
            B = first(free_energy_upper_bound(f,x,pp,st;times)).value
            # Independent marginalisation of all affine reverse transitions.
            modelmean = zeros(2)
            for i in reverse(eachindex(times))
                αi = alpha(sched,times[i])
                αprev = i == 1 ? 1.0 : alpha(sched,times[i-1])
                a = αi/αprev; β = 1-a^2
                modelmean = a.*modelmean + β*αi/a .* vec(pp.μ)
            end
            model_nll = (2log(2π)+sum(abs2,x-modelmean))/2
            @test B >= model_nll-2e-12
            check_factor_conformance(f,nothing;scope=:bound,ps=pp,st,
                bound=(evidence=x,kwargs=(;times),exact=model_nll))
            push!(gaps,B-model_nll)
        end
        @test issorted(gaps;rev=true)
        @test gaps[end] ≈ 0 atol=2e-12
        @test gaps[1] > gaps[2] > gaps[3] > 1e-4
    end
    x = [0.4,-0.7]
    @test_throws ArgumentError check_factor_conformance(f,nothing;scope=:bound,ps,st)
    @test_throws ArgumentError check_factor_conformance(f,nothing;scope=:bound,ps,st,
        bound=(evidence=x,kwargs=(;times),exact=Inf))
    # The stationary chain's NLL is exact. A falsely larger likelihood cost must fail.
    captured = capture_conformance_checks() do
        check_factor_conformance(f,nothing;scope=:bound,ps,st,
            bound=(evidence=x,kwargs=(;times),exact=(2log(2π)+sum(abs2,x))/2+1))
    end
    @test conformance_failures(captured) == 1
    estimate,next = estimate_free_energy_upper_bound(f,x,ps,st;times,nsamples=12000,rng=Xoshiro(9002))
    exact = first(free_energy_upper_bound(f,x,ps,st;times)).value
    @test estimate isa FreeEnergyBoundEstimate
    @test estimate.nsamples == 12000
    @test estimate.standard_error > 0
    @test abs(estimate.value-exact) < 5estimate.standard_error
    @test next === st
    prox_factor = DiffusionFactor((x=2,),NoisePredictor(layer,sched);prox=ProxDM(MixtureProx(layer)))
    prox_ps,prox_st = LuxCore.setup(Xoshiro(9002),prox_factor)
    prox_bound,prox_next = free_energy_upper_bound(prox_factor,x,prox_ps,prox_st;times)
    @test prox_bound.value ≈ exact
    @test prox_next === prox_st
    @test_throws ArgumentError free_energy_upper_bound(f,x,ps,st;times=[0.2,0.2])
    @test_throws ArgumentError free_energy_upper_bound(f,x,ps,st;times=[0.4,0.2])
    @test_throws ArgumentError free_energy_upper_bound(f,x,ps,st;times=[0.0])
    @test_throws DimensionMismatch free_energy_upper_bound(f,[1.0],ps,st;times)
    @test_throws ArgumentError free_energy(f,TrivialBelief(),x,ps,st)
    mismatched = DiffusionFactor((x=2,),NoisePredictor(layer,VPSDE()))
    @test_throws ArgumentError free_energy_upper_bound(mismatched,x,ps,st;times)
    mixture = DiffusionFactor((x=2,),NoisePredictor(GaussianMixtureEps(sched,[0.0 1.0;0.0 1.0]),sched))
    mp,ms = LuxCore.setup(Xoshiro(9002),mixture)
    @test_throws ArgumentError free_energy_upper_bound(mixture,x,mp,ms;times)
    est = first(estimate_free_energy_upper_bound(mixture,x,mp,ms;times,nsamples=100,rng=Xoshiro(1)))
    @test isfinite(est.value) && est.standard_error > 0
    # Nonstationary Gaussian score: independently propagate the reverse model's
    # mean and variance, rather than identifying its terminal law with the data law.
    for width in (0.6,1.7)
        gf = DiffusionFactor((x=2,),NoisePredictor(GaussianMixtureEps(sched,fill(0.3,2,1);s=width),sched))
        gp,gs = LuxCore.setup(Xoshiro(1),gf)
        m,v = zeros(2),1.0
        for i in reverse(eachindex(times))
            αi = alpha(sched,times[i]); σ²i = sigma(sched,times[i])^2
            αprev = i == 1 ? 1.0 : alpha(sched,times[i-1])
            a = αi/αprev; β = 1-a^2
            vi = αi^2*width^2+σ²i
            M = (1-β/vi)/a
            m = M.*m + β*αi/(a*vi).*vec(gp.μ)
            v = M^2*v+β
        end
        nll = (2log(2π*v)+sum(abs2,x-m)/v)/2
        B = first(free_energy_upper_bound(gf,x,gp,gs;times)).value
        @test B >= nll-2e-12
        check_factor_conformance(gf,nothing;scope=:bound,ps=gp,st=gs,
            bound=(evidence=x,kwargs=(;times),exact=nll))
        estimate = first(estimate_free_energy_upper_bound(gf,x,gp,gs;times,nsamples=10000,rng=Xoshiro(9004)))
        @test abs(estimate.value-B) < 5estimate.standard_error
    end
end
