include(joinpath(@__DIR__, "../../../test/factor_conformance.jl"))
@testset "Diffusion factor partial conformance" begin
    nt = (;)
    # Isotropic N(0,I), one fixed noise-free node: g(z)=c*z, c=λ*σ².
    # No stochastic comparison or additional quadrature approximation in the test.
    t, λ = 0.2, 0.7
    nodes = FieldNodes([t], zeros(2,1), [1.0])
    f = DiffusionFactor((x=1,y=1), PRED;
        prox=ImplicitProx(nodes; λ, message=:gaussian, tol=1e-12))
    c = λ * sigma(SCHED,t)^2
    ps, st = LuxCore.setup(Xoshiro(42), f)
    for p in supported_polarities(f)[1:2]
        src = only(observed_channels(p)); dst = only(unobserved_channels(p))
        inputs = NamedTuple{(src,)}((DiracBelief([2.0]),))
        π = Gaussian([0.4], ones(1,1))
        site = GaussianBelief([0.0], fill(c,1,1))
        post = Gaussian([0.4/(1+c)], fill(1/(1+c),1,1))
        check_factor_conformance(f, p; π, inputs, ps, st,
            pointwise=([1.0,2.0],p,[2.0,3.0]), msgs=inputs,
            expected_site=site, expected_posterior=post,
            game_gap="LP-009: score surrogate has no Definition 20 entropy; graph infers its own polarity")
        E = first(LenticulumCore.energy(f, [1.0,2.0], p, [2.0,3.0], ps, st))
        @test E[:clamp] ≈ 0.5
        @test E[:score] ≈ λ * sigma(SCHED,t)^3 * alpha(SCHED,t) * 5
        F = first(local_free_energy(f, inputs, ps, st))
        @test F[:clamp] ≈ 0
        @test F[:score] ≈ λ * sigma(SCHED,t)^3 * alpha(SCHED,t) * 4
        @test !applicable(free_energy, f, π, inputs, ps, st)
        @test !applicable(LenticulumCore.entropy, f, π, inputs, ps, st)
    end
    # Remaining binary assignments, including a joint lens at (U,U).
    for p in (Polarity((x=Unobserved(),y=Latent())),
              Polarity((x=Latent(),y=Unobserved())),
              Polarity((x=Unobserved(),y=Unobserved())))
        d = length(unobserved_channels(p))
        # g(z)=c*z is the gradient of c*||z||²/2. Thus the site is N(0,I/c).
        # Multiplying by the N(0,I) anchor gives precision (1+c)I, zero mean.
        π = Gaussian(zeros(d), Matrix{Float64}(I,d,d))
        site = GaussianBelief(zeros(d), Matrix{Float64}(c*I,d,d))
        post = Gaussian(zeros(d), Matrix{Float64}(I,d,d)/(1+c))
        scalar_site = GaussianBelief([0.0],fill(c,1,1))
        check_factor_conformance(f, p; π, inputs=nt, ps, st,
            pointwise=([1.0,2.0],p,[2.0,3.0]), msgs=nt,
            expected_site=site, expected_posterior=post,
            expected_target_sites=(x=scalar_site,y=scalar_site),
            game_gap="LP-009: no posterior entropy or explicit game/graph split")
        @test p in supported_polarities(f)
        if d == 2
            joint, _ = invert(first(assemble(f,p,ps,st)),π,nt,ps,st)
            @test belief_mean(joint) ≈ zeros(2)
            @test length(belief_mean(joint)) == 2
            @test belief_cov(joint) ≈ Matrix{Float64}(I,2,2)/(1+c)
        end
    end
    # Unary configuration also goes through the checker. Its graph default anchor is zero.
    f1 = DiffusionFactor((x=1,), PRED;
        prox=ImplicitProx(FieldNodes([t], zeros(1,1), [1.0]); λ, tol=1e-12))
    check_factor_conformance(f1, only(supported_polarities(f1)); π=TrivialBelief(),
        inputs=nt, ps=nt, st=nt, pointwise=([1.0],only(supported_polarities(f1)),[2.0]),
        msgs=nt, expected_site=DiracBelief([0.0]), expected_posterior=DiracBelief([0.0]),
        game_gap="LP-009: score surrogate is not a posterior entropy")
end

@testset "Diffusion polarity enumeration and cap (LP-017)" begin
    for blocks in ((x=1,y=1), (x=1,y=1,z=1), (w=1,x=1,y=1,z=1))
        f = DiffusionFactor(blocks, PRED)
        listed = supported_polarities(f)
        n = length(blocks)
        @test length(listed) == 3^n - 2^n
        @test Set(listed) == Set(filter(p -> supports_polarity(f,p), collect(polarities(f))))
        @test all(p -> supports_polarity(f,p), listed)
        @test map(p -> only(unobserved_channels(p)), listed[1:n]) == keys(blocks)
        @test all(p -> length(observed_channels(p)) == n-1, listed[1:n])
        @test listed == supported_polarities(f)
        @test !issink(f) && !isunidirectional(f)
    end
    f = DiffusionFactor((a=1,b=1,c=1,d=1,e=1), PRED)
    err = try supported_polarities(f) catch e; e end
    @test err isa ArgumentError
    @test occursin("supports_polarity", sprint(showerror, err))
    @test occursin("polarities", sprint(showerror, err))
    @test supports_polarity(f, first(polarities(f)))
    @test !issink(f) && !isunidirectional(f)
    @test isunidirectional(DiffusionFactor((x=1,),PRED))
    b = GraphBuilder()
    factor!(b,:diffusion,f)
    for n in keys(f.blocks)
        variable!(b,n)
        connect!(b,:diffusion,n,n)
    end
    g = build(b)
    @test validate(g) === g
    p = resolve_polarity(g,1,:a,(:b,:c))
    @test observed_channels(p) == (:b,:c) && latent_channels(p) == (:d,:e)
    @test check_legal(g,1,p) === nothing  # no eager list needed above the cap
end

@testset "Joint diffusion messages (LP-017)" begin
    # Noncontiguous, unequal target blocks exercise concatenation and joint-prior slicing.
    t, λ = 0.2, 0.7
    nodes = FieldNodes([t], zeros(4,1), [1.0])
    c = λ*sigma(SCHED,t)^2
    p = Polarity((a=Unobserved(),b=Observed(),c=Unobserved()))
    π = Gaussian([0.4,-0.2,0.7], Matrix{Float64}(I,3,3))
    inputs = (b=DiracBelief([2.0]),)
    for mode in (:point, :gaussian, :mixture)
        f = DiffusionFactor((a=1,b=1,c=2), PRED;
            prox=ImplicitProx(nodes; λ, message=mode, tol=1e-12))
        ps, st = LuxCore.setup(Xoshiro(42),f)
        lens = first(assemble(f,p,ps,st))
        post = first(invert(lens,π,inputs,ps,st))
        site = first(likelihood_message(lens,π,inputs,ps,st))
        μ = [0.4,-0.2,0.7]/(1+c)
        if mode === :point
            @test post isa DiracBelief && post.value ≈ μ
            @test site.value == post.value
        else
            pp = mode === :mixture ? only(post.components) : post
            ss = mode === :mixture ? only(site.components) : site
            @test belief_mean(pp) ≈ μ
            @test belief_cov(pp) ≈ Matrix{Float64}(I,3,3)/(1+c)
            @test ss.η ≈ zeros(3) atol=1e-10
            @test ss.Λ ≈ Matrix{Float64}(c*I,3,3)
            @test belief_distance(combine(π,site),post) < 1e-8
        end
        msg = first(factor_message(f,:c,p,inputs,π,ps,st))
        @test length(mode === :point ? msg.value : belief_mean(mode === :mixture ? only(msg.components) : msg)) == 2
        if mode === :point
            g = implicit_factor_pullback(f,p,inputs,π,(a=[1.0],c=[2.0,3.0]),ps,st)
            @test g.prior ≈ [1.0,2.0,3.0]/(1+c)
        end
    end
    # Correlated prior field: off-diagonal posterior covariance must survive joint selection.
    Σ = [1.0 0.4; 0.4 1.5]
    pred = NoisePredictor(AnalyticEps(SCHED,Σ),SCHED)
    f = DiffusionFactor((x=1,y=1),pred;
        prox=ImplicitProx(FieldNodes([t],zeros(2,1),[1.0]); λ, message=:gaussian,tol=1e-12))
    ps,st = LuxCore.setup(Xoshiro(42),f)
    lens = first(assemble(f,Polarity((x=Unobserved(),y=Unobserved())),ps,st))
    π = Gaussian([0.4,-0.2],Matrix{Float64}(I,2,2))
    J = λ*sigma(SCHED,t)^2 * inv(alpha(SCHED,t)^2*Σ + sigma(SCHED,t)^2*I)
    post = first(invert(lens,π,(;),ps,st))
    site = first(likelihood_message(lens,π,(;),ps,st))
    @test belief_cov(post) ≈ inv(I+J)
    @test belief_mean(post) ≈ (I+J)\[0.4,-0.2]
    @test site.Λ ≈ J
    @test site.η ≈ zeros(2) atol=1e-12
end

@testset "Joint mixture rejects undefined anchor division (LP-017)" begin
    # λ=0 leaves only the N(z₀,I) anchor. Dividing it out gives flat components,
    # which have no finite Gaussian normalizer and cannot carry mixture weights.
    f = DiffusionFactor((x=1,y=1),PRED;
        prox=ImplicitProx(FieldNodes([0.2],zeros(2,1),[1.0]); λ=0, message=:mixture))
    ps,st = LuxCore.setup(Xoshiro(42),f)
    lens = first(assemble(f,Polarity((x=Unobserved(),y=Unobserved())),ps,st))
    π = Gaussian(zeros(2),Matrix{Float64}(I,2,2))
    post = first(invert(lens,π,(;),ps,st))
    @test belief_cov(only(post.components)) ≈ Matrix{Float64}(I,2,2)
    err = try likelihood_message(lens,π,(;),ps,st) catch e; e end
    @test err isa ArgumentError
    @test occursin("improper mixture component",sprint(showerror,err))
end

@testset "Joint mixture anchor pooling (LP-017)" begin
    pred = NoisePredictor(GaussianMixtureEps(SCHED,[0.0 0.0; 2.0 -2.0];s=0.4),SCHED)
    nodes = field_nodes(Xoshiro(1),2;samples=8)
    λ = density_lambda(SCHED,nodes)
    f = DiffusionFactor((x=1,y=1),pred;prox=ImplicitProx(nodes;λ,message=:mixture))
    ps,st = LuxCore.setup(Xoshiro(42),f)
    p = Polarity((x=Unobserved(),y=Unobserved()),(x=0.3,y=0.5))
    lens = first(assemble(f,p,ps,st))
    anchor = Gaussian([0.1,0.4],Matrix(Diagonal([1/0.3^2,4.0])))
    post = first(invert(lens,anchor,(;),ps,st))
    site = first(likelihood_message(lens,anchor,(;),ps,st))
    pooled = combine(site,anchor)
    @test length(post.components) == length(site.components) == 2
    @test mixture_weights(pooled) ≈ mixture_weights(post) rtol=1e-8
    for (a,b) in zip(pooled.components,post.components)
        @test length(belief_mean(a)) == 2
        @test belief_mean(a) ≈ belief_mean(b)
        @test belief_cov(a) ≈ belief_cov(b)
    end
end
