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
    for p in supported_polarities(f)
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
    # All remaining accepted O/U/L role assignments at arity two. These are absent
    # from enumeration (LP-017). Single-target latent cases can still check the lens.
    for p in (Polarity((x=Unobserved(),y=Latent())),
              Polarity((x=Latent(),y=Unobserved())),
              Polarity((x=Unobserved(),y=Unobserved())))
        multi = length(unobserved_channels(p)) > 1
        π = Gaussian([0.4], ones(1,1))
        check_factor_conformance(f, p; π, inputs=nt, ps, st,
            pointwise=([1.0,2.0],p,[2.0,3.0]), msgs=nt,
            expected_site=GaussianBelief([0.0],fill(c,1,1)),
            expected_posterior=Gaussian([0.4/(1+c)],fill(1/(1+c),1,1)),
            polarity_gap="LP-017: accepted polarity omitted from enumeration",
            message_gap=multi ? "LP-017: inversion returns only the first unobserved block, not a joint posterior" : nothing,
            game_gap="LP-009: no posterior entropy or explicit game/graph split")
        @test_broken p in supported_polarities(f)
        if multi
            post, _ = invert(first(assemble(f,p,ps,st)),π,nt,ps,st)
            @test belief_mean(post) ≈ [0.4/(1+c)]
            @test_broken length(belief_mean(post)) == 2
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
