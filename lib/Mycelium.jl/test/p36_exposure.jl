using LuxCore, Random, LinearAlgebra

include("support/affine.jl")

@testset "P36 exposure: clamped substitution and Gaussian parameter likelihood" begin
    rng = Xoshiro(36); nt = (;)
    for _ in 1:5
        f = P36Affine{true}(0.2 + rand(rng)); e = expose(f, :a => :weight)
        ps, st = LuxCore.setup(rng, e)
        a, x, y = 0.5+rand(rng), randn(rng), randn(rng)
        msgs = (x = DiracBelief([x]), y = DiracBelief([y]), weight = DiracBelief([a]))
        fixed = (a = [a], b = ps.b)
        @test LuxCore.parameterlength(e) == 1
        @test keys(ps) == (:b,)
        @test islearnable(e) && !isfrozen(e) && Mycelium.uses_cavity(e)
        @test length(channels(e)) == 3
        for (target, pol) in ((:y, Polarity(x=Observed(), y=Unobserved(), weight=Observed())),
                              (:x, Polarity(x=Unobserved(), y=Observed(), weight=Observed())))
            @test supports_polarity(e, pol)
            m, _ = factor_message(e, target, pol, msgs, TrivialBelief(), ps, st)
            ref, _ = factor_message(f, target, target === :y ? first(supported_polarities(f)) : last(supported_polarities(f)), msgs, nothing, fixed, st)
            @test m.η ≈ ref.η && m.Λ ≈ ref.Λ
            @test belief_mean(m) ≈ [target === :y ? a*x+only(ps.b) : (y-only(ps.b))/a]
            @test belief_cov(m) ≈ fill(target === :y ? f.q : f.q/a^2,1,1)
        end
        @test first(local_free_energy(e, msgs, ps, st)) ≈ (y-a*x-only(ps.b))^2/(2f.q)+log(2π*f.q)/2
        @test first(local_free_energy(e, msgs, ps, st)) == first(local_free_energy(f, msgs, fixed, st))
        pol = Polarity(x=Observed(), y=Observed(), weight=Unobserved())
        @test supports_polarity(e, pol)
        m, _ = factor_message(e, :weight, pol, (x=msgs.x,y=msgs.y), nothing, ps, st)
        @test m.η ≈ [x*(y-only(ps.b))/f.q] && m.Λ ≈ fill(x^2/f.q,1,1)
        soft = merge(msgs, (weight=Gaussian(0.4, 0.7),))
        @test first(local_free_energy(e, soft, ps, st)) ≈ ((y-only(ps.b)-0.4x)^2+0.7x^2)/(2f.q)+log(2π*f.q)/2
        @test !hasnopolarity(e) && !isunidirectional(e)
        @test pol in supported_polarities(e)
        @test !supports_polarity(e, Polarity(x=Unobserved()))
        for op in (energyspace, scalarisation, pointwise_energyspace, pointwise_scalarisation)
            @test typeof(op(e)) == typeof(op(f))
        end
    end
    f = P36Affine{false}(0.5); e = expose(f, :a)
    pol = Polarity(x=Observed(), y=Observed(), a=Unobserved())
    @test !supports_polarity(e, pol)
    @test_throws ArgumentError factor_message(e, :a, pol, (x=DiracBelief([1.]),y=DiracBelief([2.])), nothing, (b=[0.3],), nt)
    @test_throws ArgumentError local_free_energy(e, (x=DiracBelief([1.]),y=DiracBelief([2.]),a=Gaussian(0.,1.)), (b=[0.3],), nt)
    @test_throws ArgumentError parameter_message(f, :a, pol, nt, nt, nt)
    @test_throws ArgumentError parameter_free_energy(f, nt, nt, nt)
    for names in ((:a, :a), (:a=>:x,), (:a=>:w,:b=>:w), (:missing,), (1,))
        @test_throws ArgumentError expose(f, names)
    end
end

# Structural refusal paths need no Gaussian implementation.
struct P36ExposureShape <: AbstractLenticulumFactor
    names::Tuple
    ps::NamedTuple
end
LenticulumCore.channels(f::P36ExposureShape) = Tuple(LenticulumCore.Channel(n,1) for n in f.names)
LuxCore.initialparameters(::AbstractRNG,f::P36ExposureShape) = f.ps
LenticulumCore.supports_polarity(::P36ExposureShape,::Polarity) = false
@testset "P36 exposure: invalid blocks, enumeration cap and shadowed namespace" begin
    @test_throws ArgumentError expose(P36ExposureShape((:x,), (a=1.,)), :a)
    many = expose(P36ExposureShape((:x,:y,:z,:u,:v), (a=[1.],)), :a)
    @test_throws ArgumentError supported_polarities(many)
    e = expose(P36ExposureShape((:a,), (a=[1.],)), :a=>:weight)
    pol = Polarity(a=Observed(),weight=Unobserved())
    @test !supports_polarity(e, pol)
    @test hasnopolarity(e) && !isunidirectional(e)
    @test_throws ArgumentError factor_message(e,:weight,pol,(a=DiracBelief([1.]),),nothing,(;),(;))
    @test_throws ArgumentError local_free_energy(e,(a=DiracBelief([1.]),weight=Gaussian(0.,1.)),(;),(;))
end

# The residual form is also used by the overlay oracle in p36_overlay_ad.jl.
Mycelium.cluster_form(f::P36Affine, ps) = ClusterForm(
    (x=reshape(-ps.a,1,1), y=ones(1,1)), reshape(ps.b,1,1), fill(f.q,1,1), [0.], nothing)
