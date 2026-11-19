using LuxCore, Random, LinearAlgebra

# y = a*x + b + N(0,q). The test supplies the parameter-channel rule; expose supplies routing.
struct P36Affine{Beliefs} <: AbstractLenticulumFactor
    q::Float64
end
LuxCore.parameterlength(::P36Affine) = 2
LenticulumCore.channels(::P36Affine) = (LenticulumCore.Channel(:x, 1), LenticulumCore.Channel(:y, 1))
LenticulumCore.supported_polarities(::P36Affine) =
    (Polarity(x = Observed(), y = Unobserved()), Polarity(x = Unobserved(), y = Observed()))
LenticulumCore.supports_polarity(f::P36Affine, p::Polarity) = p in supported_polarities(f)
LuxCore.initialparameters(::AbstractRNG, ::P36Affine) = (a = [1.2], b = [0.3])
LuxCore.initialstates(::AbstractRNG, ::P36Affine) = (;)
LenticulumCore.energyspace(::P36Affine) = ScalarEnergySpace()
LenticulumCore.scalarisation(::P36Affine) = IdentityScalarisation()
LenticulumCore.pointwise_energyspace(::P36Affine) = ScalarEnergySpace()
LenticulumCore.pointwise_scalarisation(::P36Affine) = IdentityScalarisation()
function Mycelium.factor_message(f::P36Affine, target::Symbol, pol, inputs, prior, ps, st)
    a, b = only(ps.a), only(ps.b)
    if target === :y
        x = only(inputs.x.value)
        return Gaussian(a*x + b, f.q), st
    end
    y = only(inputs.y.value)
    return GaussianBelief([a*(y-b)/f.q], fill(a^2/f.q, 1, 1)), st
end
Mycelium.local_free_energy(f::P36Affine, msgs, ps, st) =
    (((only(msgs.y.value)-only(ps.a)*only(msgs.x.value)-only(ps.b))^2/f.q + log(2π*f.q))/2, st)
Mycelium.supports_parameter_polarity(::P36Affine{true}, p) =
    p[:a] isa Unobserved && p[:x] isa Observed && p[:y] isa Observed
function Mycelium.parameter_message(f::P36Affine{true}, target, p, msgs, ps, st)
    x, y, b = only(msgs.x.value), only(msgs.y.value), only(ps.b)
    return GaussianBelief([x*(y-b)/f.q], fill(x^2/f.q, 1, 1)), st
end
function Mycelium.parameter_free_energy(f::P36Affine{true}, msgs, ps, st)
    x, y, b = only(msgs.x.value), only(msgs.y.value), only(ps.b)
    μ, v = only(belief_mean(msgs.a)), only(belief_cov(msgs.a))
    return ((y-b-x*μ)^2 + x^2*v)/ (2f.q) + log(2π*f.q)/2, st
end

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
