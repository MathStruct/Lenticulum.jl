using Random, LuxCore
# A hand-written unary normal, joined by a relay to a pinned variable: a two-factor tree.
struct P36Normal{Learnable} <: AbstractLenticulumFactor end
LenticulumCore.channels(::P36Normal) = (LenticulumCore.Channel(:x,1),)
LenticulumCore.supported_polarities(::P36Normal) = (Polarity(x=Unobserved()),)
LenticulumCore.supports_polarity(::P36Normal,p::Polarity) = p[:x] isa Unobserved
LenticulumCore.islearnable(::P36Normal{L}) where L = L
LuxCore.initialparameters(::AbstractRNG,::P36Normal) = (mu=[0.3],)
Mycelium.factor_message(::P36Normal,::Symbol,pol,msgs,prior,ps,st) = (Gaussian(only(ps.mu),0.7),st)
Mycelium.local_free_energy(::P36Normal,msgs,ps,st) = ((only(msgs.x.value)-only(ps.mu))^2/1.4+log(2π*0.7)/2,st)
Mycelium.factor_belief(::P36Normal,msgs,ps,st) = msgs.x
Mycelium.belief_free_energy(::P36Normal,b::DiracBelief,ps,st) = ((only(b.value)-only(ps.mu))^2/1.4+log(2π*0.7)/2,st)

@testset "P36 sample_gradient: pinned normal tree against negative log density" begin
    for learnable in (true,false)
        b=GraphBuilder();variable!(b,:x,1);variable!(b,:pin,1)
        factor!(b,:p,P36Normal{learnable}());connect!(b,:p,:x,:x)
        factor!(b,:relay,RelayFactor(:x,:pin));connect!(b,:relay,:x,:x);connect!(b,:relay,:pin,:pin)
        g=build(b);ps,st=LuxCore.setup(Xoshiro(36),g)
        sampler=GibbsSampler(g,(:pin,));buf=SampleBuffer(WindowFilter(4))
        for (x,w) in ((-0.4,1.),(1.2,3.),(100.,0.))
            record!(buf,sampler,(pin=[x],),ps,st;weight=w)
        end
        @test sample_weights(buf) ≈ [0.25,0.75,0.]
        ref = learnable ? (0.3-(0.25*(-0.4)+0.75*1.2))/0.7 : 0.
        default=sample_gradient(buf,sampler,ps,st)
        through=sample_gradient(buf,sampler,ps,st;through_inference=true)
        @test only(default.p.mu) ≈ ref atol=1e-8
        @test only(through.p.mu) ≈ ref atol=1e-8
        @test default.p.mu ≈ through.p.mu atol=1e-8
        # If a zero-weight sample is evaluated, this missing pin must fail inference.
        buf.entries[end]=merge(buf.entries[end],(pins=Dict{Int,Any}(),))
        @test sample_gradient(buf,sampler,ps,st).p.mu ≈ [ref] atol=1e-8
        @test sample_gradient(buf,sampler,ps,st;through_inference=true).p.mu ≈ [ref] atol=1e-8
    end
end
