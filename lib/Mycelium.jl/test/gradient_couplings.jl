using Test, Mycelium, LenticulumCore, Random, LuxCore
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

struct CoupledAffine{F,L} <: AbstractLenticulumFactor
    regularisation::Float64
end
LenticulumCore.channels(::CoupledAffine) = (LenticulumCore.Channel(:in,nothing), LenticulumCore.Channel(:out,nothing))
LenticulumCore.supported_polarities(::CoupledAffine) = (Polarity(in=Observed(),out=Unobserved()),)
LenticulumCore.supports_polarity(f::CoupledAffine, p::Polarity) = p == only(supported_polarities(f))
LenticulumCore.isfrozen(::CoupledAffine{F,L}) where {F,L} = F
LenticulumCore.islearnable(::CoupledAffine{F,L}) where {F,L} = L
Mycelium.factor_message(::CoupledAffine, target::Symbol, pol, inputs, prior, ps, st) =
    (DiracBelief(ps.gain * point(inputs.in) + ps.encoder.b + ps.decoder.b), st)
Mycelium.local_free_energy(f::CoupledAffine, beliefs, ps, st) = (f.regularisation * ps.gain^2 / 2, st)

function coupled_chain(coupling; detach=false, freeze=(), frozen=false, learnable=true)
    b = GraphBuilder()
    for v in (:x,:y,:z); variable!(b,v); end
    factor!(b,:data,DataFactor(:x,2.0))
    factor!(b,:a,CoupledAffine{false,true}(0.1))
    factor!(b,:b,CoupledAffine{frozen,learnable}(0.0); freeze)
    factor!(b,:loss,LossFactor((:z,), z -> (z-1)^2/2))
    connect!(b,:data,:x,:x;direction=Emitting(),coupling)
    connect!(b,:a,:in,:x;direction=Absorbing(),coupling)
    connect!(b,:a,:out,:y;direction=Emitting(),coupling,detach)
    connect!(b,:b,:in,:y;direction=Absorbing(),coupling)
    connect!(b,:b,:out,:z;direction=Emitting(),coupling)
    connect!(b,:loss,:z,:z;direction=Absorbing(),coupling)
    g=build(b)
    p=(data=(;),a=(gain=0.7,encoder=(b=0.1,),decoder=(b=-0.2,)),
       b=(gain=1.3,encoder=(b=0.2,),decoder=(b=-0.1,)),loss=(;))
    st=(data=(;),a=(;),b=(;),loss=(;))
    return g,p,st,forward_backward_schedule(g)
end

@testset "LP-011 mixed chain, detach, parameter subsets" begin
    for ad in (nothing, AutoForwardDiff(), AutoZygote())
        for c in (ExactCoupling(),PathwiseCoupling())
            g,p,st,sched=coupled_chain(c)
            _,_,_,s=infer!(g,sched,p,st)
            grad=free_energy_gradient(s,g,p,st;ad,schedule=sched)
            r=1.3*(2*0.7+0.1-0.2)+0.2-0.1-1
            @test grad.a.gain ≈ 0.1*0.7+2*1.3*r atol=2e-8
            @test grad.b.encoder.b ≈ r atol=2e-8
            pd,_,os=parameter_step(s,g,p,st;ad,schedule=sched,rule=GradientDescent(0.01))
            @test pd.a.gain ≈ p.a.gain-0.01*grad.a.gain
            @test length(os.b)==3

            gd,pd,sd,sc=coupled_chain(c;detach=true)
            _,_,_,store=infer!(gd,sc,pd,sd)
            d=free_energy_gradient(store,gd,pd,sd;ad,schedule=sc)
            @test d.a.gain ≈ 0.07 atol=2e-8
            @test d.b.gain ≈ r*1.3 atol=2e-8
            # Raw differentiable evaluation observes detach too (ForwardDiff/ChainRules).
            if ad !== nothing
                raw=parameter_gradient(pd;ad) do q
                    _,_,_,ss=infer!(gd,sc,q,sd)
                    first(scalar_free_energy(ss,gd,q,sd))
                end
                @test raw.a.gain ≈ 0.07 atol=2e-8
            end

            for freeze in ((:encoder,), :all, ((:encoder,),(:decoder,)))
                gf,pf,sf,sch=coupled_chain(c;freeze)
                _,_,_,store=infer!(gf,sch,pf,sf)
                f=free_energy_gradient(store,gf,pf,sf;ad,schedule=sch)
                @test f.a.gain ≈ grad.a.gain atol=2e-8
                @test f.b.encoder.b == 0
                @test isfrozen(factornode(gf,:b))
                pn,_,on=parameter_step(store,gf,pf,sf;ad,schedule=sch)
                @test pn.b.encoder == pf.b.encoder
                if freeze === :all
                    @test !haskey(on,:b)
                    @test pn.b == pf.b
                else
                    @test length(on.b) == (freeze == (:encoder,) ? 2 : 1)
                end
            end
            for opts in ((frozen=true,), (learnable=false,))
                gf,pf,sf,sch=coupled_chain(c;opts...)
                _,_,_,store=infer!(gf,sch,pf,sf)
                f=free_energy_gradient(store,gf,pf,sf;ad,schedule=sch)
                @test f.a.gain ≈ grad.a.gain atol=2e-8
                @test all(iszero,LenticulumCore._flatten(f.b))
                @test (2 in learnable_factors(gf))
                @test (3 in learnable_factors(gf)) == get(opts,:learnable,true)
            end
        end
        g,p,st,sched=coupled_chain(DiagonalCoupling())
        _,_,_,s=infer!(g,sched,p,st)
        d=free_energy_gradient(s,g,p,st;ad)
        @test d.a.gain ≈ 0.07 atol=2e-8
        @test all(iszero,LenticulumCore._flatten(d.b))
        @test_throws ArgumentError free_energy_gradient(s,first(coupled_chain(ExactCoupling())),p,st;ad)
    end
end

@testset "LP-011 mixed boundaries and restrictions" begin
    for ad in (nothing,AutoForwardDiff(),AutoZygote())
        g,p,st,sched=coupled_chain(ExactCoupling())
        # Keep a->y pathwise, cut y->b diagonally; b still learns from the loss.
        e=g.edges[3]; g.edges[3]=Edge(e.factor,e.channel,e.variable,e.direction,PathwiseCoupling())
        e=g.edges[4]; g.edges[4]=Edge(e.factor,e.channel,e.variable,e.direction,DiagonalCoupling())
        grad=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched)
        @test grad.a.gain ≈ 0.07 atol=2e-8
        @test grad.b.gain ≈ 0.79*1.3 atol=2e-8
    end
    g,p,st,sched=coupled_chain(ExactCoupling())
    @test free_energy_gradient(MessageStore(g),g,p,st;schedule=FloodingSchedule(collect(tasks(sched))),nsweeps=6).a.gain ≈ 0.1*0.7+2*1.3*0.79 atol=2e-8
    @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;schedule=sched,nsweeps=0)
    @test_throws ArgumentError Mycelium._check_pathwise(CategoricalBelief([0.3,0.7]))
    for ad in (AutoEnzyme(),AutoMooncake())
        @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched)
    end
end

struct CoupledSource{C} <: AbstractLenticulumFactor end
LenticulumCore.channels(::CoupledSource) = (LenticulumCore.Channel(:x,nothing),)
LenticulumCore.supported_polarities(::CoupledSource) = (Polarity(x=Unobserved()),)
LenticulumCore.supports_polarity(::CoupledSource,p::Polarity) = p == Polarity(x=Unobserved())
LenticulumCore.islearnable(::CoupledSource) = true
Mycelium.local_free_energy(::CoupledSource,beliefs,ps,st)=(0.0,haskey(st,:calls) ? (calls=st.calls+10,) : st)
Mycelium.factor_message(::CoupledSource{false},target::Symbol,pol,inputs,prior,ps,st) = (GaussianBelief([ps.μ],ones(1,1)),haskey(st,:calls) ? (calls=st.calls+1,) : st)
Mycelium.factor_message(::CoupledSource{true},target::Symbol,pol,inputs,prior,ps,st) = (CategoricalBelief([ps.μ,1-ps.μ]),st)
function score_graph(c;categorical=false)
    b=GraphBuilder(); variable!(b,:x)
    factor!(b,:source,CoupledSource{categorical}())
    factor!(b,:loss,LossFactor((:x,),x -> categorical ? x^2 : (only(x)+0.2)^2/2))
    connect!(b,:source,:x,:x;direction=Emitting(),coupling=ExactCoupling())
    connect!(b,:loss,:x,:x;direction=Absorbing(),coupling=c)
    g=build(b)
    return g,(source=(μ=categorical ? 0.3 : 0.8,),loss=(;)),(source=(;),loss=(;)),forward_backward_schedule(g)
end
@testset "LP-011 REINFORCE and measured baseline variance" begin
    estimates=[]
    for baseline in (nothing,2.0)
        g,p,st,sched=score_graph(ScoreFunctionCoupling(;baseline))
        estimate=free_energy_gradient(MessageStore(g),g,p,st;ad=AutoForwardDiff(),schedule=sched,rng=Xoshiro(11),nsamples=4000)
        @test estimate isa GradientEstimate
        @test abs(estimate.value.source.μ-1) < 5estimate.standard_error.source.μ
        @test estimate.nsamples==4000
        push!(estimates,estimate)
        @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;schedule=sched,nsamples=10)
        @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;schedule=sched,rng=Xoshiro(1),nsamples=1)
    end
    ratio=(estimates[2].standard_error.source.μ/estimates[1].standard_error.source.μ)^2
    @test ratio < 0.6
    @info "LP-011 Gaussian score measurement" plain=estimates[1] baseline=estimates[2] variance_ratio=ratio
    g,p,st,sched=score_graph(ScoreFunctionCoupling();categorical=true)
    e=free_energy_gradient(MessageStore(g),g,p,st;ad=AutoForwardDiff(),schedule=sched,rng=Xoshiro(13),nsamples=4000)
    # The toy source has zero local cost: its analytic Bethe variable correction
    # contributes H(p), in addition to E[state²] = 4 - 3p.
    expected = -3 + log(0.7/0.3)
    @test abs(e.value.source.μ-expected) < 5e.standard_error.source.μ
    @info "LP-011 categorical score measurement" estimate=e
    for categorical in (false,true)
        g,p,st,sched=score_graph(ScoreFunctionCoupling();categorical)
        estimate=free_energy_gradient(MessageStore(g),g,p,st;ad=AutoZygote(),schedule=sched,rng=Xoshiro(19),nsamples=256)
        target=categorical ? -3+log(0.7/0.3) : 1.0
        @test abs(estimate.value.source.μ-target) < 6estimate.standard_error.source.μ
    end
    g,p,st,sched=score_graph(ScoreFunctionCoupling())
    output=parameter_step(MessageStore(g),g,p,st;ad=AutoForwardDiff(),schedule=sched,rng=Xoshiro(20),nsamples=32)
    @test length(output)==4
    @test output[4] isa GradientEstimate
    reference=free_energy_gradient(MessageStore(g),g,p,st;schedule=sched,rng=Xoshiro(20),nsamples=32)
    @test reference.value.source.μ ≈ output[4].value.source.μ atol=1e-7
    @test reference.standard_error.source.μ ≈ output[4].standard_error.source.μ atol=1e-7
    counted=(source=(calls=0,),loss=(;))
    output=parameter_step(MessageStore(g),g,p,counted;ad=AutoForwardDiff(),schedule=sched,rng=Xoshiro(20),nsamples=32)
    @test output[2].source.calls==11
    @test counted.source.calls==0

end

@testset "LP-011 straight-through and stop primitives" begin
    x=[0.2,1.4,-0.6]
    @test straight_through(x -> round.(x),identity,x) == round.(x)
    for ad in (AutoForwardDiff(),AutoZygote())
        d=parameter_gradient(q -> sum(straight_through(x -> round.(x),identity,q.x)),(x=x,);ad)
        @test d.x == ones(3)
        fd=parameter_gradient(q -> sum(q.x),(x=x,))
        @test d.x ≈ fd.x atol=1e-9
        @test parameter_gradient(q -> sum(stop_gradient(q.x)),(x=x,);ad).x == zeros(3)
    end

    for ad in (AutoForwardDiff(),AutoZygote())
        g,p,st,sched=coupled_chain(ExactCoupling())
        g.factors[3]=FactorNode(3,:b,StraightThroughFactor(round,identity))
        p=merge(p,(a=merge(p.a,(gain=0.9,)),))
        grad=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched)
        @test grad.a.gain ≈ 2.09 atol=1e-10
        @test all(iszero,LenticulumCore._flatten(grad.b))
        @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;schedule=sched)
    end
    @test_throws ArgumentError LenticulumCore._trainable_indices((x=1.0,),(:missing,))
    @test_throws ArgumentError LenticulumCore._trainable_indices((x=1.0,),:invalid)
end
