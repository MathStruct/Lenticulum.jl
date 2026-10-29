using Test, Mycelium, LenticulumCore, LinearAlgebra, ADTypes
using DifferentiationInterface, ForwardDiff, Zygote, Krylov

# Unnormalised Gaussian potentials, independent of top-level GaussianFactor.
# Unary exp(-q*x²/2+h*x), pair exp(-k*(x-y)²/2).
struct AdjointUnary <: AbstractLenticulumFactor end
struct AdjointPair <: AbstractLenticulumFactor end
LenticulumCore.channels(::AdjointUnary) = (LenticulumCore.Channel(:x,nothing),)
LenticulumCore.channels(::AdjointPair) = (LenticulumCore.Channel(:x,nothing),LenticulumCore.Channel(:y,nothing))
LenticulumCore.supported_polarities(::AdjointUnary) = (Polarity(x=Unobserved()),)
LenticulumCore.supported_polarities(::AdjointPair) = (Polarity(x=Unobserved(),y=Observed()),Polarity(x=Observed(),y=Unobserved()))
LenticulumCore.supports_polarity(f::Union{AdjointUnary,AdjointPair},p::Polarity) = p in supported_polarities(f)
LenticulumCore.islearnable(::Union{AdjointUnary,AdjointPair}) = true
Mycelium.uses_cavity(::Union{AdjointUnary,AdjointPair}) = true
Mycelium.factor_message(::AdjointUnary,target::Symbol,pol,inputs,prior,ps,st) = (GaussianBelief([ps.h],fill(ps.q,1,1)), st)
function Mycelium.factor_message(::AdjointPair,target::Symbol,pol,inputs,prior,ps,st)
    b = inputs[target === :x ? :y : :x]
    q,h = only(b.Λ),only(b.η)
    return GaussianBelief([ps.k*h/(ps.k+q)],fill(ps.k*q/(ps.k+q),1,1)),st
end

adjoint_qh(::Union{Nothing,TrivialBelief}) = (0.0,0.0)
adjoint_qh(b::GaussianBelief) = (only(b.Λ),only(b.η))
function Mycelium.local_free_energy(::AdjointUnary,beliefs,ps,st)
    q,h = adjoint_qh(beliefs.x)
    C = inv(ps.q+q); m = C*(ps.h+h)
    return ps.q*(m*m+C)/2-ps.h*m-log(2π*exp(1)*C)/2,st
end
function Mycelium.local_free_energy(::AdjointPair,beliefs,ps,st)
    qx,hx = adjoint_qh(beliefs.x); qy,hy = adjoint_qh(beliefs.y)
    qx+qy == 0 && return 0.0,st # no normalisable joint yet during startup
    Λ = [ps.k+qx -ps.k; -ps.k ps.k+qy]
    C = inv(Λ); m = C*[hx,hy]
    H = (2*(1+log(2π))-logdet(Λ))/2
    return ps.k*((m[1]-m[2])^2+C[1,1]+C[2,2]-2C[1,2])/2-H,st
end

function adjoint_gaussian_graph(;loopy=false,coupling=ExactCoupling(),cut=false,diagonal=false,freeze=())
    b=GraphBuilder()
    for v in (:x,:y,:z); variable!(b,v); end
    for (f,v) in ((:u,:x),(:v,:y),(:w,:z))
        factor!(b,f,AdjointUnary();freeze=f===:u ? freeze : ())
        connect!(b,f,:x,v;coupling)
    end
    pairs=loopy ? ((:a,:x,:y),(:b,:y,:z),(:c,:z,:x)) : ((:a,:x,:y),(:b,:y,:z))
    for (f,x,y) in pairs
        factor!(b,f,AdjointPair())
        connect!(b,f,:x,x;coupling=f===:a && diagonal ? DiagonalCoupling() : coupling,detach=f===:a && cut)
        connect!(b,f,:y,y;coupling)
    end
    g=build(b)
    p=(u=(q=0.7,h=0.4),v=(q=0.9,h=-0.3),w=(q=0.6,h=0.1),a=(k=2.0,),b=(k=1.3,))
    loopy && (p=merge(p,(c=(k=1.7,),)))
    st=NamedTuple{keys(p)}(map(_ -> (;),values(p)))
    return g,p,st
end
adjoint_flat(p) = LenticulumCore._flatten(p)
function adjoint_reference_loss(p,g,st,sched,n,damping)
    s=MessageStore(g)
    for _ in 1:n; _,st=sweep!(s,g,sched,p,st;damping); end
    return first(scalar_free_energy(s,g,p,st))
end

@testset "LP-002 C realised flooding/residual/damped replay" begin
    for loopy in (false,true), coupling in (ExactCoupling(),PathwiseCoupling())
        g,p,st=adjoint_gaussian_graph(;loopy,coupling)
        for sched in (SequentialSchedule(all_tasks(g)),flooding_schedule(g),ResidualSchedule(maxtasks=5)), damping in (0.0,0.35)
            n=12
            ops=Mycelium._realised_tasks(g,sched,n,p,st;damping)
            tape,store,state=Mycelium._reference_tape(g,ops,p,st,nothing,damping)
            reference=adjoint_reference_loss(p,g,st,sched,n,damping)
            @test Mycelium._graph_scalar(store,g,p,state) ≈ reference atol=2e-12
            # Residual choices are fixed before perturbation, including at ties.
            realised=sched isa ResidualSchedule ? SequentialSchedule(ops) : sched
            count=sched isa ResidualSchedule ? 1 : n
            fd=parameter_gradient(q -> adjoint_reference_loss(q,g,st,realised,count,damping),p)
            for ad in (nothing,AutoForwardDiff(),AutoZygote())
                grad=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched,nsweeps=n,damping)
                @test adjoint_flat(grad) ≈ adjoint_flat(fd) atol=5e-7 rtol=5e-7
            end
            # Raw ChainRules inference uses the realised count/choices and damping.
            # SequentialSchedule avoids TreeSchedule's intentional one-sweep stop.
            raw=parameter_gradient(p;ad=AutoZygote()) do q
                _,_,_,s=infer!(g,sched,q,st;maxsweeps=n,tol=0.0,damping)
                first(scalar_free_energy(s,g,q,st))
            end
            @test adjoint_flat(raw) ≈ adjoint_flat(fd) atol=5e-7 rtol=5e-7
        end
    end
end

@testset "LP-002 C implicit graph adjoint and Gaussian closed forms" begin
    reference_errors=Float64[]
    for loopy in (false,true), coupling in (ExactCoupling(),PathwiseCoupling())
        g,p,st=adjoint_gaussian_graph(;loopy,coupling)
        sched=loopy ? flooding_schedule(g) : tree_schedule(g)
        _,report,_,s=infer!(g,sched,p,st;tol=1e-12,maxsweeps=1000)
        @test report.converged
        # Dense exact global Gaussian oracle. Bethe is exact on the tree only;
        # loopy Bethe references reconverge at every parameter perturbation.
        function closed(q)
            Q=diagm([q.u.q,q.v.q,q.w.q])+q.a.k*[1 -1 0;-1 1 0;0 0 0]+q.b.k*[0 0 0;0 1 -1;0 -1 1]
            h=[q.u.h,q.v.h,q.w.h]
            return (logdet(Q)-3log(2π)-dot(h,Q\h))/2
        end
        reference=parameter_gradient(p) do q
            if loopy
                _,r,_,ss=infer!(g,sched,q,st;tol=1e-13,maxsweeps=1000)
                r.converged || error("reference not converged")
                first(scalar_free_energy(ss,g,q,st))
            else
                closed(q)
            end
        end
        if !loopy
            @test first(scalar_free_energy(s,g,p,st)) ≈ closed(p) atol=1e-12
        end
        for ad in (nothing,AutoForwardDiff(),AutoZygote()), solver in (:dense,:krylov)
            grad=free_energy_gradient(s,g,p,st;ad,adjoint=:implicit,schedule=sched,
                linear_solver=solver,tol=1e-12,linear_tol=ad===nothing ? 2e-7 : 1e-10)
            @test adjoint_flat(grad) ≈ adjoint_flat(reference) atol=5e-7 rtol=5e-7
            push!(reference_errors,norm(adjoint_flat(grad)-adjoint_flat(reference))/max(1,norm(adjoint_flat(reference))))
            unroll=free_energy_gradient(s,g,p,st;ad,schedule=sched,nsweeps=loopy ? 100 : 2)
            @test adjoint_flat(grad) ≈ adjoint_flat(unroll) atol=5e-7 rtol=5e-7
        end
        if loopy && coupling isa ExactCoupling
            implicit=free_energy_gradient(s,g,p,st;ad=AutoForwardDiff(),adjoint=:implicit,schedule=sched,tol=1e-12)
            errors=map((4,8,16,32,64)) do n
                d=free_energy_gradient(s,g,p,st;ad=AutoForwardDiff(),schedule=sched,nsweeps=n)
                norm(adjoint_flat(d)-adjoint_flat(implicit))
            end
            @test all(diff(collect(errors)) .< 0)
            @test last(errors) < 1e-8
            @info "LP-002 C unrolled convergence to implicit Bethe gradient" sweeps=(4,8,16,32,64) errors
        end
    end
    @info "LP-002 C maximum scaled implicit gradient error to Gaussian references" error=maximum(reference_errors)
end

@testset "LP-002 C implicit cuts, freezing, damping, failures" begin
    for opts in ((cut=true,),(diagonal=true,),(freeze=(:h,),),(freeze=:all,))
        g,p,st=adjoint_gaussian_graph(;loopy=true,opts...)
        sched=SequentialSchedule(all_tasks(g))
        for ad in (nothing,AutoForwardDiff(),AutoZygote())
            d=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched,adjoint=:implicit,tol=1e-12,
                linear_tol=ad===nothing ? 2e-7 : 1e-10)
            u=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched,nsweeps=60)
            @test adjoint_flat(d) ≈ adjoint_flat(u) atol=5e-7 rtol=5e-7
            damped=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=ResidualSchedule(maxtasks=10),
                damping=0.3,adjoint=:implicit,tol=1e-12,linear_tol=ad===nothing ? 2e-7 : 1e-10)
            @test adjoint_flat(damped) ≈ adjoint_flat(d) atol=5e-7 rtol=5e-7
        end
    end
    gd,pd,sd=adjoint_gaussian_graph(;loopy=true,coupling=DiagonalCoupling())
    dsched=flooding_schedule(gd)
    _,_,_,ds=infer!(gd,dsched,pd,sd;tol=1e-12,maxsweeps=1000)
    for ad in (nothing,AutoForwardDiff(),AutoZygote())
        fixed=free_energy_gradient(ds,gd,pd,sd;ad)
        @test any(!iszero,adjoint_flat(fixed))
        for solver in (:dense,:krylov)
            d=free_energy_gradient(ds,gd,pd,sd;ad,schedule=dsched,adjoint=:implicit,
                linear_solver=solver,tol=1e-12,linear_tol=ad===nothing ? 2e-7 : 1e-10)
            @test adjoint_flat(d) ≈ adjoint_flat(fixed) atol=2e-8 rtol=2e-8
        end
    end
    g,p,st=adjoint_gaussian_graph(;loopy=true)
    for kw in ((maxsweeps=1,), (linear_solver=:bad,), (tol=0.0,), (linear_maxiter=0,),
            (schedule=SequentialSchedule(all_tasks(g)[1:1]),), (adjoint=:bad,), (damping=1.0,))
        args=merge((schedule=flooding_schedule(g),adjoint=:implicit),kw)
        @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;args...)
    end
    @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,merge(st,(u=(counter=1,),));schedule=flooding_schedule(g),adjoint=:implicit)
    @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;ad=AutoZygote(),schedule=flooding_schedule(g),adjoint=:implicit,linear_solver=:krylov,linear_maxiter=1)
    # A zero-budget ResidualSchedule is now refused at construction (LP-019 item 9).
    @test_throws ArgumentError ResidualSchedule(maxtasks=0)
    for sched in (SequentialSchedule([MessageTask(0,:to_factor)]),
            SequentialSchedule([MessageTask(1,:bad)]))
        for adjoint in (:unrolled,:implicit)
            @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;schedule=sched,adjoint)
        end
    end
    for ad in (AutoEnzyme(),AutoMooncake()), adjoint in (:unrolled,:implicit)
        @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;ad,
            schedule=flooding_schedule(g),adjoint)
    end
    edge=g.edges[1]
    g.edges[1]=Edge(edge.factor,edge.channel,edge.variable,edge.direction,ScoreFunctionCoupling())
    @test_throws (msg -> occursin("score estimators are unsupported",msg)) free_energy_gradient(
        MessageStore(g),g,p,st;schedule=flooding_schedule(g),adjoint=:implicit)
end

struct AdjointDirac <: AbstractLenticulumFactor end
LenticulumCore.channels(::AdjointDirac) = channels(AdjointUnary())
LenticulumCore.supported_polarities(::AdjointDirac) = supported_polarities(AdjointUnary())
LenticulumCore.supports_polarity(::AdjointDirac,p::Polarity) = p in supported_polarities(AdjointUnary())
LenticulumCore.islearnable(::AdjointDirac) = true
Mycelium.factor_message(::AdjointDirac,t::Symbol,pol,inputs,prior,p,st) = (DiracBelief(p.a),st)
Mycelium.local_free_energy(::AdjointDirac,bs,p,st) = (0.0,st)

struct AdjointIntroducesState <: AbstractLenticulumFactor end
LenticulumCore.channels(::AdjointIntroducesState) = channels(AdjointDirac())
LenticulumCore.supported_polarities(::AdjointIntroducesState) = supported_polarities(AdjointDirac())
LenticulumCore.supports_polarity(::AdjointIntroducesState,p::Polarity) = p in supported_polarities(AdjointDirac())
Mycelium.factor_message(::AdjointIntroducesState,t::Symbol,pol,inputs,prior,p,st) = (DiracBelief(0.7),(counter=1,))
Mycelium.local_free_energy(::AdjointIntroducesState,bs,p,st) = (0.0,st)

struct SingularGraphContext{A}
    x::Vector{Float64}
    ps::NamedTuple{(),Tuple{}}
    ad::A
    ε::Float64
end
Mycelium._graph_state_map(x,p,c::SingularGraphContext) = x

@testset "LP-002 C Dirac fixed point, parameter step and singular system" begin
    b=GraphBuilder(); variable!(b,:x)
    factor!(b,:source,AdjointDirac()); factor!(b,:loss,LossFactor((:x,),x->x*x/2))
    connect!(b,:source,:x,:x;direction=Emitting(),coupling=ExactCoupling())
    connect!(b,:loss,:x,:x;direction=Absorbing(),coupling=ExactCoupling())
    g=build(b); p=(source=(a=0.7,),loss=(;)); st=(source=(;),loss=(;))
    for ad in (nothing,AutoForwardDiff(),AutoZygote()), solver in (:dense,:krylov)
        kw=(;ad,schedule=flooding_schedule(g),adjoint=:implicit,linear_solver=solver)
        d=free_energy_gradient(MessageStore(g),g,p,st;kw...)
        @test d.source.a ≈ p.source.a atol=1e-8
        pn,sn,on=parameter_step(MessageStore(g),g,p,st;kw...)
        @test pn.source.a ≈ 0.99p.source.a atol=1e-8
        @test sn == st
    end
    for (coupling,detach) in ((DiagonalCoupling(),false),(ExactCoupling(),true))
        for i in eachindex(g.edges)
            e=g.edges[i]
            g.edges[i]=Edge(e.factor,e.channel,e.variable,e.direction,coupling,detach)
        end
        _,_,_,store=infer!(g,flooding_schedule(g),p,st)
        for ad in (nothing,AutoForwardDiff(),AutoZygote()), solver in (:dense,:krylov)
            d=free_energy_gradient(store,g,p,st;ad,schedule=flooding_schedule(g),
                adjoint=:implicit,linear_solver=solver)
            @test d.source.a == 0.0
            @test d == free_energy_gradient(store,g,p,st;ad,schedule=flooding_schedule(g),nsweeps=3)
        end
    end
    ctx=SingularGraphContext([0.0],(;),AutoForwardDiff(),1e-6)
    @test_throws ArgumentError Mycelium._graph_linear_solve(Val(:dense),nothing,[1.0],ctx,1e-10,100)
    b2=GraphBuilder(); variable!(b2,:x); factor!(b2,:source,AdjointIntroducesState())
    connect!(b2,:source,:x,:x;direction=Emitting(),coupling=ExactCoupling())
    g2=build(b2)
    @test_throws (msg -> occursin("preserve empty factor state",msg)) free_energy_gradient(
        MessageStore(g2),g2,(source=(;),),(source=(;),);schedule=flooding_schedule(g2),adjoint=:implicit)
end

# Correlated two-dimensional messages exercise full precision coordinates,
# including symmetric off-diagonal directions (the scalar fixture cannot).
struct AdjointDenseUnary <: AbstractLenticulumFactor end
LenticulumCore.channels(::AdjointDenseUnary) = channels(AdjointUnary())
LenticulumCore.supported_polarities(::AdjointDenseUnary) = supported_polarities(AdjointUnary())
LenticulumCore.supports_polarity(::AdjointDenseUnary,p::Polarity) = p in supported_polarities(AdjointUnary())
LenticulumCore.islearnable(::AdjointDenseUnary) = true
Mycelium.uses_cavity(::AdjointDenseUnary) = true
adjoint_symmetric(Q) = (Q + Q') / 2
Mycelium.factor_message(::AdjointDenseUnary,t::Symbol,pol,inputs,prior,p,st) =
    (GaussianBelief(p.h,adjoint_symmetric(p.Q)),st)
function Mycelium.local_free_energy(::AdjointDenseUnary,bs,p,st)
    Q = adjoint_symmetric(p.Q)
    cavity = bs.x
    Λ = Q + cavity.Λ
    C = inv(Λ); m = C * (p.h + cavity.η)
    H = (2*(1+log(2π))-logdet(Λ))/2
    return (dot(m,Q*m)+tr(Q*C))/2-dot(p.h,m)-H,st
end

@testset "LP-002 C correlated dense Gaussian message coordinates" begin
    b=GraphBuilder(); variable!(b,:x)
    for f in (:a,:b)
        factor!(b,f,AdjointDenseUnary())
        connect!(b,f,:x,:x;coupling=ExactCoupling())
    end
    g=build(b)
    p=(a=(Q=[1.2 0.3;0.3 0.8],h=[0.4,-0.2]),
       b=(Q=[0.7 -0.1;-0.1 1.1],h=[-0.1,0.3]))
    st=(a=(;),b=(;)); sched=tree_schedule(g)
    function closed(q)
        Q=adjoint_symmetric(q.a.Q+q.b.Q); h=q.a.h+q.b.h
        (logdet(Q)-2log(2π)-dot(h,Q\h))/2
    end
    _,_,_,store=infer!(g,sched,p,st)
    @test first(scalar_free_energy(store,g,p,st)) ≈ closed(p) atol=1e-12
    reference=parameter_gradient(closed,p;ad=AutoForwardDiff())
    for ad in (nothing,AutoForwardDiff(),AutoZygote()), solver in (:dense,:krylov)
        d=free_energy_gradient(store,g,p,st;ad,schedule=sched,adjoint=:implicit,
            linear_solver=solver,linear_tol=ad===nothing ? 2e-7 : 1e-10)
        u=free_energy_gradient(store,g,p,st;ad,schedule=sched,nsweeps=2)
        @test adjoint_flat(d) ≈ adjoint_flat(reference) atol=2e-8 rtol=2e-8
        @test adjoint_flat(d) ≈ adjoint_flat(u) atol=2e-8 rtol=2e-8
    end
end
