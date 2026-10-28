# LP-002 batch B: no hardware GPU, no scalar indexing allowances.
using JLArrays
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote, Krylov
const DEVICE_DI = DifferentiationInterface

struct DeviceCell <: LuxCore.AbstractLuxLayer end
(::DeviceCell)(inp, ps, st) = (ps.W * inp[1] .+ ps.U * inp[2] .+ ps.b, st)
struct DeviceDynamics <: LuxCore.AbstractLuxLayer end
(::DeviceDynamics)(inp, ps, st) = (ps.A * inp[1], st)
device_tree(x::AbstractArray) = JLArray(x)
device_tree(x::NamedTuple) = map(device_tree, x)
device_tree(x) = x
host_tree(x::AbstractArray) = Array(x)
host_tree(x::NamedTuple) = map(host_tree, x)
host_tree(x) = x

reverse_tuple(t) = (t[2],t[1])

device_tol(::Type{Float32}) = 2f-5
device_tol(::Type{Float64}) = 1e-9
const DEVICE_ERRORS = Dict(Float32=>0.0,Float64=>0.0)
function device_check(got, want, T; tol=device_tol(T))
    @test got isa JLArray || got isa SubArray{<:Any,<:Any,<:JLArray}
    @test eltype(got) === T
    host = Array(got)
    @test host ≈ want rtol=tol atol=tol
    DEVICE_ERRORS[T] = max(DEVICE_ERRORS[T], norm(host-want)/max(1,norm(want)))
end

@testset "LP-002 device DEQ solves and IFT" begin
    JLArrays.allowscalar(false)
    for T in (Float32, Float64)
        ps = (W=T[0.2 0.1; -0.05 0.3], U=T[-0.8 0.1; 0.05 -0.9], b=T[0.2,-0.1])
        x = T[0.7,-0.3]
        z = (I-ps.W) \ (ps.U*x+ps.b)
        dx, dp = JLArray(x), device_tree(ps)
        tol = T === Float32 ? 2e-6 : 1e-11
        linear_tol = T === Float32 ? 3e-6 : 1e-10
        solvers = (PicardSolver(;tol,maxiters=100), BroydenSolver(;tol), LimitedBroydenSolver(;tol,memory=5))
        Id = Matrix{T}(I,2,2)
        exact = (I-ps.W) \ hcat(kron(z',Id),kron(x',Id),Id)
        y = z + T[0.2,-0.15]
        Ry = hcat(kron(y',Id),kron(x',Id),Id)
        r = (I-ps.W)*y-ps.U*x-ps.b
        for solver in (PicardSolver(damping=0.7,maxiters=1,tol=0),
                       BroydenSolver(init_scale=0.7,maxiters=1,tol=0),
                       LimitedBroydenSolver(init_scale=0.7,maxiters=1,tol=0))
            out, _ = solve_root(v -> v .- (dp.W*v .+ dp.U*dx .+ dp.b), zero(dx), solver)
            device_check(out,T(0.7).*(ps.U*x+ps.b),T)
        end
        for solver in solvers
            f = DEQFactor(DeviceCell(), (x=2,z=2);solver)
            out, report, _ = solve_state(f,dx,dp,(;))
            cpu = first(solve_state(f,x,ps,(;)))
            @test report.converged
            device_check(out,z,T)
            @test Array(out) ≈ cpu rtol=device_tol(T) atol=device_tol(T)
            # No LU exists on JLArrays: fail clearly instead of scalar indexing.
            forward = DEQFactor(DeviceCell(), (x=2,z=2);solver,
                ift=KrylovIFT(rtol=linear_tol,atol=linear_tol/10,memory=2,itmax=40))
            for ad in (AutoForwardDiff(), AutoZygote())
                grad=parameter_gradient(p -> sum(abs2,first(solve_state(forward,dx,p,(;);ad)))/2,dp;ad)
                device_check(LenticulumCore._flatten(grad),exact'z,T)
                @test_throws (msg -> startswith(msg,"ArgumentError:") && occursin("device LU",msg)) parameter_gradient(
                    p -> sum(abs2,first(solve_state(f,dx,p,(;);ad)))/2, dp;ad)
                @test_throws (msg -> startswith(msg,"ArgumentError:") && occursin("device LU",msg)) parameter_gradient(
                    p -> sum(abs2,first(solve_input(f,JLArray(y),p,(;);ad)))/2, dp;ad)
            end
        end
        for (precond,ldiv) in ((nothing,false), (JacobiIFT(),false), (BroydenIFT(),false),
                        (ctx -> (;left=Diagonal(JLArray(inv.(diag(ctx.transposed ? (I-ps.W)' : I-ps.W))))),false),
                        (ctx -> (;right=Diagonal(JLArray(inv.(diag(ctx.transposed ? (I-ps.W)' : I-ps.W))))),false),
                        (ctx -> (;left=Diagonal(JLArray(diag(ctx.transposed ? (I-ps.W)' : I-ps.W)))),true))
            f = DEQFactor(DeviceCell(), (x=2,z=2); solver=solvers[3],
                ift=KrylovIFT(;precond,ldiv,rtol=linear_tol,atol=linear_tol/10,memory=2,itmax=40))
            for ad in (AutoForwardDiff(), AutoZygote())
                g = parameter_gradient(p -> sum(abs2,first(solve_state(f,dx,p,(;);ad)))/2,dp;ad)
                device_check(LenticulumCore._flatten(g), exact'z,T)
                gx = LenticulumCore._gradient(ad, x -> sum(abs2, first(solve_state(f,x,dp,(;);ad)))/2,dx)
                device_check(gx, ((I-ps.W)\ps.U)'z,T)
                # Explicit active warm start: carries no tangent.
                gw = LenticulumCore._gradient(ad, warm -> sum(first(solve_state(f,dx,dp,(;);ad,z₀=warm))),dx)
                device_check(gw === nothing ? zero(dx) : gw,zeros(T,2),T)
            end
        end
        # Reverse polarity: -U is contractive for Picard on this fixture.
        reverse_exact = -(ps.U \ hcat(kron(y',Id),kron((ps.U\((I-ps.W)*y-ps.b))',Id),Id))
        xrev = ps.U \ ((I-ps.W)*y-ps.b)
        for solver in solvers
            for precond in (nothing, JacobiIFT(), BroydenIFT())
                precond isa BroydenIFT && !(solver isa LimitedBroydenSolver) && continue
                f = DEQFactor(DeviceCell(), (x=2,z=2);solver,
                    ift=KrylovIFT(;precond,rtol=linear_tol,atol=linear_tol/10,memory=2,itmax=40))
                out, rep, _ = solve_input(f,JLArray(y),dp,(;))
                @test rep.converged
                device_check(out,xrev,T)
                for ad in (AutoForwardDiff(), AutoZygote())
                    g=parameter_gradient(p -> sum(abs2,first(solve_input(f,JLArray(y),p,(;);ad)))/2,dp;ad)
                    device_check(LenticulumCore._flatten(g),reverse_exact'xrev,T)
                    gy=LenticulumCore._gradient(ad,v -> sum(abs2,first(solve_input(f,v,dp,(;);ad)))/2,JLArray(y))
                    device_check(gy,(ps.U\(I-ps.W))'xrev,T)
                end
            end
        end
        # Pointwise parameter entry points, independent of root rules.
        f=DEQFactor(DeviceCell(),(x=2,z=2);solver=solvers[3])
        for ad in (AutoForwardDiff(),AutoZygote())
            eg=energy_gradient(f,dx,nothing,JLArray(y),dp,(;);ad)
            fg=free_energy_gradient(f,DiracBelief(dx),DiracBelief(JLArray(y)),dp,(;);ad)
            device_check(LenticulumCore._flatten(eg),-Ry'r,T)
            device_check(LenticulumCore._flatten(fg),-Ry'r,T)
        end
        for ad in (AutoEnzyme(),AutoMooncake())
            @test_throws ArgumentError energy_gradient(f,dx,nothing,JLArray(y),dp,(;);ad)
            @test_throws ArgumentError solve_state(f,dx,dp,(;);ad)
            @test_throws ArgumentError solve_input(f,JLArray(y),dp,(;);ad)
        end
    end
end

@testset "LP-002 device Euler and RK4 flows and gradients" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64), itg in (EulerIntegrator(4),RK4Integrator(4)), policy in (:threaded,:fixed)
        ps=(A=T[-0.2 0.1;-0.05 -0.3],)
        x=T[0.7,-0.3]; dp=device_tree(ps); dx=JLArray(x)
        f=NeuralODEFactor(DeviceDynamics(),2;integrator=itg,state_policy=policy,tspan=(0.0,0.8))
        for reverse in (false,true)
            flow=reverse ? flow_reverse : flow_forward
            h=T((reverse ? -0.8 : 0.8)/4)
            B=h*ps.A; Id=Matrix{T}(I,2,2)
            P=itg isa EulerIntegrator ? Id+B : Id+B+B^2/2+B^3/6+B^4/24
            want=P^4*x
            out=first(flow(f,dx,dp,(;)))
            device_check(out,want,T)
            @test Array(out) ≈ first(flow(f,x,ps,(;))) atol=device_tol(T) rtol=device_tol(T)
            if policy === :threaded
                t0,t1=reverse ? reverse_tuple(f.tspan) : f.tspan
                hc=(t1-t0)/4
                legacy=foldl(1:4;init=(x,(;))) do (z,st),k
                    ImplicitLayers._lux_step(f,z,ps,st,t0+(k-1)*hc,hc,itg)
                end
                @test isequal(first(flow(f,x,ps,(;))),first(legacy))
            end
            for ad in (AutoForwardDiff(),AutoZygote())
                gx=LenticulumCore._gradient(ad, x -> sum(abs2,first(flow(f,x,dp,(;))))/2,dx)
                device_check(gx,(P^4)'want,T)
                # Dense CPU differentiation of the known discrete polynomial.
                closed=p -> begin
                    B=h*reshape(p,2,2)
                    Q=itg isa EulerIntegrator ? Id+B : Id+B+B^2/2+B^3/6+B^4/24
                    sum(abs2,Q^4*x)/2
                end
                wantg=ForwardDiff.gradient(closed,vec(ps.A))
                g=parameter_gradient(p -> sum(abs2,first(flow(f,dx,p,(;))))/2,dp;ad)
                device_check(vec(g.A),wantg,T)
            end
        end
        y=T[0.4,-0.1]
        for ad in (AutoForwardDiff(),AutoZygote())
            eg=energy_gradient(f,dx,nothing,JLArray(y),dp,(;);ad)
            cpu=energy_gradient(f,x,nothing,y,ps,(;);ad=AutoForwardDiff())
            device_check(vec(eg.A),vec(cpu.A),T)
            fg=free_energy_gradient(f,DiracBelief(dx),DiracBelief(JLArray(y)),dp,(;);ad)
            device_check(vec(fg.A),vec(cpu.A),T)
        end
    end
end


# A terminal quadratic gives exact Gaussian expectation, so Gaussian replay tests
# means, precisions and variable entropy without stochastic sampling noise.
struct DeviceSource{G} <: AbstractLenticulumFactor end
LenticulumCore.channels(::DeviceSource) = (LenticulumCore.Channel(:x,nothing),)
LenticulumCore.supported_polarities(::DeviceSource) = (Polarity(x=Unobserved()),)
LenticulumCore.supports_polarity(::DeviceSource,p::Polarity) = p == Polarity(x=Unobserved())
LenticulumCore.islearnable(::DeviceSource) = true
Mycelium.local_free_energy(::DeviceSource,b,ps,st) = (sum(abs2,ps.μ)/20,st)
Mycelium.factor_message(::DeviceSource{false},target::Symbol,pol,inputs,prior,ps,st) = (DiracBelief(ps.μ),st)
Mycelium.factor_message(::DeviceSource{true},target::Symbol,pol,inputs,prior,ps,st) =
    (GaussianBelief(ps.λ .* ps.μ,Diagonal(ps.λ)),st)
struct DeviceQuadratic{V} <: AbstractLenticulumFactor
    target::V
end
LenticulumCore.channels(::DeviceQuadratic) = (LenticulumCore.Channel(:x,nothing),)
LenticulumCore.supported_polarities(::DeviceQuadratic) = (Polarity(x=Observed()),)
LenticulumCore.islearnable(::DeviceQuadratic) = false
Mycelium.issink(::DeviceQuadratic) = true
Mycelium.local_free_energy(f::DeviceQuadratic,bs,ps,st) =
    (device_quadratic(bs.x,f.target),st)
device_quadratic(b::DiracBelief,y) = sum(abs2,b.value-y)/2
device_quadratic(b::GaussianBelief,y) = sum(abs2,belief_mean(b)-y)/2 + sum(inv.(b.Λ.diag))/2

function device_graph(T,G,coupling;device=true,detach=false,freeze=())
    cv(x)=device ? JLArray(x) : x
    b=GraphBuilder(); variable!(b,:x)
    factor!(b,:source,DeviceSource{G}();freeze)
    factor!(b,:loss,DeviceQuadratic(cv(T[0.2,-0.1])))
    connect!(b,:source,:x,:x;direction=Emitting(),coupling,detach)
    connect!(b,:loss,:x,:x;direction=Absorbing(),coupling)
    g=build(b)
    ps=(source=(μ=cv(T[0.7,-0.3]),λ=cv(T[1.4,0.8])),loss=(;))
    st=(source=(;),loss=(;))
    return g,ps,st,forward_backward_schedule(g)
end
@testset "LP-011 device Gaussian/Dirac replay, cuts and parameter steps" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64), gaussian in (false,true), c in (ExactCoupling(),PathwiseCoupling(),DiagonalCoupling()),
        opts in ((;), (detach=true,), (freeze=(:λ,),), (freeze=:all,))
        g,ps,st,sched=device_graph(T,gaussian,c;opts...)
        gc,pc,sc,csched=device_graph(T,gaussian,c;device=false,opts...)
        _,_,_,store=infer!(g,sched,ps,st)
        _,_,_,cstore=infer!(gc,csched,pc,sc)
        coupled=!(c isa DiagonalCoupling)
        kw=coupled ? (;schedule=sched) : (;)
        ckw=coupled ? (;schedule=csched) : (;)
        cut=!coupled || get(opts,:detach,false)
        wantμ=pc.source.μ/10 + (cut ? zero(pc.source.μ) : pc.source.μ-T[0.2,-0.1])
        wantλ=gaussian && !cut ? -inv.(pc.source.λ).^2/2 - inv.(pc.source.λ)/2 : zero(pc.source.λ)
        get(opts,:freeze,()) == (:λ,) && (wantλ=zero(wantλ))
        if get(opts,:freeze,()) === :all
            wantμ=zero(wantμ); wantλ=zero(wantλ)
        end
        for ad in (AutoForwardDiff(),AutoZygote())
            grad=free_energy_gradient(store,g,ps,st;ad,kw...)
            cpu=free_energy_gradient(cstore,gc,pc,sc;ad=AutoForwardDiff(),ckw...)
            device_check(grad.source.μ,wantμ,T)
            device_check(grad.source.λ,wantλ,T)
            @test Array(grad.source.μ) ≈ cpu.source.μ atol=device_tol(T) rtol=device_tol(T)
            # Explicit and default learning rates preserve device parameter precision.
            η=T(0.03)
            pn,sn,on=parameter_step(store,g,ps,st;ad,rule=GradientDescent(η),kw...)
            device_check(pn.source.μ,pc.source.μ-η*wantμ,T)
            device_check(pn.source.λ,pc.source.λ-η*wantλ,T)
            defaultps,_,_=parameter_step(store,g,ps,st;ad,kw...)
            device_check(defaultps.source.μ,pc.source.μ-T(0.01)*wantμ,T)
            @test sn == st
            @test haskey(on,:source) == (get(opts,:freeze,()) !== :all)
        end
    end
    for T in (Float32,Float64)
        b=GaussianBelief(JLArray(T[1,2]),JLArray(Matrix{T}(I,2,2)))
        @test_throws ArgumentError belief_mean(b)
        @test_throws ArgumentError belief_cov(b)
        bad=GaussianBelief(JLArray(T[1,2]),Diagonal(JLArray(T[1,0])))
        @test !isproper(bad)
        @test_throws ArgumentError belief_mean(bad)
    end
end

@testset "LP-011 device score estimates and raw replay" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64), ad in (AutoForwardDiff(),AutoZygote())
        g,p,st,sched=device_graph(T,true,ExactCoupling())
        gc,pc,sc,csched=device_graph(T,true,ExactCoupling();device=false)
        # Exact/pathwise raw infer! follows the same realised static schedule.
        raw=parameter_gradient(p;ad) do q
            _,_,_,s=infer!(g,sched,q,st)
            first(scalar_free_energy(s,g,q,st))
        end
        device_check(raw.source.μ,pc.source.μ/10+pc.source.μ-T[0.2,-0.1],T)
        device_check(raw.source.λ,-inv.(pc.source.λ).^2/2-inv.(pc.source.λ)/2,T)
        for (graph, scheduler) in ((g,sched),(gc,csched))
            edge=graph.edges[2]
            graph.edges[2]=Edge(edge.factor,edge.channel,edge.variable,edge.direction,ScoreFunctionCoupling(baseline=0.2))
        end
        n=8
        est=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched,rng=Xoshiro(42),nsamples=n)
        cpu=free_energy_gradient(MessageStore(gc),gc,pc,sc;ad=AutoForwardDiff(),schedule=csched,rng=Xoshiro(42),nsamples=n)
        device_check(est.value.source.μ,cpu.value.source.μ,T)
        device_check(est.value.source.λ,cpu.value.source.λ,T)
        device_check(est.standard_error.source.μ,cpu.standard_error.source.μ,T)
        device_check(est.standard_error.source.λ,cpu.standard_error.source.λ,T)
        @test est.nsamples==n
        step=parameter_step(MessageStore(g),g,p,st;ad,schedule=sched,rng=Xoshiro(42),nsamples=n)
        @test length(step)==4
        device_check(step[1].source.μ,pc.source.μ-T(0.01)*cpu.value.source.μ,T)
        # Diagonal Gaussian accessors use only elementwise operations/reductions.
        b=Gaussian(JLArray(pc.source.μ),Diagonal(JLArray(inv.(pc.source.λ))))
        device_check(belief_mean(b),pc.source.μ,T)
        device_check(belief_cov(b).diag,inv.(pc.source.λ),T)
        for adbad in (AutoEnzyme(),AutoMooncake())
            @test_throws ArgumentError free_energy_gradient(MessageStore(g),g,p,st;ad=adbad,schedule=sched,rng=Xoshiro(42),nsamples=n)
        end
    end
end

@testset "LP-002 device unsupported references and optimiser coefficients" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64)
        x=JLArray(T[0.4,-0.2]); ps=(b=copy(x),)
        @test_throws ArgumentError parameter_gradient(p -> sum(abs2,p.b),ps)
        @test_throws ArgumentError ImplicitLayers.fd_jacobian(identity,x)
        @test_throws ArgumentError ImplicitLayers._divergence((x,t)->x,x,0.0)
        for ad in (AutoForwardDiff(),AutoZygote())
            @test_throws ArgumentError parameter_gradient(p -> p.s*sum(p.b),(s=T(1),b=x);ad)
        end
        for rule in (GradientDescent(0.03),Momentum(0.03,0.8),Nesterov(0.03,0.8))
            state=JLArray(T[0.1,0.2]); grad=JLArray(T[0.3,-0.4])
            look=rule_get(rule,state,x)
            cpu=rule_get(rule,Array(state),Array(x))
            device_check(look,cpu,T)
            sn,pn=rule_put(rule,state,x,grad)
            sc,pc=rule_put(rule,Array(state),Array(x),Array(grad))
            device_check(sn,sc,T); device_check(pn,pc,T)
        end
        f=DEQFactor(DeviceCell(),(x=2,z=2))
        @test_throws ArgumentError deq_hvp(sum,f,x,ps,(;),x;ad=AutoForwardDiff())
    end
end

@testset "LP-002 device implicit-factor graph parameter driver" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64), ode in (false,true)
        x=T[0.7,-0.3]; y=T[0.4,-0.1]
        model=ode ? NeuralODEFactor(DeviceDynamics(),2;integrator=EulerIntegrator(4),channels=(:x,:y)) :
            DEQFactor(DeviceCell(),(x=2,y=2);channels=(:x,:y),solver=LimitedBroydenSolver())
        p=ode ? (A=T[-0.2 0.1;-0.05 -0.3],) : (W=T[0.2 0.1;-0.05 0.3],U=T[-0.8 0.1;0.05 -0.9],b=T[0.2,-0.1])
        dp=device_tree(p)
        b=GraphBuilder();variable!(b,:x);variable!(b,:y)
        factor!(b,:data,DataFactor(:x,JLArray(x)));factor!(b,:target,DataFactor(:y,JLArray(y)))
        factor!(b,:model,model)
        connect!(b,:data,:x,:x;direction=Emitting());connect!(b,:target,:y,:y;direction=Emitting())
        connect!(b,:model,:x,:x;direction=Absorbing());connect!(b,:model,:y,:y;direction=Absorbing())
        graph=build(b);ps=(data=(;),target=(;),model=dp);st=(data=(;),target=(;),model=(;))
        _,_,_,store=infer!(graph,forward_backward_schedule(graph),ps,st)
        ref=energy_gradient(model,x,nothing,y,p,(;);ad=AutoForwardDiff())
        for ad in (AutoForwardDiff(),AutoZygote())
            grad=free_energy_gradient(store,graph,ps,st;ad)
            device_check(LenticulumCore._flatten(grad.model),LenticulumCore._flatten(ref),T)
            pn,_,_=parameter_step(store,graph,ps,st;ad)
            device_check(LenticulumCore._flatten(pn.model),LenticulumCore._flatten(p)-T(0.01)*LenticulumCore._flatten(ref),T)
        end
    end
end

struct DeviceNonlinear <: LuxCore.AbstractLuxLayer end
(::DeviceNonlinear)(inp,ps,st)=(ps.a .* tanh.(inp[1]) .+ ps.b .* inp[2] .+ ps.c,st)
@testset "LP-002 nonlinear device IFT" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64)
        x=T[0.3,-0.4];p=(a=T[0.2,0.3],b=T[-0.8,-0.9],c=T[0.1,-0.1])
        dp=device_tree(p);dx=JLArray(x)
        tol=T===Float32 ? 2e-6 : 1e-11
        f=DEQFactor(DeviceNonlinear(),(x=2,z=2);solver=LimitedBroydenSolver(;tol),
            ift=KrylovIFT(rtol=tol,atol=tol/10,memory=2,itmax=40,precond=JacobiIFT()))
        ref=DEQFactor(DeviceNonlinear(),(x=2,z=2);solver=BroydenSolver(tol=1e-12))
        z=first(solve_state(ref,x,p,(;)))
        d=1 .- p.a .* (1 .- tanh.(z).^2)
        exact=vcat(tanh.(z).*z./d,x.*z./d,z./d)
        for ad in (AutoForwardDiff(),AutoZygote())
            g=parameter_gradient(q -> sum(abs2,first(solve_state(f,dx,q,(;);ad)))/2,dp;ad)
            device_check(LenticulumCore._flatten(g),exact,T)
            out=first(solve_state(f,dx,dp,(;);ad))
            device_check(out,z,T)
            y=T[0.2,-0.1]
            rev=(y-p.a.*tanh.(y)-p.c)./p.b
            gr=parameter_gradient(q -> sum(abs2,first(solve_input(f,JLArray(y),q,(;);ad)))/2,dp;ad)
            exactr=vcat(-tanh.(y).*rev./p.b,-rev.^2 ./p.b,-rev./p.b)
            device_check(LenticulumCore._flatten(gr),exactr,T)
        end
    end
end

struct DeviceStateDynamics <: LuxCore.AbstractLuxLayer end
(::DeviceStateDynamics)(inp,ps,st)=(ps.A*inp[1] .+ st.k .* ps.c,(k=st.k+1,))
@testset "LP-002 device functional flow state" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64),itg in (EulerIntegrator(2),RK4Integrator(2)),policy in (:threaded,:fixed)
        x=T[0.3,-0.4];y=T[0.1,0.2];p=(A=T[-0.2 0.1;-0.05 -0.3],c=T[0.02,-0.03])
        f=NeuralODEFactor(DeviceStateDynamics(),2;integrator=itg,state_policy=policy)
        st=(k=1,);dp=device_tree(p)
        ref=energy_gradient(f,x,nothing,y,p,st;ad=AutoForwardDiff())
        for ad in (AutoForwardDiff(),AutoZygote())
            g=energy_gradient(f,JLArray(x),nothing,JLArray(y),dp,st;ad)
            device_check(LenticulumCore._flatten(g),LenticulumCore._flatten(ref),T)
        end
        out,sn=flow_forward(f,JLArray(x),dp,st)
        cpu,sc=flow_forward(f,x,p,st)
        device_check(out,cpu,T)
        @test sn==sc
        @test sn.k==(policy===:fixed ? 2 : itg isa EulerIntegrator ? 3 : 9)
        @test st.k==1
    end
end

@testset "LP-002 device Gaussian moment and density derivatives" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64), ad in (AutoForwardDiff(),AutoZygote())
        p=(μ=JLArray(T[0.7,-0.3]),λ=JLArray(T[1.4,0.8]))
        moment=parameter_gradient(p;ad) do q
            b=GaussianBelief(q.λ .* q.μ,Diagonal(q.λ))
            sum(abs2,belief_mean(b))/2+sum(belief_cov(b).diag)/2
        end
        device_check(moment.μ,Array(p.μ),T)
        device_check(moment.λ,-inv.(Array(p.λ)).^2/2,T)
        belief=GaussianBelief(p.λ .* p.μ,Diagonal(p.λ))
        partition=LenticulumCore.logpartition(belief)
        @test partition isa T
        @test partition ≈ LenticulumCore.logpartition(GaussianBelief(
            Array(belief.η),Diagonal(Array(p.λ)))) rtol=device_tol(T) atol=device_tol(T)
        partition_grad=parameter_gradient(p;ad) do q
            LenticulumCore.logpartition(GaussianBelief(q.λ .* q.μ,Diagonal(q.λ)))
        end
        device_check(partition_grad.μ,Array(p.λ).*Array(p.μ),T)
        device_check(partition_grad.λ,(Array(p.μ).^2-inv.(Array(p.λ)))/2,T)
        y=JLArray(T[0.2,-0.1])
        density=parameter_gradient(p;ad) do q
            belief_logdensity(GaussianBelief(q.λ .* q.μ,Diagonal(q.λ)),y)
        end
        d=Array(y)-Array(p.μ)
        device_check(density.μ,Array(p.λ).*d,T)
        device_check(density.λ,inv.(Array(p.λ))/2-d.^2/2,T)
    end
end

@testset "LP-011 pooled device Gaussian messages" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64),ad in (AutoForwardDiff(),AutoZygote())
        p=(a=(μ=T[0.7,-0.3],λ=T[1.4,0.8]),b=(μ=T[-0.1,0.4],λ=T[0.6,1.2]),loss=(;))
        b=GraphBuilder();variable!(b,:x)
        factor!(b,:a,DeviceSource{true}());factor!(b,:b,DeviceSource{true}())
        y=T[0.2,-0.1];factor!(b,:loss,DeviceQuadratic(JLArray(y)))
        connect!(b,:a,:x,:x;direction=Emitting(),coupling=PathwiseCoupling())
        connect!(b,:b,:x,:x;direction=Emitting(),coupling=PathwiseCoupling())
        connect!(b,:loss,:x,:x;direction=Absorbing(),coupling=ExactCoupling())
        graph=build(b);sched=forward_backward_schedule(graph);st=(a=(;),b=(;),loss=(;))
        grad=free_energy_gradient(MessageStore(graph),graph,device_tree(p),st;ad,schedule=sched)
        λ=p.a.λ+p.b.λ;μ=(p.a.λ.*p.a.μ+p.b.λ.*p.b.μ)./λ
        for name in (:a,:b)
            q=p[name]
            device_check(grad[name].μ,q.μ/10+(μ-y).*q.λ./λ,T)
            device_check(grad[name].λ,(μ-y).*(q.μ-μ)./λ-inv.(λ).^2/2-inv.(λ),T)
        end
    end
end

@testset "LP-011 device implicit messages use endpoint/IFT rules" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64),ode in (false,true),rev in (false,true)
        x=T[0.7,-0.3];target=T[0.1,-0.2]
        p=ode ? (A=T[-0.2 0.1;-0.05 -0.3],) :
            (W=T[0.2 0.1;-0.05 0.3],U=T[-0.8 0.1;0.05 -0.9],b=T[0.2,-0.1])
        tol=T===Float32 ? 2e-6 : 1e-11
        model=ode ? NeuralODEFactor(DeviceDynamics(),2;channels=(:x,:y),integrator=EulerIntegrator(4)) :
            DEQFactor(DeviceCell(),(x=2,y=2);channels=(:x,:y),solver=LimitedBroydenSolver(;tol),
                ift=KrylovIFT(rtol=tol,atol=tol/10,memory=2,itmax=40,precond=JacobiIFT()))
        observed,solved=rev ? (:y,:x) : (:x,:y)
        b=GraphBuilder();variable!(b,:x);variable!(b,:y)
        factor!(b,:data,DataFactor(observed,JLArray(x)));factor!(b,:model,model)
        dtarget=JLArray(target)
        factor!(b,:loss,LossFactor((solved,),y -> sum(abs2,y-dtarget)/2))
        connect!(b,:data,observed,observed;direction=Emitting(),coupling=ExactCoupling())
        connect!(b,:model,:x,:x;direction=Bidirectional(),coupling=ExactCoupling())
        connect!(b,:model,:y,:y;direction=Bidirectional(),coupling=PathwiseCoupling())
        connect!(b,:loss,solved,solved;direction=Absorbing(),coupling=ExactCoupling())
        graph=build(b);sched=forward_backward_schedule(graph)
        ps=(data=(;),model=device_tree(p),loss=(;));st=(data=(;),model=(;),loss=(;))
        closed=q -> ode ? (Matrix{eltype(q.A)}(I,2,2)+(rev ? -1 : 1)*q.A/4)^4*x :
            rev ? q.U\((I-q.W)*x-q.b) : (I-q.W)\(q.U*x+q.b)
        reference=parameter_gradient(q -> sum(abs2,closed(q)-target)/2,p;ad=AutoForwardDiff())
        for ad in (AutoForwardDiff(),AutoZygote())
            grad=free_energy_gradient(MessageStore(graph),graph,ps,st;ad,schedule=sched)
            device_check(LenticulumCore._flatten(grad.model),LenticulumCore._flatten(reference),T)
            if !ode && ad isa AutoForwardDiff
                @test_throws ArgumentError parameter_gradient(ps;ad) do q
                    _,_,_,store=infer!(graph,sched,q,st)
                    first(scalar_free_energy(store,graph,q,st))
                end
                @test_throws ArgumentError Mycelium._gradient_message(ad, model, :x,
                    nothing, (;), TrivialBelief(), ps.model, (k=1,))
            end
            if ad isa AutoZygote
                raw=parameter_gradient(ps;ad) do q
                    _,_,_,store=infer!(graph,sched,q,st)
                    first(scalar_free_energy(store,graph,q,st))
                end
                device_check(LenticulumCore._flatten(raw.model),LenticulumCore._flatten(reference),T)
            end
            pn,_,_=parameter_step(MessageStore(graph),graph,ps,st;ad,schedule=sched)
            device_check(LenticulumCore._flatten(pn.model),LenticulumCore._flatten(p)-T(0.01)*LenticulumCore._flatten(reference),T)
        end
    end
end

# The fixed-policy endpoint state evaluation must receive typed time too, even
# though its vector-field value is discarded. A Float64 time can widen a Float32
# activation before that return, including in a user's device layer.
struct DeviceTimeDynamics{T} <: LuxCore.AbstractLuxLayer end
function (::DeviceTimeDynamics{T})(inp,ps,st) where {T}
    @assert typeof(ForwardDiff.value(inp[2])) === T
    return ps.a .* inp[1] .+ ps.b .* inp[2], (k=st.k+1,)
end
@testset "LP-002 device time-dependent stages and endpoint state" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64),itg in (EulerIntegrator(2),RK4Integrator(2)),policy in (:threaded,:fixed)
        x=T[0.3,-0.4];p=(a=T[-0.2,0.1],b=T[0.1,-0.05]);dp=device_tree(p)
        f=NeuralODEFactor(DeviceTimeDynamics{T}(),2;integrator=itg,state_policy=policy,tspan=(0.1,0.9))
        # Augment z'=a*z+b*t by t'=1, 1'=0. The integrator's stability
        # polynomial on this constant 4x4 system is a dense discrete oracle.
        reference=q -> begin
            A=[Diagonal(q.a) q.b zeros(T,2); zeros(T,1,3) ones(T,1,1); zeros(T,1,4)]
            B=((T(0.9)-T(0.1))/2)*A
            Id=Matrix{eltype(A)}(I,4,4)
            P=itg isa EulerIntegrator ? Id+B : Id+B+B^2/2+B^3/6+B^4/24
            return (P^2*vcat(x,T[0.1,1]))[1:2]
        end
        out,sn=flow_forward(f,JLArray(x),dp,(k=1,))
        device_check(out,reference(p),T)
        @test sn.k==(policy===:fixed ? 2 : itg isa EulerIntegrator ? 3 : 9)
        want=parameter_gradient(q -> sum(abs2,reference(q))/2,p;ad=AutoForwardDiff())
        for ad in (AutoForwardDiff(),AutoZygote())
            g=parameter_gradient(q -> sum(abs2,first(flow_forward(f,JLArray(x),q,(k=1,))))/2,dp;ad)
            device_check(LenticulumCore._flatten(g),LenticulumCore._flatten(want),T)
        end
    end
end
@testset "LP-002 device parameter trees with empty branches" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64)
        p=(empty=(;), W=T[0.2 0.1;-0.05 0.3], U=T[-0.8 0.1;0.05 -0.9], b=T[0.2,-0.1], tail=(;))
        dp=device_tree(p); x=T[0.7,-0.3]; z=(I-p.W)\(p.U*x+p.b)
        flat=LenticulumCore._flatten_active(dp)
        device_check(flat,LenticulumCore._flatten(p),T)
        empty=LenticulumCore._flatten_active((outer=(zero=JLArray(T[]),empty=(;)),))
        device_check(empty,T[],T)
        tol=T===Float32 ? 2e-6 : 1e-11
        f=DEQFactor(DeviceCell(),(x=2,z=2);solver=LimitedBroydenSolver(;tol),
            ift=KrylovIFT(rtol=tol,atol=tol/10,memory=2,itmax=40))
        Id=Matrix{T}(I,2,2)
        exact=((I-p.W)\hcat(kron(z',Id),kron(x',Id),Id))'z
        for ad in (AutoForwardDiff(),AutoZygote())
            g=parameter_gradient(q -> sum(abs2,first(solve_state(f,JLArray(x),q,(;);ad)))/2,dp;ad)
            device_check(LenticulumCore._flatten(g),exact,T)
            @test g.empty==g.tail==(;)
        end
    end
end
@info "LP-002 batch B device error measurement" float32=DEVICE_ERRORS[Float32] float64=DEVICE_ERRORS[Float64]
