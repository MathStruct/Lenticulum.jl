using OrdinaryDiffEqVerner, OrdinaryDiffEqSDIRK

# Incoming state changes the equation; returned state changes as well. In fixed
# mode no rejected trial or sensitivity replay is permitted to advance it.
struct FixedAffineDynamics <: LuxCore.AbstractLuxLayer end
(::FixedAffineDynamics)(inp,p,s) =
    (p.a .* inp[1] .+ s.k .* p.c, (k=s.k+1,))
const BATCH_ENZYME = AutoEnzyme(;mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
batch_endpoint(x,f,p,s,reverse) = first(reverse ? flow_reverse(f,x,p,s) : flow_forward(f,x,p,s))
batch_scalar_endpoint(x,args...) = only(batch_endpoint(x,args...))

@testset "LP-002 fixed-state SciML breadth" begin
    errors = Float64[]
    T, x, y, ps, st = 0.7, [1.2], [0.5], (a=-0.35,c=0.12), (k=2,)
    algorithms = (Tsit5(),Vern7(),Kvaerno3(),KenCarp4())
    for alg in algorithms, sa in (nothing, InterpolatingAdjoint(;autojacvec=ZygoteVJP()),
                                 GaussAdjoint(;autojacvec=ZygoteVJP()))
        itg = SciMLIntegrator(alg;sensealg=sa,abstol=1e-12,reltol=1e-11)
        f = NeuralODEFactor(FixedAffineDynamics(),1;tspan=(0.,T),integrator=itg,state_policy=:fixed)
        bad_adjoint = alg isa Kvaerno3 && sa isa InterpolatingAdjoint
        if bad_adjoint
            hint = msg -> startswith(msg,"ArgumentError:") && occursin("Kvaerno3 with InterpolatingAdjoint",msg)
            for ad in (AutoZygote(),AutoMooncake(),BATCH_ENZYME)
                @test_throws hint energy_gradient(f,x,nothing,y,ps,st;ad)
                @test_throws hint energy_jacobian(f,x,nothing,y,ps,st;wrt=:ps,ad)
            end
        end
        rk = NeuralODEFactor(FixedAffineDynamics(),1;tspan=(0.,T),integrator=RK4Integrator(1000),state_policy=:fixed)
        for reverse in (false,true)
            h = reverse ? -T : T
            e = exp(ps.a*h)
            endpoint = x[1]*e + st.k*ps.c*expm1(ps.a*h)/ps.a
            da = h*x[1]*e + st.k*ps.c*(h*ps.a*e-expm1(ps.a*h))/ps.a^2
            dc = st.k*expm1(ps.a*h)/ps.a
            got, out = reverse ? flow_reverse(f,x,ps,st) : flow_forward(f,x,ps,st)
            @test got ≈ [endpoint] atol=3e-9
            @test out == (k=3,)
            @test (reverse ? flow_reverse(rk,x,ps,st) : flow_forward(rk,x,ps,st))[1] ≈ got atol=3e-9
            @test last(flow_forward(rk,x,ps,st)) == out
            # Enzyme's tested continuous endpoint mode is reverse with the default
            # ForwardDiffSensitivity; broaden the other backends over adjoints.
            backends = bad_adjoint ? (AutoForwardDiff(),) : sa === nothing && alg isa typeof(first(algorithms)) ?
                (AutoForwardDiff(),AutoZygote(),AutoMooncake(),BATCH_ENZYME) :
                (AutoForwardDiff(),AutoZygote(),AutoMooncake())
            for ad in backends
                g = parameter_gradient(p -> only(batch_endpoint(x,f,p,st,reverse)),ps;ad)
                push!(errors,norm(LC._flatten(g)-[da,dc]))
                @test LC._flatten(g) ≈ [da,dc] atol=3e-8
                if ad isa AutoEnzyme
                    @test DI.gradient(batch_scalar_endpoint,ad,x,map(DI.Constant,(f,ps,st,reverse))...) ≈ [e] atol=3e-8
                else
                    j = DI.jacobian(batch_endpoint,ad,x,map(DI.Constant,(f,ps,st,reverse))...)
                    @test j ≈ fill(e,1,1) atol=3e-8
                end
                if !reverse
                    @test LC._flatten(energy_gradient(f,x,nothing,y,ps,st;ad)) ≈ (endpoint-y[1]).*[da,dc] atol=3e-8
                    if ad isa AutoEnzyme
                        @test_throws ArgumentError energy_jacobian(f,x,nothing,y,ps,st;wrt=:ps,ad)
                    else
                        @test energy_jacobian(f,x,nothing,y,ps,st;wrt=:ps,ad) ≈ -[da dc] atol=3e-8
                    end
                end
            end
        end
    end
    # A fast decaying mode exercises the implicit solver on a stiff linear field.
    sf = NeuralODEFactor(FixedAffineDynamics(),1;tspan=(0.,1.),state_policy=:fixed,
        integrator=SciMLIntegrator(Kvaerno3();abstol=1e-12,reltol=1e-11))
    sp = (a=-50.,c=0.1)
    exact = 1.2exp(-50.)+st.k*sp.c*expm1(-50.)/sp.a
    @test first(flow_forward(sf,x,sp,st)) ≈ [exact] atol=2e-11
    for ad in (AutoForwardDiff(),AutoZygote(),AutoMooncake())
        gp = parameter_gradient(p -> only(first(flow_forward(sf,x,p,st))),sp;ad)
        @test gp.a ≈ 1.2exp(-50.)+st.k*sp.c*(-50exp(-50.)-expm1(-50.))/sp.a^2 atol=3e-9
        @test gp.c ≈ st.k*expm1(-50.)/sp.a atol=3e-9
    end
    @test_throws ArgumentError NeuralODEFactor(FixedAffineDynamics(),1;state_policy=:bad)
    @test_throws ArgumentError flow_logdet(sf,x,sp,st)
    for ad in (AutoEnzyme(),AutoEnzyme(;mode=Enzyme.Reverse),ENZYME_FLOW_BACKENDS[2])
        @test_throws ArgumentError energy_gradient(sf,x,nothing,y,sp,st;ad)
    end
    @info "LP-002 batch A SciML max closed-form derivative error" maximum(errors)
end

struct HistoryCell{M} <: LuxCore.AbstractLuxLayer
    A::M
end
(c::HistoryCell)(inp,p,s) = (inp[1] .- c.A*inp[1] .+ inp[2] .+ p.b,s)

@testset "LP-002 retained Broyden preconditioning: non-diagonal" begin
    rng, n = Xoshiro(81), 12
    Q = Matrix(qr(randn(rng,n,n)).Q)
    A = Q*Diagonal(10. .^ range(-2,0;length=n))*Q' * (I+0.15triu(ones(n,n),1)/n)
    @test !issymmetric(A)
    @test norm(A-Diagonal(diag(A))) > 0.1
    x, ps, st = randn(rng,n), (b=randn(rng,n),), (;)
    solver = LimitedBroydenSolver(memory=60,maxiters=300,tol=1e-8,init_scale=10.0)
    measurements, errors = Dict(), Float64[]
    exactz = A\(x+ps.b)
    for ad in (AutoForwardDiff(),AutoZygote(),AutoEnzyme(),AutoMooncake())
        dense = DEQFactor(HistoryCell(A),(x=n,z=n);solver)
        gd = parameter_gradient(p -> sum(abs2,first(solve_state(dense,x,p,st;ad)))/2,ps;ad).b
        @test gd ≈ A'\exactz rtol=3e-8
        counts = Dict()
        for (label,precond) in (("none",nothing),("Jacobi",JacobiIFT()),("Broyden",BroydenIFT()))
            reports = []
            ift = KrylovIFT(;precond,memory=n,itmax=200,rtol=1e-11,atol=1e-12,on_solve=r -> push!(reports,r))
            f = DEQFactor(HistoryCell(A),(x=n,z=n);solver,ift)
            g = parameter_gradient(p -> sum(abs2,first(solve_state(f,x,p,st;ad)))/2,ps;ad).b
            push!(errors,norm(g-gd)/norm(gd))
            @test g ≈ gd rtol=1e-8 atol=1e-8
            @test all(r -> r.solved,reports)
            counts[label] = maximum(r.iterations for r in reports)
            # Public reports preserve their original type; no history leaks out.
            @test solve_state(f,x,ps,st;ad)[2] isa SolveReport
        end
        measurements[string(nameof(typeof(ad)))] = counts
        @info "LP-002 history GMRES row" backend=string(nameof(typeof(ad))) none=counts["none"] Jacobi=counts["Jacobi"] Broyden=counts["Broyden"]
        @test counts["Broyden"] > 0 # Record improvements or regressions, without promising a speedup.
    end
    # Distinguishes H from H' directly, rather than relying on GMRES correcting
    # a wrongly oriented preconditioner. Also check reverse polarity's -U sign.
    ext = Base.get_extension(ImplicitLayers,:ImplicitLayersKrylovExt)
    for reverse in (false,true), transposed in (false,true)
        p, s = LuxCore.setup(Xoshiro(2),CELL)
        f = DEQFactor(CELL,(x=2,z=2);solver=LimitedBroydenSolver(memory=20,tol=1e-13),ift=KrylovIFT(precond=BroydenIFT()))
        problem = reverse ? ImplicitLayers._InputProblem(f,f.solver,f.ift) : f
        obs = [0.7,-0.3]
        u = vcat(obs,LC._flatten(p)); off = LC.offset_tree(p)
        z, r, _ = ImplicitLayers._implicit_state(AutoForwardDiff(),problem,u,p,off,s,zeros(2))
        hist = ImplicitLayers._solve_history(r)
        system = ImplicitLayers._ift_prepare(AutoForwardDiff(),problem,u,z,p,off,s,transposed,hist)
        H = hist.scale*Matrix{Float64}(I,2,2)+sum(a*b' for (a,b) in zip(hist.a,hist.b))
        v = [0.8,-0.2]; out = similar(v)
        mul!(out,system.left,v)
        @test out ≈ (transposed ? H' : H)*v atol=1e-12
        @test DI.jacobian(reverse ? reverse_observation : ad_solved_input,AutoForwardDiff(),obs,
            map(DI.Constant,(f,p,s,AutoForwardDiff()))...) ≈ (reverse ? p.U\(I-p.W) : (I-p.W)\p.U) atol=1e-9
    end
    # Exercise eviction as well as the full retained history above.
    p2,s2=LuxCore.setup(Xoshiro(2),CELL); obs=[0.7,-0.3]
    bounded=DEQFactor(CELL,(x=2,z=2);solver=LimitedBroydenSolver(memory=2,tol=1e-10),
        ift=KrylovIFT(precond=BroydenIFT(),rtol=1e-10,atol=1e-12))
    u2=vcat(obs,LC._flatten(p2)); off2=LC.offset_tree(p2)
    _,r2,_=ImplicitLayers._implicit_state(AutoForwardDiff(),bounded,u2,p2,off2,s2,zeros(2))
    @test length(ImplicitLayers._solve_history(r2).a) == 2
    for ad in (AutoForwardDiff(),AutoZygote(),AutoEnzyme(),AutoMooncake())
        @test DI.jacobian(ad_solved_input,ad,obs,map(DI.Constant,(bounded,p2,s2,ad))...) ≈ (I-p2.W)\p2.U atol=1e-8
    end
    # No secants at an exact warm start is legal: use the initial scale.
    f = DEQFactor(HistoryCell(A),(x=n,z=n);solver,ift=KrylovIFT(precond=BroydenIFT(),memory=n))
    @test first(solve_state(f,x,ps,st;ad=AutoForwardDiff(),z₀=exactz)) ≈ exactz
    for (s,k) in ((BroydenSolver(),KrylovIFT(precond=BroydenIFT())),
                  (solver,KrylovIFT(precond=BroydenIFT(),ldiv=true)))
        f = DEQFactor(HistoryCell(A),(x=n,z=n);solver=s,ift=k)
        @test_throws ArgumentError solve_state(f,x,ps,st;ad=AutoForwardDiff())
    end
    @info "LP-002 non-diagonal Broyden history" condition=cond(A) measurements max_relative_gradient_error=maximum(errors)
end

@testset "LP-002 reverse-input second order" begin
    ad = AutoForwardDiff()
    ps, st = LuxCore.setup(Xoshiro(21),CELL)
    f = DEQFactor(CELL,(x=2,z=2);solver=LimitedBroydenSolver(tol=1e-13))
    z = [0.7,-0.3]; θ, off = LC._flatten(ps), LC.offset_tree(ps)
    u = vcat(z,θ)
    loss(x) = sum(abs2,x)/2
    exact(u) = begin
        p = LC._unflatten(u[3:end],off,ps)
        loss(p.U\((I-p.W)*u[1:2]-p.b))
    end
    H = ForwardDiff.hessian(exact,u)
    errors, fd_errors = Float64[], Float64[]
    for wrt in (:joint,:z,:ps)
        inds = wrt === :joint ? (1:length(u)) : wrt === :z ? (1:2) : (3:length(u))
        v = collect(range(-0.2,0.3;length=length(inds)))
        hv = solve_input_hvp(loss,f,z,ps,st,v;wrt,ad)
        push!(errors,norm(hv-H[inds,inds]*v))
        @test hv ≈ H[inds,inds]*v atol=1e-9
        function grad(w)
            Zygote.gradient(w -> begin
                zz, pp = wrt === :joint ? (w[1:2],LC._unflatten(w[3:end],off,ps)) :
                    wrt === :ps ? (z,LC._unflatten(w,off,ps)) : (w,ps)
                loss(first(solve_input(f,zz,pp,st;ad=AutoZygote())))
            end,w)[1]
        end
        fd = (grad(u[inds]+1e-5v)-grad(u[inds]-1e-5v))/(2e-5)
        push!(fd_errors,norm(hv-fd))
        @test hv ≈ fd atol=2e-7
    end
    nf = DEQFactor(NumericsCell(),(x=1,z=1);solver=BroydenSolver(tol=1e-13))
    np, ns, zz = (a=0.2,b=0.1),(bias=0.03,),[0.8]
    # x=sqrt(z-a*z²-b-bias); x_zz=-a/x-(1-2az)²/(4x³).
    xx = sqrt(zz[1]-np.a*zz[1]^2-np.b-ns.bias)
    analytic = -np.a/xx-(1-2np.a*zz[1])^2/(4xx^3)
    @test only(solve_input_hvp(only,nf,zz,np,ns,[1.];wrt=:z,ad,x₀=[0.5])) ≈ analytic atol=1e-9
    nu = vcat(zz,LC._flatten(np)); nv = [0.2,-0.1,0.3]
    closed(u) = sqrt(u[1]-u[2]*u[1]^2-u[3]-ns.bias)
    hv = solve_input_hvp(only,nf,zz,np,ns,nv;wrt=:joint,ad,x₀=[0.5])
    push!(errors,norm(hv-ForwardDiff.hessian(closed,nu)*nv))
    @test hv ≈ ForwardDiff.hessian(closed,nu)*nv atol=1e-9
    @test_throws ArgumentError solve_input_hvp(loss,f,z,ps,st,ones(2);wrt=:x,ad)
    @test_throws ArgumentError solve_input_hvp(loss,f,z,ps,st,ones(2);wrt=:z,ad=AutoZygote())
    @test_throws ArgumentError solve_input_hvp(loss,f,z,ps,st,ones(1);wrt=:z,ad)
    mf = DEQFactor(CELL,(x=2,z=2);ift=KrylovIFT())
    @test_throws ArgumentError solve_input_hvp(loss,mf,z,ps,st,ones(2);wrt=:z,ad)
    rect = DEQFactor(CELL,(x=1,z=2))
    @test_throws ArgumentError solve_input_hvp(loss,rect,z,ps,st,ones(2);wrt=:z,ad)
    @info "LP-002 reverse-input HVP" max_closed_absolute=maximum(errors) max_first_gradient_fd_absolute=maximum(fd_errors)
end

LuxCore.initialparameters(::AbstractRNG,::FixedAffineDynamics) = (a=-0.35,c=0.12)
LuxCore.initialstates(::AbstractRNG,::FixedAffineDynamics) = (k=2,)
@testset "LP-002 fixed-state SciML graph gradients" begin
    b = GraphBuilder()
    variable!(b,:z0); variable!(b,:z1)
    factor!(b,:input,DataFactor(:z0,[1.2])); factor!(b,:target,DataFactor(:z1,[0.5]))
    f = NeuralODEFactor(FixedAffineDynamics(),1;tspan=(0.,0.7),state_policy=:fixed,
        integrator=SciMLIntegrator(Tsit5();abstol=1e-12,reltol=1e-11))
    factor!(b,:model,f)
    connect!(b,:input,:z0,:z0;direction=Emitting())
    connect!(b,:target,:z1,:z1;direction=Emitting())
    connect!(b,:model,:z0,:z0;direction=Absorbing())
    connect!(b,:model,:z1,:z1;direction=Absorbing())
    g=build(b); ps,st=LuxCore.setup(Xoshiro(3),g)
    store=MessageStore(g)
    _,st=propagate!(store,g,tree_schedule(g),ps,st)
    k=st.model.k; T=0.7; a=ps.model.a; c=ps.model.c; e=exp(a*T)
    endpoint=1.2*e+k*c*expm1(a*T)/a
    expected=(endpoint-0.5).*[T*1.2*e+k*c*(T*a*e-expm1(a*T))/a^2,k*expm1(a*T)/a]
    for ad in (AutoForwardDiff(),AutoZygote(),AutoMooncake(),BATCH_ENZYME)
        grad=free_energy_gradient(store,g,ps,st;ad)
        @test LC._flatten(grad.model) ≈ expected atol=3e-8
        next,_,_=parameter_step(store,g,ps,st;ad,rule=GradientDescent(0.01))
        @test LC._flatten(next.model) ≈ LC._flatten(ps.model)-0.01expected atol=3e-9
        @test st.model.k == k
    end
end

struct BatchClosureCell <: LuxCore.AbstractLuxLayer end
(::BatchClosureCell)(inp,p,s) = (p.a .* inp[1] .+ inp[2],s)
function batch_closure_probe(ad)
    f=DEQFactor(BatchClosureCell(),(x=1,z=1)); x=[1.2]; ps=(a=[0.2],); st=(;)
    θ=LC._flatten(ps); off=LC.offset_tree(ps)
    loss=t -> sum(first(solve_state(f,x,LC._unflatten(t,off,ps),st;ad)))
    DI.gradient(loss,ad,θ)
end
@testset "LP-002 outer Enzyme closure boundary" begin
    @test_throws Enzyme.Compiler.EnzymeMutabilityException batch_closure_probe(AutoEnzyme())
    @test batch_closure_probe(AutoEnzyme(;function_annotation=Enzyme.Const)) ≈ [1.2/(1-0.2)^2] atol=1e-10
end

struct NestedFixedDynamics <: LuxCore.AbstractLuxLayer end
(::NestedFixedDynamics)(inp,p,s) = (p.a .* inp[1] .+ s.layer.shift,
    (layer=(shift=s.layer.shift .+ inp[2].*inp[1],),))
@testset "LP-002 nested floating Lux state is constant per solve" begin
    ps,st,x,T=(a=-0.35,),(layer=(shift=[0.24],),),[1.2],0.7
    f=NeuralODEFactor(NestedFixedDynamics(),1;tspan=(0.,T),state_policy=:fixed,
        integrator=SciMLIntegrator(Tsit5();abstol=1e-12,reltol=1e-11))
    e=exp(ps.a*T)
    endpoint=x[1]*e+st.layer.shift[1]*expm1(ps.a*T)/ps.a
    da=T*x[1]*e+st.layer.shift[1]*(T*ps.a*e-expm1(ps.a*T))/ps.a^2
    z,out=flow_forward(f,x,ps,st)
    @test z ≈ [endpoint] atol=3e-10
    @test out.layer.shift ≈ st.layer.shift+T*z atol=1e-12
    for ad in (AutoForwardDiff(),AutoZygote(),AutoMooncake())
        @test parameter_gradient(p -> only(first(flow_forward(f,x,p,st))),ps;ad).a ≈ da atol=3e-8
    end
    rk=NeuralODEFactor(NestedFixedDynamics(),1;tspan=(0.,T),state_policy=:fixed,
        integrator=RK4Integrator(1000))
    @test parameter_gradient(p -> only(first(flow_forward(rk,x,p,st))),ps;ad=AutoForwardDiff()).a ≈ da atol=3e-10
    @test st.layer.shift == [0.24]
end

@testset "LP-002 stateless Enzyme SciML scalar gradients" begin
    f=NeuralODEFactor(LinearDynamics(fill(-0.35,1,1)),1;tspan=(0.,0.7),
        integrator=SciMLIntegrator(Tsit5();abstol=1e-12,reltol=1e-11))
    p,s=(A=fill(-0.35,1,1),),(;)
    x,y=[1.2],[0.5]; endpoint=1.2exp(-0.35*0.7); d=0.7endpoint
    @test only(energy_gradient(f,x,nothing,y,p,s;ad=BATCH_ENZYME).A) ≈ (endpoint-y[1])*d atol=3e-8
    @test only(free_energy_gradient(f,DiracBelief(x),DiracBelief(y),p,s;ad=BATCH_ENZYME).A) ≈ (endpoint-y[1])*d atol=3e-8
end

@testset "LP-002 reverse-time stateless continuous adjoints" begin
    for sa in (InterpolatingAdjoint(;autojacvec=ZygoteVJP()),GaussAdjoint(;autojacvec=ZygoteVJP()))
        f=NeuralODEFactor(LinearDynamics(fill(-0.35,1,1)),1;tspan=(0.,0.7),
            integrator=SciMLIntegrator(Tsit5();sensealg=sa,abstol=1e-12,reltol=1e-11))
        p,s=(A=fill(-0.35,1,1),),(;)
        for ad in (AutoZygote(),AutoMooncake())
            got=parameter_gradient(p -> only(first(flow_reverse(f,[1.2],p,s))),p;ad)
            @test only(got.A) ≈ -0.7*1.2exp(0.35*0.7) atol=3e-8
        end
    end
end
