using Krylov

@testset "LP-002 limited-memory Broyden" begin
    cases = (
        ("contractive", u -> u - (W₁*u + [1.0,2.0]), zeros(2)),
        ("noncontractive", u -> -0.6u - [1.0,-2.0], zeros(2)),
        ("nonlinear", u -> u - 0.2u.^2 .- 0.5, zeros(2)),
    )
    for (name, F, u0) in cases
        dense_trace, limited_trace = Vector{Float64}[], Vector{Float64}[]
        ud, rd = solve_root(u -> (push!(dense_trace, copy(u)); F(u)), u0,
            BroydenSolver(maxiters=100, tol=1e-12))
        ul, rl = solve_root(u -> (push!(limited_trace, copy(u)); F(u)), u0,
            LimitedBroydenSolver(memory=100, maxiters=100, tol=1e-12))
        @test rd.converged && rl.converged
        @test length(dense_trace) == length(limited_trace)
        @test all(isapprox(a,b; atol=2e-12,rtol=2e-12) for (a,b) in zip(dense_trace,limited_trace))
        @test ul ≈ ud atol=2e-12
        @test rl.nfe == rl.iterations+1
        small, rs = solve_root(F,u0,LimitedBroydenSolver(memory=2,maxiters=100,tol=1e-12))
        @test rs.converged
        @test norm(F(small)) <= 1e-12
        @info "LP-002 Broyden iterations" case=name dense=rd.iterations full_memory=rl.iterations memory2=rs.iterations
        for scale in (1.0, 0.5, -1.0)
            a, _ = solve_root(F,u0,LimitedBroydenSolver(maxiters=1,tol=0,init_scale=scale))
            b, _ = solve_root(F,u0,PicardSolver(maxiters=1,tol=0,damping=scale))
            @test a == b
        end
    end
    # Total warmed allocations scale linearly at fixed memory/iterations. A
    # dense 20,000-square Float64 inverse alone would require 3.2 GB.
    allocations = Int[]
    for n in (10_000,20_000)
        d = collect(range(0.5,1.5;length=n))
        F = u -> d.*u .- 1
        u0, solver = zeros(n), LimitedBroydenSolver(memory=2,maxiters=10,tol=0)
        solve_root(F,u0,solver)
        push!(allocations,@allocated solve_root(F,u0,solver))
    end
    @test allocations[2] < 3allocations[1]+1_000_000
    @test allocations[2] < 64*1024^2
    @info "LP-002 limited Broyden allocations (n=10000,20000; memory=2; ten steps)" allocations
    for kwargs in ((memory=0,), (maxiters=-1,), (tol=-1,), (tol=NaN,), (init_scale=0,), (init_scale=Inf,))
        @test_throws ArgumentError LimitedBroydenSolver(;kwargs...)
    end
    _, r = solve_root(identity,zeros(2),LimitedBroydenSolver())
    @test r.converged && r.iterations == 0 && r.nfe == 1
    _, r = solve_root(u -> ones(2),zeros(2),LimitedBroydenSolver(memory=1,maxiters=3))
    @test !r.converged && r.iterations == 3 && r.nfe == 4 # degenerate secant
    _, r = solve_root(u -> fill(Inf,2),zeros(2),LimitedBroydenSolver())
    @test !r.converged && r.iterations == 1 && r.nfe == 2
    _, r = solve_root(identity,ones(2),LimitedBroydenSolver(maxiters=0))
    @test !r.converged && r.nfe == 1
end

reverse_parameters(θ, f, z, off, ps, st, ad) = first(solve_input(f,z,LC._unflatten(θ,off,ps),st;ad))
reverse_loss(θ,args...) = sum(abs2,reverse_parameters(θ,args...))/2
reverse_observation(z,f,ps,st,ad) = first(solve_input(f,z,ps,st;ad,x₀=z))
reverse_warm(x0,f,z,ps,st,ad) = first(solve_input(f,z,ps,st;ad,x₀=x0))

@testset "LP-002 reverse-input IFT in both linear solvers" begin
    ps, st = LuxCore.setup(Xoshiro(20),CELL)
    z = [0.7,-0.3]
    x = xstar(z)
    θ, off = LC._flatten(ps), LC.offset_tree(ps)
    Id = Matrix{Float64}(I,2,2)
    exact = -(ps.U \ hcat(kron(z',Id),kron(x',Id),Id))
    for ift in (DenseIFT(), KrylovIFT(rtol=1e-11,atol=1e-13,precond=JacobiIFT())),
        ad in (AutoForwardDiff(),AutoZygote(),AutoEnzyme(),AutoMooncake(),
               AutoEnzyme(;mode=Enzyme.Forward),AutoEnzyme(;mode=Enzyme.Reverse))
        f = DEQFactor(CELL,(x=2,z=2);solver=LimitedBroydenSolver(memory=10,tol=1e-12),ift)
        ctx = map(DI.Constant,(f,z,off,ps,st,ad))
        J = DI.jacobian(reverse_parameters,ad,θ,ctx...)
        @test J ≈ exact atol=1e-9
        @test DI.gradient(reverse_loss,ad,θ,ctx...) ≈ exact'x atol=1e-9
        @test DI.jacobian(reverse_observation,ad,z,map(DI.Constant,(f,ps,st,ad))...) ≈ ps.U\(I-ps.W) atol=1e-9
        @test iszero(DI.jacobian(reverse_warm,ad,x,map(DI.Constant,(f,z,ps,st,ad))...))
        @test J ≈ LC._jacobian(nothing, t -> reverse_parameters(t,f,z,off,ps,st,ad), θ) atol=2e-7
    end
end

reverse_nonlinear_parameters(θ,f,z,off,ps,st,ad) = first(solve_input(f,z,LC._unflatten(θ,off,ps),st;ad,x₀=[0.5]))

struct NumericsCell <: LuxCore.AbstractLuxLayer end
(::NumericsCell)(inp,ps,st) = (ps.a .* inp[1].^2 .+ ps.b .+ inp[2].^2 .+ st.bias,(bias=st.bias+1,))
@testset "LP-002 nonlinear reverse-input and fixed state" begin
    f = DEQFactor(NumericsCell(),(x=1,z=1);solver=BroydenSolver(tol=1e-13))
    ps, st, z = (a=0.2,b=0.1), (bias=0.03,), [0.8]
    x = sqrt(z[1]-ps.a*z[1]^2-ps.b-st.bias)
    θ, off = LC._flatten(ps), LC.offset_tree(ps)
    for ad in (AutoForwardDiff(),AutoZygote(),AutoEnzyme(),AutoMooncake())
        # Warm start chooses the positive branch; its derivative is zero.
        @test DI.jacobian(reverse_nonlinear_parameters, ad, θ, map(DI.Constant,(f,z,off,ps,st,ad))...) ≈ [-z[1]^2 -1]/(2x) atol=1e-9
        got, rep, out = solve_input(f,z,ps,st;ad,x₀=[0.5])
        @test rep.converged && only(got) ≈ x
        @test out.bias == st.bias+1
        bad = DEQFactor(CELL,(x=2,z=2);solver=BroydenSolver(maxiters=0))
        @test_throws ArgumentError solve_input(bad,[1.,2.],LuxCore.setup(Xoshiro(1),CELL)...;ad)
        singular = DEQFactor(LinearCell(zeros(1,1),zeros(1,1),zeros(1)),(x=1,z=1))
        @test_throws SingularException DI.jacobian(reverse_observation,ad,[0.],
            map(DI.Constant,(singular,(W=zeros(1,1),U=zeros(1,1),b=zeros(1)),(;),ad))...)
    end
    @test_throws ArgumentError solve_input(f,z,ps,st;ad=AutoFiniteDiff())
end

struct DiagonalNumericsCell{V} <: LuxCore.AbstractLuxLayer
    diagonal::V
end
(c::DiagonalNumericsCell)(inp,ps,st) = ((1 .- c.diagonal).*inp[1] .+ inp[2] .+ ps.b,st)
@testset "LP-002 Krylov preconditioning" begin
    n = 20
    d = 10.0 .^ range(-3,0;length=n)
    cell, x, ps, st = DiagonalNumericsCell(d), d.*collect(range(0.1,0.5;length=n)), (b=zeros(n),), (;)
    exactz = x./d
    counts = Dict{String,Int}()
    errors = Float64[]
    measurements = Dict{String,Dict{String,Int}}()
    for ad in (AutoForwardDiff(),AutoZygote(),AutoEnzyme(),AutoMooncake())
        dense = DEQFactor(cell,(x=n,z=n);solver=BroydenSolver(maxiters=500,tol=1e-13))
        gd = parameter_gradient(p -> sum(abs2,first(solve_state(dense,x,p,st;ad,z₀=exactz)))/2,ps;ad).b
        for (label,precond,ldiv) in (
            ("none",nothing,false), ("Jacobi",JacobiIFT(),false),
            ("left",Diagonal(1 ./d),false),
            ("right",ctx -> (;right=Diagonal(1 ./d)),false),
            ("left solve",ctx -> (;left=Diagonal(d)),true))
            reports = []
            ift = KrylovIFT(;rtol=1e-12,atol=1e-13,itmax=200,memory=n,precond,ldiv,
                on_solve=r -> push!(reports,r))
            f = DEQFactor(cell,(x=n,z=n);solver=dense.solver,ift)
            g = parameter_gradient(p -> sum(abs2,first(solve_state(f,x,p,st;ad,z₀=exactz)))/2,ps;ad).b
            push!(errors,norm(g-gd)/norm(gd))
            @test g ≈ gd rtol=1e-9 atol=1e-9
            @test g ≈ exactz./d rtol=1e-9 atol=1e-9
            @test all(r -> r.residual <= 1e-13+1e-12*norm(exactz),reports)
            counts[label] = maximum(r.iterations for r in reports)
            if label != "none" && !(ad isa AutoForwardDiff)
                @test counts[label] < counts["none"]
            end
        end
        measurements[string(nameof(typeof(ad)))] = copy(counts)
    end
    @info "LP-002 Krylov iterations (condition number 1000)" measurements max_relative_gradient_error=maximum(errors)
    # A nonsymmetric operator makes the orientation requirement observable.
    p2,s2 = LuxCore.setup(Xoshiro(22),CELL)
    observed = [0.7,-0.3]
    for reverse in (false,true), ad in (AutoForwardDiff(),AutoZygote())
        orientations = Bool[]
        factory = ctx -> begin
            @assert ctx.factor.cell === CELL
            @assert ctx.solved === (reverse ? :x : :z)
            push!(orientations,ctx.transposed)
            J = reverse ? -p2.U : Matrix{Float64}(I,2,2)-p2.W
            (;right=inv(ctx.transposed ? J' : J))
        end
        f = DEQFactor(CELL,(x=2,z=2);solver=BroydenSolver(tol=1e-12),
            ift=KrylovIFT(precond=factory,rtol=1e-11,atol=1e-13))
        if reverse
            got = DI.jacobian(reverse_observation,ad,observed,map(DI.Constant,(f,p2,s2,ad))...)
            @test got ≈ p2.U\(I-p2.W) atol=1e-9
        else
            got = DI.jacobian(ad_solved_input,ad,observed,map(DI.Constant,(f,p2,s2,ad))...)
            @test got ≈ (I-p2.W)\p2.U atol=1e-9
        end
        @test all(==(ad isa AutoZygote),orientations)
        @test !isempty(orientations)
    end
    @test_throws ArgumentError JacobiIFT(min_diagonal=-1)
    for (d,ift) in ((zeros(2),KrylovIFT(precond=JacobiIFT())),
                    (ones(2),KrylovIFT(precond=JacobiIFT(),ldiv=true)),
                    (ones(2),KrylovIFT(precond=ctx -> (;bad=I))))
        f = DEQFactor(DiagonalNumericsCell(d),(x=2,z=2);ift)
        @test_throws ArgumentError DI.jacobian(ad_solved_input,AutoForwardDiff(),zeros(2),
            map(DI.Constant,(f,(b=zeros(2),),(;),AutoForwardDiff()))...)
    end
end

@testset "LP-002 second-order dense IFT adjoint" begin
    ad = AutoForwardDiff()
    f = DEQFactor(CELL,(x=2,z=2);solver=LimitedBroydenSolver(tol=1e-13))
    ps, st, x = LuxCore.setup(Xoshiro(21),CELL)..., [0.7,-0.3]
    θ, off = LC._flatten(ps), LC.offset_tree(ps)
    u = vcat(x,θ)
    loss(z) = sum(abs2,z)/2
    exactloss(u) = begin
        p = LC._unflatten(u[3:end],off,ps)
        loss((I-p.W)\(p.U*u[1:2]+p.b))
    end
    H = ForwardDiff.hessian(exactloss,u)
    closed_errors, fd_errors = Float64[], Float64[]
    for wrt in (:joint,:x,:ps)
        inds = wrt === :joint ? (1:length(u)) : wrt === :x ? (1:2) : (3:length(u))
        v = collect(range(-0.2,0.3;length=length(inds)))
        hv = deq_hvp(loss,f,x,ps,st,v;wrt,ad)
        push!(closed_errors,norm(hv-H[inds,inds]*v))
        @test hv ≈ H[inds,inds]*v atol=1e-9
        function first_gradient(w)
            xx, pp = wrt === :x ? (w,ps) : wrt === :ps ? (x,LC._unflatten(w,off,ps)) : (w[1:2],LC._unflatten(w[3:end],off,ps))
            if wrt === :x
                return Zygote.gradient(t -> loss(first(solve_state(f,t,pp,st;ad=AutoZygote()))),xx)[1]
            end
            jointloss(t) = begin
                xxx, ppp = wrt === :ps ? (x,LC._unflatten(t,off,ps)) : (t[1:2],LC._unflatten(t[3:end],off,ps))
                loss(first(solve_state(f,xxx,ppp,st;ad=AutoZygote())))
            end
            return Zygote.gradient(jointloss,w)[1]
        end
        h = 1e-5
        fd = (first_gradient(u[inds]+h*v)-first_gradient(u[inds]-h*v))/(2h)
        push!(fd_errors,norm(hv-fd))
        @test hv ≈ fd rtol=2e-7 atol=2e-7
    end
    # z = a*z²+b+x: z_x=1/q, z_xx=2a/q³, q=1-2az.
    nf = DEQFactor(QuadraticADCell(),(x=1,z=1);solver=BroydenSolver(tol=1e-13))
    np, xx = (a=0.2,b=0.4), [0.1]
    z = (1-sqrt(1-4np.a*(np.b+xx[1])))/(2np.a)
    q = 1-2np.a*z
    @test only(deq_hvp(z -> only(z),nf,xx,np,st,[1.];wrt=:x,ad)) ≈ 2np.a/q^3 atol=1e-10
    @test only(deq_hvp(loss,nf,xx,np,st,[1.];wrt=:x,ad)) ≈ 1/q^2+z*2np.a/q^3 atol=1e-10
    nθ, noff = LC._flatten(np), LC.offset_tree(np)
    nu = vcat(xx,nθ)
    nv = [0.2,-0.1,0.3]
    nclosed(u) = begin
        p = LC._unflatten(u[2:end],noff,np)
        zz = (1-sqrt(1-4p.a*(p.b+u[1])))/(2p.a)
        zz^2/2
    end
    nhv = deq_hvp(loss,nf,xx,np,st,nv;wrt=:joint,ad)
    push!(closed_errors,norm(nhv-ForwardDiff.hessian(nclosed,nu)*nv))
    @test nhv ≈ ForwardDiff.hessian(nclosed,nu)*nv atol=1e-9
    ng(u) = Zygote.gradient(t -> loss(first(solve_state(nf,t[1:1],LC._unflatten(t[2:end],noff,np),st;ad=AutoZygote()))),u)[1]
    nfd = (ng(nu+1e-5nv)-ng(nu-1e-5nv))/(2e-5)
    push!(fd_errors,norm(nhv-nfd))
    @test nhv ≈ nfd atol=2e-7
    @info "LP-002 HVP errors" max_closed_absolute=maximum(closed_errors) max_first_gradient_fd_absolute=maximum(fd_errors)
    nested_hint = msg -> startswith(msg,"ArgumentError:") && occursin("deq_hvp",msg)
    @test_throws nested_hint Zygote.hessian_reverse(t -> loss(first(solve_state(f,t,ps,st;ad=AutoZygote()))),x)
    mf = DEQFactor(CELL,(x=2,z=2);ift=KrylovIFT())
    @test_throws ArgumentError deq_hvp(loss,mf,x,ps,st,ones(length(θ));ad)
    bad = DEQFactor(CELL,(x=2,z=2);solver=BroydenSolver(maxiters=0))
    @test_throws ArgumentError deq_hvp(loss,bad,x,ps,st,ones(length(θ));ad)
    sf = DEQFactor(LinearCell(ones(1,1),zeros(1,1),zeros(1)),(x=1,z=1))
    @test_throws SingularException deq_hvp(loss,sf,[0.],(W=ones(1,1),U=zeros(1,1),b=zeros(1)),st,ones(3);ad)
    for backend in (nothing,AutoZygote(),AutoEnzyme(),AutoMooncake(),
                    DI.SecondOrder(AutoForwardDiff(),AutoZygote()),
                    DI.SecondOrder(AutoEnzyme(),AutoMooncake()))
        @test_throws ArgumentError deq_hvp(loss,f,x,ps,st,ones(length(θ));ad=backend)
    end
    @test_throws ArgumentError deq_hvp(loss,f,x,ps,st,[1.];ad)
    @test_throws ArgumentError deq_hvp(loss,f,x,ps,st,ones(length(θ));ad,wrt=:bad)
    for backend in (AutoForwardDiff(),AutoZygote(),AutoEnzyme(),AutoMooncake())
        @test_throws nested_hint ForwardDiff.hessian(t -> loss(first(solve_state(f,t,ps,st;ad=backend))),x)
    end
end
