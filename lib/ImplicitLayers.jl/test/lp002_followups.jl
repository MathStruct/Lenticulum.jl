using SciMLSensitivity, OrdinaryDiffEqTsit5, Krylov

@testset "SciML stateful primals after loading SciMLSensitivity" begin
    @test Base.get_extension(ImplicitLayers, :ImplicitLayersSciMLSensitivityExt) !== nothing
    include("sciml_state.jl")
end

struct NonlinearSensitivityDynamics <: LuxCore.AbstractLuxLayer end
(::NonlinearSensitivityDynamics)(inp, ps, st) = (tanh.(ps.W * inp[1] .+ ps.b), st)
struct GrowingSensitivityState <: LuxCore.AbstractLuxLayer end
(::GrowingSensitivityState)(inp, ps, st) = (ps.A*inp[1], (counter=1,))

@testset "LP-002 SciML endpoint sensitivities" begin
    a, z0, y, T = -0.35, [1.2], [0.5], 0.7
    errors = Float64[]
    # ForwardDiffSensitivity is the small-problem default; also exercise a
    # continuous adjoint, with its own interpolated forward trajectory.
    for sensealg in (nothing, InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
                     GaussAdjoint(; autojacvec=ZygoteVJP()))
        itg = SciMLIntegrator(Tsit5(); sensealg, abstol=1e-12, reltol=1e-10)
        f = NeuralODEFactor(LinearDynamics(fill(a,1,1)), 1; tspan=(0.0,T), integrator=itg)
        ps, st = LuxCore.setup(Xoshiro(11), f)
        endpoint = only(z0)*exp(a*T)
        derivative = T*endpoint
        @test only(first(flow_forward(f, z0, ps, st))) ≈ endpoint atol=2e-10
        for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake())
            J = only(energy_jacobian(f, z0, nothing, y, ps, st; wrt=:ps, ad))
            push!(errors, abs(J + derivative))
            @test J ≈ -derivative atol=2e-8
            g = energy_gradient(f, z0, nothing, y, ps, st; ad)
            @test only(g.A) ≈ (endpoint-only(y))*derivative atol=2e-8
            gf = free_energy_gradient(f, DiracBelief(z0), DiracBelief(y), ps, st; ad)
            @test gf.A ≈ g.A atol=2e-8
        end
        f = NeuralODEFactor(NonlinearSensitivityDynamics(), 2; integrator=itg)
        ps, st = (W=[-0.2 0.3; 0.1 -0.4], b=[0.05,-0.1]), (;)
        x, yn = [0.7,-0.4], [0.1,0.3]
        reference = energy_jacobian(f, x, nothing, yn, ps, st; wrt=:ps)
        for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake())
            @test energy_jacobian(f, x, nothing, yn, ps, st; wrt=:ps, ad) ≈ reference atol=3e-7
        end
        state = (counter=0,)
        @test first(flow_forward(f, x, ps, state)) ≈ first(flow_forward(f, x, ps, st))
        @test last(flow_forward(f, x, ps, state)) == state
        for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake(), AutoEnzyme())
            hint = msg -> startswith(msg, "ArgumentError:") && occursin("recursively empty Lux state", msg)
            @test_throws hint energy_jacobian(f, x, nothing, yn, ps, state; wrt=:ps, ad)
            @test_throws hint energy_gradient(f, x, nothing, yn, ps, state; ad)
            @test_throws hint free_energy_gradient(f, DiracBelief(x), DiracBelief(yn), ps, state; ad)
        end
        @test_throws ArgumentError energy_gradient(f, x, nothing, yn, ps, st; ad=AutoEnzyme())
    end
    # Empty nested Lux state (as in a stateless Chain) is accepted; a layer that
    # introduces evaluation state is rejected even if its incoming state was empty.
    f = NeuralODEFactor(LinearDynamics(fill(a,1,1)), 1;
        integrator=SciMLIntegrator(Tsit5(); abstol=1e-12, reltol=1e-10))
    ps = (A=fill(a,1,1),)
    nested = (layer1=(;), layer2=())
    @test last(flow_forward(f, z0, ps, nested)) === nested
    growing = NeuralODEFactor(GrowingSensitivityState(), 1; integrator=f.integrator)
    @test_throws ArgumentError flow_forward(growing, z0, ps, (;))
    @info "SciML linear endpoint derivative max absolute error" maximum(errors)
end

@testset "SciML stateful AD rejects before RHS evaluation" begin
    counter = Ref(0)
    f = NeuralODEFactor(CountingDynamics(), 1;
        integrator=SciMLIntegrator(Tsit5()), input=(u,t) -> (counter[] += 1; (u,t)))
    ps, st = LuxCore.setup(Xoshiro(15), f)
    for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake(), AutoEnzyme()),
        state in (st, (layer=(calls=0,),))
        hint = msg -> startswith(msg, "ArgumentError:") && occursin("recursively empty Lux state", msg)
        @test_throws hint energy_jacobian(f, [1.0], nothing, [0.5], ps, state; wrt=:ps, ad)
        @test_throws hint energy_gradient(f, [1.0], nothing, [0.5], ps, state; ad)
        @test_throws hint free_energy_gradient(f, DiracBelief([1.0]), DiracBelief([0.5]), ps, state; ad)
    end
    @test counter[] == 0
    # Reference differences remain available for legacy evaluation-state semantics.
    @test only(energy_jacobian(f, [1.0], nothing, [0.5], ps, st; wrt=:ps)) ≈ -exp(-0.5) atol=1e-6
    b = GraphBuilder()
    variable!(b, :z0); variable!(b, :z1)
    factor!(b, :input, DataFactor(:z0, [1.0]))
    factor!(b, :target, DataFactor(:z1, [0.5]))
    factor!(b, :model, f)
    connect!(b, :input, :z0, :z0; direction=Emitting())
    connect!(b, :target, :z1, :z1; direction=Emitting())
    connect!(b, :model, :z0, :z0; direction=Absorbing())
    connect!(b, :model, :z1, :z1; direction=Absorbing())
    g = build(b)
    gps, gst = LuxCore.setup(Xoshiro(16), g)
    store = MessageStore(g)
    _, gst = propagate!(store, g, tree_schedule(g), gps, gst)
    @test isfinite(first(scalar_free_energy(store, g, gps, gst)))
    calls = counter[]
    for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake(), AutoEnzyme())
        hint = msg -> startswith(msg, "ArgumentError:") && occursin("recursively empty Lux state", msg)
        @test_throws hint free_energy_gradient(store, g, gps, gst; ad)
        @test_throws hint parameter_step(store, g, gps, gst; ad)
    end
    @test counter[] == calls
end

@testset "LP-002 SciML graph learns continuous-time coefficient" begin
    truth = 0.4
    for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake())
        itg = SciMLIntegrator(Tsit5(); abstol=1e-12, reltol=1e-10)
        g = ode_training_graph(-0.2, exp(truth); integrator=itg)
        ps, st = LuxCore.setup(Xoshiro(12), g)
        store = MessageStore(g)
        _, st = propagate!(store, g, tree_schedule(g), ps, st)
        before = first(scalar_free_energy(store, g, ps, st))
        grad = free_energy_gradient(store, g, ps, st; ad)
        @test only(grad.model.A) ≈ (exp(-0.2)-exp(truth))*exp(-0.2) atol=2e-8
        os = nothing
        for _ in 1:100
            ps, st, os = parameter_step(store, g, ps, st; ad,
                rule=GradientDescent(0.2), optimiser_state=os)
        end
        @test only(ps.model.A) ≈ truth atol=1e-8
        @test first(scalar_free_energy(store, g, ps, st)) < before*1e-12
    end
end

@testset "LP-002 Krylov IFT matches dense and exact" begin
    ps, st = LuxCore.setup(Xoshiro(13), CELL)
    x, z = [0.7,-0.3], zstar([0.7,-0.3])
    θ, off = LC._flatten(ps), LC.offset_tree(ps)
    Id = Matrix{Float64}(I,2,2)
    exact = (I-ps.W) \ hcat(kron(z', Id), kron(x', Id), Id)
    for ad in (AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake(),
               AutoEnzyme(; mode=Enzyme.Forward), AutoEnzyme(; mode=Enzyme.Reverse))
        f = DEQFactor(CELL, (x=2,z=2); solver=BroydenSolver(tol=1e-12),
            ift=KrylovIFT(rtol=1e-11, atol=1e-13))
        ctx = map(DI.Constant, (f,x,off,ps,st,ad))
        J = DI.jacobian(ad_solved_parameters, ad, θ, ctx...)
        dense = DEQFactor(CELL, (x=2,z=2); solver=BroydenSolver(tol=1e-12))
        @test J ≈ DI.jacobian(ad_solved_parameters, ad, θ,
            map(DI.Constant,(dense,x,off,ps,st,ad))...) atol=1e-9
        @test J ≈ exact atol=1e-9
        @test DI.gradient(ad_solved_loss, ad, θ, ctx...) ≈ exact'z atol=1e-9
        @test DI.jacobian(ad_solved_input, ad, x, map(DI.Constant,(f,ps,st,ad))...) ≈ (I-ps.W)\ps.U atol=1e-9
        bad = DEQFactor(CELL, (x=2,z=2); solver=BroydenSolver(tol=1e-12),
            ift=KrylovIFT(itmax=1, rtol=1e-14, atol=0))
        @test_throws ArgumentError DI.gradient(ad_solved_loss, ad, θ,
            map(DI.Constant,(bad,x,off,ps,st,ad))...)
    end
end

# Fail immediately if ANY backend attempts the dense state-Jacobian API for this
# fixture, independent of chunking, AD call counts, or incidental allocations.
struct MatrixFreeOnlyCell{M} <: LuxCore.AbstractLuxLayer
    W::M
end
(l::MatrixFreeOnlyCell)(inp, ps, st) = (l.W*inp[1] .+ ps.b .+ inp[2], st)
function ImplicitLayers._root_jacobian(ad, f::DEQFactor{names,D,<:MatrixFreeOnlyCell},
                                     args...) where {names,D}
    error("dense state Jacobian forbidden by matrix-free test")
end
@testset "LP-002 200-dimensional matrix-free IFT" begin
    rng, n = Xoshiro(14), 200
    W = 0.3 .* randn(rng,n,n) ./ sqrt(n)
    @test opnorm(W) < 0.7
    x, ps, st = randn(rng,n), (b=randn(rng,n),), (;)
    f = DEQFactor(MatrixFreeOnlyCell(W), (x=n,z=n);
        solver=PicardSolver(tol=1e-12, maxiters=200), ift=KrylovIFT(rtol=1e-10,atol=1e-12))
    exactz = (I-W) \ (x+ps.b)
    exactg = (I-W)' \ exactz
    errors = Float64[]
    for ad in (AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake())
        # scalar loss requests one reverse RHS; ForwardDiff also covers JVP solves.
        g = parameter_gradient(p -> sum(abs2,first(solve_state(f,x,p,st;ad)))/2, ps; ad)
        push!(errors, norm(g.b-exactg)/norm(exactg))
        @test g.b ≈ exactg rtol=3e-9 atol=1e-9
    end
    @info "200-dimensional Krylov gradient max relative error" maximum(errors)
end
