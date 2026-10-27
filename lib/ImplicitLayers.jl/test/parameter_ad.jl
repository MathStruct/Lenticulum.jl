using DifferentiationInterface, ADTypes, ForwardDiff, Zygote, ChainRulesCore
using Enzyme, EnzymeCore, Mooncake
const DI = DifferentiationInterface
const LC = LenticulumCore
const ENZYME_FLOW_BACKENDS = (
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)),
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward)),
)

# State really affects the dynamics: verifies functional state threading as well as
# an empty-state linear field. At stage k, dz/dt = a*z + k*c.
struct ADCountingDynamics <: LuxCore.AbstractLuxLayer end
(::ADCountingDynamics)(inp, ps, st) = (ps.A * inp[1] .+ st.k .* ps.c, (k = st.k + 1,))

ad_solved_parameters(θ, f, x, off, ps, st, ad) =
    first(solve_state(f, x, LC._unflatten(θ, off, ps), st; ad))
ad_solved_loss(θ, args...) = sum(abs2, ad_solved_parameters(θ, args...)) / 2
ad_solved_input(x, f, ps, st, ad) = first(solve_state(f, x, ps, st; ad))
ad_warm_input(x, f, ps, st, ad) = first(solve_state(f, x, ps, st; ad, z₀=x))
ad_warm_only(z0, f, x, ps, st, ad) = first(solve_state(f, x, ps, st; ad, z₀=z0))

@testset "LP-002 DEQ implicit solve: four AD backends" begin
    f = DEQFactor(CELL, (x = 2, z = 2); solver = BroydenSolver(tol = 1e-12))
    ps, st = LuxCore.setup(Xoshiro(3), f)
    x = [0.7, -0.3]
    z = zstar(x)
    Id = Matrix{Float64}(I, 2, 2)
    R = hcat(kron(z', Id), kron(x', Id), Id)
    exact = (I - ps.W) \ R
    θ, off = LC._flatten(ps), LC.offset_tree(ps)
    for ad in (AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake())
        @test deq_sensitivity(f, x, z, ps, st; wrt = :ps, ad) ≈ exact atol = 1e-10
        @test deq_sensitivity(f, x, z, ps, st; ad) ≈ (I-ps.W) \ ps.U atol = 1e-10
        solved = let f=f, x=x, off=off, ps=ps, st=st, ad=ad
            θ -> first(solve_state(f, x, LC._unflatten(θ, off, ps), st; ad))
        end
        J = DI.jacobian(ad_solved_parameters, ad, θ, map(DI.Constant, (f, x, off, ps, st, ad))...)
        @test J ≈ exact atol = 1e-9
        @test J ≈ LC._jacobian(nothing, solved, θ) atol = 2e-7
        @test DI.jacobian(ad_solved_input, ad, x, map(DI.Constant, (f, ps, st, ad))...) ≈ (I-ps.W) \ ps.U atol = 1e-9
        # Even a dual-valued warm start must not differentiate iterations.
        @test DI.jacobian(ad_warm_input, ad, x, map(DI.Constant, (f, ps, st, ad))...) ≈ (I-ps.W) \ ps.U atol = 1e-9
        @test iszero(DI.jacobian(ad_warm_only, ad, x,
            map(DI.Constant, (f, x, ps, st, ad))...))
        # A nonzero downstream loss, so this checks the solve pullback rather than
        # the identically zero residual at a solved point.
        g = parameter_gradient(p -> sum(abs2, first(solve_state(f, x, p, st; ad))) / 2, ps; ad)
        @test LC._flatten(g) ≈ exact'z atol = 1e-9
        @test energy_jacobian(f, x, nothing, z, ps, st; wrt = :ps, ad) ≈ -R atol = 1e-10
        y = z .+ [0.2, -0.1]
        Ry = hcat(kron(y', Id), kron(x', Id), Id)
        r = (I-ps.W)*y - ps.U*x - ps.b
        eg = energy_gradient(f, x, nothing, y, ps, st; ad)
        @test LC._flatten(eg) ≈ -Ry'r atol = 1e-10
        fg = free_energy_gradient(f, DiracBelief(x), DiracBelief(y), ps, st; ad)
        @test LC._flatten(fg) ≈ LC._flatten(eg) atol = 1e-10
    end
    for mode in (Enzyme.Forward, Enzyme.Reverse)
        ad = AutoEnzyme(; mode)
        ctx = map(DI.Constant, (f, x, off, ps, st, ad))
        @test DI.jacobian(ad_solved_parameters, ad, θ, ctx...) ≈ exact atol=1e-9
        @test DI.gradient(ad_solved_loss, ad, θ, ctx...) ≈ exact'z atol=1e-9
    end
    bad = DEQFactor(CELL, (x = 2, z = 2); solver = PicardSolver(maxiters = 1, tol = 1e-15))
    for ad in (AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake())
        @test_throws ArgumentError solve_state(bad, x, ps, st; ad)
    end
    @test_throws ArgumentError solve_state(f, x, ps, st; ad=AutoFiniteDiff())
end

@testset "LP-002 discrete Euler/RK4 parameter derivatives" begin
    a, z0, y, n = -0.35, [1.2], [0.5], 7
    for itg in (EulerIntegrator(n), RK4Integrator(n))
        f = NeuralODEFactor(LinearDynamics(fill(a,1,1)), 1; integrator = itg)
        ps, st = LuxCore.setup(Xoshiro(4), f)
        h = 1/n
        q = a*h
        P = itg isa EulerIntegrator ? 1+q : 1+q+q^2/2+q^3/6+q^4/24
        dP = itg isa EulerIntegrator ? h : h*(1+q+q^2/2+q^3/6)
        dz = n * P^(n-1) * dP * only(z0)
        @test only(first(flow_forward(f, z0, ps, st))) ≈ P^n * only(z0)
        for ad in (nothing, AutoForwardDiff(), AutoZygote(), AutoMooncake(), ENZYME_FLOW_BACKENDS...)
            J = energy_jacobian(f, z0, nothing, y, ps, st; wrt = :ps, ad)
            @test only(J) ≈ -dz atol = 2e-9
            g = energy_gradient(f, z0, nothing, y, ps, st; ad)
            @test only(g.A) ≈ (P^n * only(z0) - only(y)) * dz atol = 2e-9
            gf = free_energy_gradient(f, DiracBelief(z0), DiracBelief(y), ps, st; ad)
            @test gf.A ≈ g.A atol = 2e-9
        end
        counted = NeuralODEFactor(ADCountingDynamics(), 1; integrator = itg)
        p, s = (A = fill(a,1,1), c = [0.03]), (k = 1,)
        expected_calls = n * (itg isa EulerIntegrator ? 1 : 4)
        @test last(flow_forward(counted, z0, p, s)).k == 1 + expected_calls
        fd = energy_gradient(counted, z0, nothing, y, p, s)
        for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake(), ENZYME_FLOW_BACKENDS...)
            g = energy_gradient(counted, z0, nothing, y, p, s; ad)
            @test g.A ≈ fd.A atol = 1e-7
            @test g.c ≈ fd.c atol = 1e-7
        end
        @test s.k == 1
        for ad in (AutoEnzyme(), AutoEnzyme(; mode = Enzyme.Reverse),
                   AutoEnzyme(; mode = Enzyme.Forward))
            hint = "AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))"
            @test_throws (msg -> startswith(msg, "ArgumentError:") && occursin(hint, msg)) energy_jacobian(f, z0, nothing, y, ps, st; wrt=:ps, ad)
            @test_throws (msg -> startswith(msg, "ArgumentError:") && occursin(hint, msg)) energy_gradient(f, z0, nothing, y, ps, st; ad)
            @test_throws (msg -> startswith(msg, "ArgumentError:") && occursin(hint, msg)) free_energy_gradient(f, DiracBelief(z0), DiracBelief(y), ps, st; ad)
        end
    end
end

@testset "LP-002 Lux wrapper scoring is opt-in" begin
    f = LuxFactor(Doubler(), :x => :y; dims = (1,1), residual = true)
    ps, st = LuxCore.setup(Xoshiro(5), f)
    for ad in (nothing, AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake())
        @test only(energy_jacobian(f, [3.0], nothing, [7.0], ps, st; wrt = :ps, ad)) ≈ -3 atol = 1e-9
        @test energy_gradient(f, [3.0], nothing, [7.0], ps, st; ad).s ≈ -3 atol = 1e-9
        @test free_energy_gradient(f, DiracBelief([3.0]), DiracBelief([7.0]), ps, st; ad).s ≈ -3 atol = 1e-9
    end
    @test_throws ArgumentError LuxFactor(Doubler(), :x => :y; residual = true)
end

# Data -> endpoint cavities; model edges need only absorb for this objective.
function ode_training_graph(a, target; integrator = EulerIntegrator(5))
    b = GraphBuilder()
    variable!(b, :z0); variable!(b, :z1)
    factor!(b, :input, DataFactor(:z0, [1.0]))
    factor!(b, :target, DataFactor(:z1, [target]))
    factor!(b, :model, NeuralODEFactor(LinearDynamics(fill(a,1,1)), 1;
        integrator))
    connect!(b, :input, :z0, :z0; direction = Emitting())
    connect!(b, :target, :z1, :z1; direction = Emitting())
    connect!(b, :model, :z0, :z0; direction = Absorbing())
    connect!(b, :model, :z1, :z1; direction = Absorbing())
    return build(b)
end
@testset "LP-002 graph parameter step learns the generating ODE parameter" begin
    truth, n = 0.4, 5
    target = (1 + truth/n)^n
    for ad in (AutoForwardDiff(), AutoZygote(), AutoMooncake(), ENZYME_FLOW_BACKENDS...)
        g = ode_training_graph(-0.2, target)
        ps, st = LuxCore.setup(Xoshiro(6), g)
        store = MessageStore(g)
        _, st = propagate!(store, g, tree_schedule(g), ps, st)
        before = first(scalar_free_energy(store, g, ps, st))
        grad = free_energy_gradient(store, g, ps, st; ad)
        @test grad.input === (;)
        @test grad.target === (;)
        @test only(grad.model.A) ≈ ((1-0.2/n)^n-target) * (1-0.2/n)^(n-1) atol = 1e-10
        os = nothing
        for _ in 1:100
            ps, st, os = parameter_step(store, g, ps, st; ad, rule = GradientDescent(0.2), optimiser_state = os)
        end
        @test only(ps.model.A) ≈ truth atol = 1e-8
        @test first(scalar_free_energy(store, g, ps, st)) < before * 1e-12
        # Nesterov: gradient evaluated at its look-ahead, update applied to old p.
        os = (model = [0.1],)
        at = merge(ps, (model = (A = ps.model.A .+ 0.05,),))
        gg = free_energy_gradient(store, g, at, st; ad)
        pp, _, newos = parameter_step(store, g, ps, st; ad, rule = Nesterov(0.2, 0.5), optimiser_state = os)
        @test newos.model ≈ [0.05] - 0.2vec(gg.model.A)
        @test vec(pp.model.A) ≈ vec(ps.model.A) + newos.model
    end
    g = ode_training_graph(-0.2, target)
    ps, st = LuxCore.setup(Xoshiro(6), g)
    store = MessageStore(g)
    _, st = propagate!(store, g, tree_schedule(g), ps, st)
    @test_throws ArgumentError parameter_step(store, g, ps, st; ad=AutoEnzyme())
end

# The actual top-level factor, not a hand-written surrogate, with its fixed noise
# precision. r = y-Ax-b gives Dr/Dvec(A) = -(x' ⊗ I), Dr/Db = -I.
using Lenticulum: GaussianFactor
@testset "LP-002 GaussianFactor AD parameter Jacobian and gradients" begin
    f = GaussianFactor(2 => 2; noise = [0.7 0.1; 0.1 0.5])
    ps, st = (A = [0.2 -0.3; 0.4 0.1], b = [0.1, -0.2]), (;)
    x, y = [0.8, -1.2], [0.5, 0.7]
    Id = Matrix{Float64}(I, 2, 2)
    J = -hcat(kron(x', Id), Id)
    r = y - ps.A*x - ps.b
    for ad in (AutoForwardDiff(), AutoZygote())
        @test energy_jacobian(f, x, nothing, y, ps, st; wrt = :ps, ad) ≈ J atol = 1e-10
        @test gauss_newton_metric(f, x, nothing, y, ps, st; wrt = :ps, ad) ≈ J'*(f.Q\J) atol = 1e-10
        grad = energy_gradient(f, x, nothing, y, ps, st; ad)
        @test grad.A ≈ -(f.Q\r)*x' atol = 1e-10
        @test grad.b ≈ -(f.Q\r) atol = 1e-10
        gf = free_energy_gradient(f, DiracBelief(x), DiracBelief(y), ps, st; ad)
        @test gf.A ≈ grad.A atol=1e-10
        @test gf.b ≈ grad.b atol=1e-10
    end
end

struct QuadraticADCell <: LuxCore.AbstractLuxLayer end
(::QuadraticADCell)(inp, ps, st) = (ps.a .* inp[1].^2 .+ ps.b .+ inp[2], st)
@testset "LP-002 nonlinear IFT and singular residual rejection" begin
    ps, st = (a = 0.2, b = 0.4), (;)
    f = DEQFactor(QuadraticADCell(), (x=1,z=1); solver = BroydenSolver(tol=1e-12))
    x = [0.1]
    # a*z^2-z+(b+x)=0, choose the smaller root.
    z = (1-sqrt(1-4ps.a*(ps.b+only(x))))/(2ps.a)
    expected = [z^2, 1.0] ./ (1-2ps.a*z)
    singular = DEQFactor(LinearCell(ones(1,1), zeros(1,1), zeros(1)), (x=1,z=1))
    sp = (W=ones(1,1), U=zeros(1,1), b=zeros(1))
    for ad in (AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake())
        got = parameter_gradient(p -> only(first(solve_state(f, x, p, st; ad))), ps; ad)
        @test LC._flatten(got) ≈ expected atol=1e-9
        @test_throws SingularException DI.jacobian(ad_solved_input, ad, [0.0],
            map(DI.Constant, (singular, sp, st, ad))...)
    end
end

struct StatefulADCell <: LuxCore.AbstractLuxLayer end
(::StatefulADCell)(inp, ps, st) = (ps.w .* inp[1] .+ inp[2] .+ st.bias, (bias=st.bias+1,))
@testset "LP-002 root uses fixed incoming Lux state" begin
    f = DEQFactor(StatefulADCell(), (x=1,z=1); solver=BroydenSolver(tol=1e-12))
    ps, st, x = (w=0.25,), (bias=0.1,), [0.7]
    exact = (only(x)+st.bias)/(1-ps.w)
    for ad in (nothing, AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake())
        z, rep, out = solve_state(f, x, ps, st; ad)
        @test only(z) ≈ exact atol=1e-11
        @test rep.converged
        @test out.bias == st.bias+1
        if ad !== nothing
            g = parameter_gradient(p -> only(first(solve_state(f, x, p, st; ad))), ps; ad)
            @test g.w ≈ exact/(1-ps.w) atol=1e-10
        end
    end
    @test st.bias == 0.1
end

struct GraphADCell <: LuxCore.AbstractLuxLayer end
LuxCore.initialparameters(::AbstractRNG, ::GraphADCell) = (w = -0.2,)
LuxCore.initialstates(::AbstractRNG, ::GraphADCell) = (;)
(::GraphADCell)(inp, ps, st) = (ps.w .* inp[1] .+ inp[2], st)
@testset "LP-002 graph parameter step through a DEQ factor" begin
    truth, x = 0.4, [0.7]
    target = x ./ (1-truth)
    for ad in (AutoForwardDiff(), AutoZygote(), AutoEnzyme(), AutoMooncake())
        b = GraphBuilder()
        variable!(b, :x); variable!(b, :z)
        factor!(b, :input, DataFactor(:x, x))
        factor!(b, :target, DataFactor(:z, target))
        model = DEQFactor(GraphADCell(), (x=1, z=1))
        factor!(b, :model, model)
        connect!(b, :input, :x, :x; direction=Emitting())
        connect!(b, :target, :z, :z; direction=Emitting())
        connect!(b, :model, :x, :x; direction=Absorbing())
        connect!(b, :model, :z, :z; direction=Absorbing())
        g = build(b)
        ps, st = LuxCore.setup(Xoshiro(7), g)
        store = MessageStore(g)
        _, st = propagate!(store, g, tree_schedule(g), ps, st)
        grad = free_energy_gradient(store, g, ps, st; ad)
        @test grad.model.w ≈ -(target[1]*(1-ps.model.w)-x[1])*target[1] atol=1e-10
        before = first(scalar_free_energy(store, g, ps, st))
        os = nothing
        for _ in 1:100
            ps, st, os = parameter_step(store, g, ps, st; ad,
                rule=GradientDescent(0.2), optimiser_state=os)
        end
        @test ps.model.w ≈ truth atol=1e-10
        @test first(scalar_free_energy(store, g, ps, st)) < before*1e-12
        @test first(solve_state(model, x, ps.model, st.model; ad)) ≈ target atol=1e-10
    end
end
