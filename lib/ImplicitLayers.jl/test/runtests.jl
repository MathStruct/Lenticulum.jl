using ImplicitLayers
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Polarity
using Mycelium
using LuxCore
using LinearAlgebra
using Random
using Test

# ---------------------------------------------------------------------------
# The oracles.
#
# A LINEAR cell  g(z,x) = Wz + Ux + b  makes every claim about a DEQ closed-form:
#
#     z★      = (I-W)⁻¹(Ux + b)            the fixed point
#     x★      = U⁻¹((I-W)z - b)            the REVERSE solve, which SciML cannot ask for
#     ∂z★/∂x  = (I-W)⁻¹U                   the implicit function theorem
#     Picard converges  ⟺  ρ(W) < 1        the caveat, made testable
#
# A LINEAR vector field  dz/dt = Az  does the same for the flow:
#
#     Φ_{t}(z)     = exp(At)z              the flow
#     tr ∂f/∂z     = tr A                  the CNF divergence, constant
#
# Same philosophy as `GaussianFactor` in the parent package: build the one case where
# everything is computable, then check the framework against arithmetic.
# ---------------------------------------------------------------------------
struct LinearCell{A,B,C} <: LuxCore.AbstractLuxLayer
    W::A
    U::B
    b::C
end
LuxCore.initialparameters(::AbstractRNG, l::LinearCell) = (W = l.W, U = l.U, b = l.b)
LuxCore.initialstates(::AbstractRNG, ::LinearCell) = NamedTuple()
(l::LinearCell)(inp, ps, st) = (ps.W * inp[1] .+ ps.U * inp[2] .+ ps.b, st)

# The simplest possible Lux layer, standing in for an assembled `DeepEquilibriumNetwork`
# or `NeuralODE` — the point being that `LuxFactor` needs to know nothing else about it.
struct Doubler <: LuxCore.AbstractLuxLayer end
LuxCore.initialparameters(::AbstractRNG, ::Doubler) = (s = 2.0,)
LuxCore.initialstates(::AbstractRNG, ::Doubler) = NamedTuple()
(::Doubler)(x, ps, st) = (ps.s .* x, st)

struct LinearDynamics{A} <: LuxCore.AbstractLuxLayer
    A::A
end
LuxCore.initialparameters(::AbstractRNG, l::LinearDynamics) = (A = l.A,)
LuxCore.initialstates(::AbstractRNG, ::LinearDynamics) = NamedTuple()
(l::LinearDynamics)(inp, ps, st) = (ps.A * inp[1], st)

# A contraction: ρ(W) = 0.5 < 1, so even Picard works.
const W₁ = [0.3 0.2; -0.1 0.4]
const U₁ = [1.0 0.5; 0.0 2.0]
const b₁ = [0.5, -0.25]
const CELL = LinearCell(W₁, U₁, b₁)
const RNG = Random.default_rng()

_err(a, b) = sqrt(sum(abs2, a .- b))

zstar(x) = (I - W₁) \ (U₁ * x .+ b₁)
xstar(z) = U₁ \ ((I - W₁) * z .- b₁)

@testset "ImplicitLayers" begin

@testset "solvers: Picard converges on a contraction, Broyden also off one" begin
    # F(u) = u - (Wu + c): root is (I-W)⁻¹c
    c = [1.0, 2.0]
    F(u) = u .- (W₁ * u .+ c)
    want = (I - W₁) \ c

    up, rp = solve_root(F, zeros(2), PicardSolver(tol = 1e-12, maxiters = 500))
    @test rp.converged
    @test up ≈ want
    @test rp.residual <= 1e-12
    @test rp.nfe == rp.iterations + 1        # one evaluation per step, plus the initial one

    ub, rb = solve_root(F, zeros(2), BroydenSolver(tol = 1e-12))
    @test rb.converged
    @test ub ≈ want
    # Broyden is a quasi-Newton method: on a LINEAR residual it should be far quicker
    @test rb.iterations < rp.iterations

    # H₀ = I makes the first Broyden step exactly a Picard step
    u1_broyden, _ = solve_root(F, zeros(2), BroydenSolver(maxiters = 1, tol = 0.0))
    u1_picard, _ = solve_root(F, zeros(2), PicardSolver(maxiters = 1, tol = 0.0))
    @test u1_broyden ≈ u1_picard

    # already at the root ⇒ zero iterations, not one
    u0, r0 = solve_root(F, want, BroydenSolver(tol = 1e-10))
    @test r0.iterations == 0 && r0.converged && r0.nfe == 1

    @test occursin("converged", string(rb))
end

@testset "the contraction caveat is real: Picard diverges, Broyden does not" begin
    # `Implicit Learners.md` §Equilibrium: "this only works if the iteration converges, and
    # unconstrained DEQs need not". Here is a cell for which it does not.
    Wbad = [1.6 0.0; 0.0 1.6]            # ρ(W) = 1.6 > 1
    c = [1.0, -2.0]
    F(u) = u .- (Wbad * u .+ c)
    want = (I - Wbad) \ c                # the fixed point still EXISTS and is unique

    _, rp = solve_root(F, zeros(2), PicardSolver(maxiters = 200, tol = 1e-10))
    @test !rp.converged                  # Picard cannot reach it

    ub, rb = solve_root(F, zeros(2), BroydenSolver(maxiters = 200, tol = 1e-12))
    @test rb.converged                   # Broyden can
    @test ub ≈ want

    # ...and the difference is invisible from the outside unless the report is threaded out,
    # which is why `SolveReport` exists and why nothing here throws.
    @test rp.residual > 1e-6
end

@testset "fd_jacobian and the implicit function theorem" begin
    # residual of the linear cell: r(z) = z - (Wz + Ux + b), so ∂r/∂z = I - W exactly
    x = [1.0, -1.0]
    F(z) = z .- (W₁ * z .+ U₁ * x .+ b₁)
    J = fd_jacobian(F, zeros(2))
    @test J ≈ (I - W₁) rtol = 1e-6

    # the IFT: ∂z★/∂x = (I - ∂_z g)⁻¹ ∂_x g = (I-W)⁻¹U
    S = ift_sensitivity(W₁, U₁)
    @test S ≈ (I - W₁) \ U₁

    # a singular I - ∂_z g is not a numerical accident — it is the relation branching, the
    # same phenomenon the algebraic family calls the discriminant
    @test_throws SingularException ift_sensitivity(Matrix{Float64}(I, 2, 2), U₁)
end

@testset "DEQFactor: the interface, and two polarities where SciML has one" begin
    f = DEQFactor(CELL, (x = 2, z = 2))
    ps, st = LuxCore.setup(RNG, f)
    @test ps.W == W₁ && ps.U == U₁ && ps.b == b₁       # parameters are the cell's, untouched
    @test LuxCore.parameterlength(f) == LuxCore.parameterlength(CELL)
    @test map(LenticulumCore.channelname, LenticulumCore.channels(f)) == (:x, :z)
    @test inchannel(f) === :x && statechannel(f) === :z
    @test indim(f) == 2 && statedim(f) == 2
    @test LenticulumCore.islearnable(f)

    # THE point: a square residual has two directions, and neither is privileged
    @test length(LenticulumCore.supported_polarities(f)) == 2
    @test !LenticulumCore.isunidirectional(f)
    fwd = Polarity(; x = Observed(), z = Unobserved())
    bwd = Polarity(; x = Unobserved(), z = Observed())
    @test LenticulumCore.supports_polarity(f, fwd)
    @test LenticulumCore.supports_polarity(f, bwd)
    @test !LenticulumCore.supports_polarity(f, Polarity(; x = Observed(), y = Unobserved()))

    # the residual is a genuine VECTOR energy, unlike the diffusion factor's
    @test LenticulumCore.energyspace(f) isa LenticulumCore.EuclideanEnergySpace
    @test LenticulumCore.dimension(LenticulumCore.energyspace(f)) == 2
    @test LenticulumCore.scalarisation(f) isa LenticulumCore.SquaredNorm
    r, _ = LenticulumCore.energy(f, [1.0, 0.0], nothing, [0.0, 0.0], ps, st)
    @test r ≈ .-(U₁ * [1.0, 0.0] .+ b₁)
    @test length(r) == 2

    @test_throws ArgumentError DEQFactor(CELL, (x = 2, z = 2, w = 1))
    @test_throws ArgumentError DEQFactor(CELL, (x = 0, z = 2))
    @test_throws ArgumentError DEQFactor(CELL, (a = 2, b = 2))     # keys must match channels
end

@testset "DEQFactor: both directions are exact on the linear oracle" begin
    f = DEQFactor(CELL, (x = 2, z = 2); solver = BroydenSolver(tol = 1e-13))
    ps, st = LuxCore.setup(RNG, f)
    x = [1.0, -0.5]

    # forward — what DeepEquilibriumNetworks.jl does
    z, rep, _ = solve_state(f, x, ps, st)
    @test rep.converged
    @test z ≈ zstar(x)

    # reverse — what it cannot be asked for. SAME residual, SAME solver.
    z_target = [0.7, 1.3]
    xs, rep2, _ = solve_input(f, z_target, ps, st)
    @test rep2.converged
    @test xs ≈ xstar(z_target)
    # and it really is the inverse: solving forward from x★ returns z_target
    zback, _, _ = solve_state(f, xs, ps, st)
    @test zback ≈ z_target

    # the IFT sensitivity, against the closed form
    @test deq_sensitivity(f, x, z, ps, st) ≈ (I - W₁) \ U₁ rtol = 1e-5

    # messages in both directions, as Diracs
    mz, _ = Mycelium.factor_message(f, :z, nothing, (x = DiracBelief(x),), TrivialBelief(), ps, st)
    @test mz isa DiracBelief && mz.value ≈ zstar(x)
    mx, _ = Mycelium.factor_message(f, :x, nothing, (z = DiracBelief(z_target),), TrivialBelief(), ps, st)
    @test mx isa DiracBelief && mx.value ≈ xstar(z_target)

    # no information in ⇒ no information out
    m0, _ = Mycelium.factor_message(f, :z, nothing, NamedTuple(), TrivialBelief(), ps, st)
    @test m0 isa TrivialBelief
    @test_throws ArgumentError Mycelium.factor_message(
        f, :nope, nothing, (x = DiracBelief(x),), TrivialBelief(), ps, st)

    # the prior warm-starts the solver: starting AT the answer costs zero iterations
    _, warm, _ = solve_state(f, x, ps, st; z₀ = zstar(x))
    @test warm.iterations == 0 && warm.nfe == 1

    # on the relation the energy vanishes
    F, _ = Mycelium.local_free_energy(f, (x = DiracBelief(x), z = DiracBelief(zstar(x))), ps, st)
    @test F < 1e-20
    Foff, _ = Mycelium.local_free_energy(f, (x = DiracBelief(x), z = DiracBelief(zstar(x) .+ 1)), ps, st)
    @test Foff > 0.1

    # the lens is a SolverInversion, which is what lens.jl reserved for this family
    lens, _ = LenticulumCore.assemble(f, Polarity(; x = Observed(), z = Unobserved()), ps, st)
    @test lens.inversion isa LenticulumCore.SolverInversion
    @test LenticulumCore.ispure(lens.model)
    b, _ = LenticulumCore.invert(lens, TrivialBelief(), (x = DiracBelief(x),), ps, st)
    @test b.value ≈ zstar(x)
end

@testset "DEQFactor: a non-square residual has only one direction" begin
    # dim x = 3, dim z = 2: solving for x is 2 equations in 3 unknowns and is not a
    # well-posed root-find. The factor says so rather than solving it badly.
    cell = LinearCell([0.3 0.2; -0.1 0.4], [1.0 0.5 0.0; 0.0 2.0 1.0], [0.5, -0.25])
    f = DEQFactor(cell, (x = 3, z = 2))
    ps, st = LuxCore.setup(RNG, f)

    @test length(LenticulumCore.supported_polarities(f)) == 1
    @test LenticulumCore.isunidirectional(f)
    @test LenticulumCore.supports_polarity(f, Polarity(; x = Observed(), z = Unobserved()))
    @test !LenticulumCore.supports_polarity(f, Polarity(; x = Unobserved(), z = Observed()))
    @test_throws ArgumentError solve_input(f, [1.0, 2.0], ps, st)

    # the forward direction still works fine
    z, rep, _ = solve_state(f, [1.0, 0.0, -1.0], ps, st)
    @test rep.converged
    @test z ≈ (I - cell.W) \ (cell.U * [1.0, 0.0, -1.0] .+ cell.b)
end

@testset "flows: RK4 against exp(At), and the round trip" begin
    A = [-0.4 0.3; -0.2 -0.5]
    vf(z, t) = A * z
    z₀ = [1.0, -2.0]
    t₀, t₁ = 0.0, 1.5
    exact = exp(A * (t₁ - t₀)) * z₀

    z_rk = integrate(vf, z₀, t₀, t₁, RK4Integrator(steps = 100))
    @test z_rk ≈ exact rtol = 1e-9

    z_eu = integrate(vf, z₀, t₀, t₁, EulerIntegrator(steps = 100))
    @test z_eu ≈ exact rtol = 1e-2
    # RK4 is emphatically better at the same step count
    @test _err(z_rk, exact) < _err(z_eu, exact) / 1000

    # fourth order: halving the step should cut the error by ~16
    e1 = _err(integrate(vf, z₀, t₀, t₁, RK4Integrator(steps = 8)), exact)
    e2 = _err(integrate(vf, z₀, t₀, t₁, RK4Integrator(steps = 16)), exact)
    @test 8 < e1 / e2 < 40

    # THE property: integrating backwards inverts the flow. Same function, endpoints swapped.
    back = integrate(vf, z_rk, t₁, t₀, RK4Integrator(steps = 100))
    @test back ≈ z₀ rtol = 1e-9
    # ...but NOT bit-exactly. The round trip is exact in exact arithmetic only.
    @test back != z₀
    @test 0 < _err(back, z₀) < 1e-9

    # the CNF divergence: tr(∂f/∂z) = tr A, constant, so Δlogdet = tr(A)·Δt
    z1, Δ = integrate_with_divergence(vf, z₀, t₀, t₁, RK4Integrator(steps = 100))
    @test z1 ≈ exact rtol = 1e-9
    @test Δ ≈ tr(A) * (t₁ - t₀) rtol = 1e-5
end

@testset "NeuralODEFactor: bidirectional for free, because a flow is invertible" begin
    A = [-0.4 0.3; -0.2 -0.5]
    dyn = LinearDynamics(A)
    f = NeuralODEFactor(dyn, 2; tspan = (0.0, 1.0), integrator = RK4Integrator(steps = 100))
    ps, st = LuxCore.setup(RNG, f)
    @test ps.A == A
    @test map(LenticulumCore.channelname, LenticulumCore.channels(f)) == (:z0, :z1)
    @test startchannel(f) === :z0 && endchannel(f) === :z1

    # both polarities, unconditionally — no dimension test, no solver, no caveat
    @test length(LenticulumCore.supported_polarities(f)) == 2
    @test !LenticulumCore.isunidirectional(f)
    @test LenticulumCore.supports_polarity(f, Polarity(; z0 = Observed(), z1 = Unobserved()))
    @test LenticulumCore.supports_polarity(f, Polarity(; z0 = Unobserved(), z1 = Observed()))

    z₀ = [1.0, -2.0]
    exact = exp(A) * z₀
    z₁, _ = flow_forward(f, z₀, ps, st)
    @test z₁ ≈ exact rtol = 1e-9
    zb, _ = flow_reverse(f, z₁, ps, st)
    @test zb ≈ z₀ rtol = 1e-9

    # the inversion is labelled EXACT, not SolverInversion — see neuralode.md §4.3
    lens, _ = LenticulumCore.assemble(f, Polarity(; z0 = Unobserved(), z1 = Observed()), ps, st)
    @test lens.inversion isa LenticulumCore.ExactInversion
    b, _ = LenticulumCore.invert(lens, TrivialBelief(), (z1 = DiracBelief(z₁),), ps, st)
    @test b isa DiracBelief
    @test b.value ≈ z₀ rtol = 1e-9

    # messages both ways
    m1, _ = Mycelium.factor_message(f, :z1, nothing, (z0 = DiracBelief(z₀),), TrivialBelief(), ps, st)
    @test m1.value ≈ exact rtol = 1e-9
    m0, _ = Mycelium.factor_message(f, :z0, nothing, (z1 = DiracBelief(exact),), TrivialBelief(), ps, st)
    @test m0.value ≈ z₀ rtol = 1e-9
    @test_throws ArgumentError Mycelium.factor_message(
        f, :zz, nothing, (z0 = DiracBelief(z₀),), TrivialBelief(), ps, st)

    # on the relation the residual vanishes — a good invariant, a poor diagnostic
    F, _ = Mycelium.local_free_energy(f, (z0 = DiracBelief(z₀), z1 = DiracBelief(z₁)), ps, st)
    @test F < 1e-18
    Foff, _ = Mycelium.local_free_energy(f, (z0 = DiracBelief(z₀), z1 = DiracBelief(z₁ .+ 1)), ps, st)
    @test Foff > 0.5

    # the density correction, for a continuous normalising flow
    _, Δ, _ = flow_logdet(f, z₀, ps, st)
    @test Δ ≈ tr(A) rtol = 1e-5

    # a flow cannot change dimension, hence one `dim` for both channels
    @test LenticulumCore.channelspace(LenticulumCore.channels(f)[1]) ==
          LenticulumCore.channelspace(LenticulumCore.channels(f)[2])
    @test_throws ArgumentError NeuralODEFactor(dyn, 0)
    @test_throws ArgumentError NeuralODEFactor(dyn, 2; tspan = (1.0, 1.0))
end

@testset "LuxFactor: wrapping the assembled layer gives one polarity" begin
    # This is what wrapping a `DeepEquilibriumNetwork` or a `NeuralODE` object gets you: the
    # model works, and it is unidirectional. The contrast with DEQFactor is the whole point.
    f = LuxFactor(Doubler(), :a => :b; dims = (2, 2))
    ps, st = LuxCore.setup(RNG, f)

    @test length(LenticulumCore.supported_polarities(f)) == 1
    @test LenticulumCore.isunidirectional(f)          # ← the whole finding, in one assertion
    @test LenticulumCore.supports_polarity(f, Polarity(; a = Observed(), b = Unobserved()))
    @test !LenticulumCore.supports_polarity(f, Polarity(; a = Unobserved(), b = Observed()))

    y, _ = apply_layer(f, [1.0, 2.0], ps, st)
    @test y ≈ [2.0, 4.0]
    m, _ = Mycelium.factor_message(f, :b, nothing, (a = DiracBelief([1.0, 2.0]),), TrivialBelief(), ps, st)
    @test m isa DiracBelief && m.value ≈ [2.0, 4.0]

    # asking for the input channel is refused, with a message naming the alternative
    err = try
        Mycelium.factor_message(f, :a, nothing, (b = DiracBelief([2.0, 4.0]),), TrivialBelief(), ps, st)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("one direction", err.msg)
    @test occursin("DEQFactor", err.msg)

    # postprocess extracts a plain array — what the SciML layers need
    g = LuxFactor(Doubler(), :a => :b; postprocess = v -> v[1:1])
    yg, _ = apply_layer(g, [1.0, 2.0], LuxCore.setup(RNG, g)...)
    @test yg == [2.0]

    # a Lux layer has no residual of its own
    @test LenticulumCore.energyspace(f) isa LenticulumCore.ScalarEnergySpace
    @test first(LenticulumCore.energy(f, [1.0], nothing, [1.0], ps, st)) == 0.0
    @test first(Mycelium.local_free_energy(f, NamedTuple(), ps, st)) == 0.0
    @test_throws ArgumentError LuxFactor(Doubler(), :a => :a)

    # the inversion is TrivialInversion: there is nothing to invert
    lens, _ = LenticulumCore.assemble(f, Polarity(; a = Observed(), b = Unobserved()), ps, st)
    @test lens.inversion isa LenticulumCore.TrivialInversion
end

@testset "the same cell: 1 polarity wrapped, 2 polarities exposed" begin
    # The comparison the package exists to make, on one object.
    wrapped = LuxFactor(CELL, :in => :out; dims = (2, 2))
    exposed = DEQFactor(CELL, (x = 2, z = 2))
    @test length(LenticulumCore.supported_polarities(wrapped)) == 1
    @test length(LenticulumCore.supported_polarities(exposed)) == 2
    @test LuxCore.parameterlength(wrapped) == LuxCore.parameterlength(exposed)
end

end
