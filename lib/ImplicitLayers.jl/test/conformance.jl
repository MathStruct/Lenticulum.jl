include(joinpath(@__DIR__, "../../../test/factor_conformance.jl"))
# Doubler uses a scalar parameter; LuxCore default counting handles arrays only.
LuxCore.parameterlength(::Doubler) = 1

@testset "Point factor and lens games (LP-009)" begin
    nt = (;)
    f = LuxFactor(Doubler(), :x => :y)
    ps, st = LuxCore.setup(Xoshiro(42), f)
    check_factor_conformance(f, only(supported_polarities(f)); π = TrivialBelief(),
        inputs = (x = DiracBelief([3.0]),), ps, st, pointwise = ([3.0], nothing, [6.0]),
        game_prior = DiracBelief([6.0]), game_y = DiracBelief([3.0]),
        msgs = (x = DiracBelief([3.0]), y = DiracBelief([6.0])),
        expected_site = DiracBelief([6.0]), expected_posterior = DiracBelief([6.0]))
    @test first(free_energy(f, TrivialBelief(), DiracBelief([3.0]), ps, st)) == 0
    @test first(LenticulumCore.entropy(f, TrivialBelief(), DiracBelief([3.0]), ps, st)) == 0
    @test_throws ArgumentError free_energy(f, TrivialBelief(), nothing, ps, st)
    @test_throws ArgumentError LenticulumCore.entropy(f, TrivialBelief(), nothing, ps, st)
    # g(z,x)=z/2+x: z*=2x; dz/dt=0: z1=z0. Both give exact, nonzero
    # incoming residuals and zero solved residuals. The anchor links the two games.
    deq = DEQFactor(LinearCell(fill(0.5,1,1), ones(1,1), zeros(1)), (x=1,z=1))
    # A latent companion gives no point to solve from; only O/U splits are supported.
    @test !supports_polarity(deq, Polarity((x=LenticulumCore.Latent(),z=Unobserved())))
    @test !supports_polarity(deq, Polarity((x=Unobserved(),z=LenticulumCore.Latent())))
    ode = NeuralODEFactor(LinearDynamics(zeros(1,1)), 1; tspan=(0.0,1.0))
    for f in (deq, ode)
        ps, st = LuxCore.setup(Xoshiro(42), f)
        names = Tuple(channelname(c) for c in LenticulumCore.channels(f))
        xy = ([1.0], nothing, [3.0])
        expected_energy = f isa DEQFactor ? 0.125 : 2.0
        @test first(scalar_energy(f, xy..., ps, st)) ≈ expected_energy
        for p in supported_polarities(f)
            src = only(observed_channels(p)); dst = only(unobserved_channels(p))
            v = src == names[1] ? [1.0] : [3.0]
            solved = f isa DEQFactor ? (dst == names[2] ? [2.0] : [1.5]) : v
            inputs = NamedTuple{(src,)}((DiracBelief(v),))
            msgs = NamedTuple{names}((DiracBelief([1.0]), DiracBelief([3.0])))
            check_factor_conformance(f, p; π = TrivialBelief(), inputs, ps, st,
                pointwise = xy, msgs, expected_site = DiracBelief(solved),
                expected_posterior = DiracBelief(solved),
                game_prior = DiracBelief([1.0]), game_y = DiracBelief([3.0]))
            @test first(local_free_energy(f, msgs, ps, st)) ≈ expected_energy
            @test first(free_energy(f, DiracBelief([1.0]), DiracBelief([3.0]), ps, st)) ≈ expected_energy
            @test first(LenticulumCore.entropy(f, DiracBelief([1.0]), DiracBelief([3.0]), ps, st)) == 0
            solved_xy = dst == names[2] ? (v, nothing, solved) : (solved, nothing, v)
            @test first(scalar_energy(f, solved_xy..., ps, st)) ≈ 0 atol=1e-16
            lens, _ = assemble(f, p, ps, st)
            unanchored, _ = free_energy(lens, TrivialBelief(), inputs, ps, st)
            @test unanchored ≈ first(scalar_energy(f, solved_xy..., ps, st)) atol=1e-16
            @test first(scalar_free_energy(lens, TrivialBelief(), inputs, ps, st; σ=WeightedSum(2.0))) ≈ 2unanchored
            @test first(LenticulumCore.entropy(lens, TrivialBelief(), inputs, ps, st)) == 0
            u₀ = dst == names[2] ? [3.0] : [1.0]
            @test isinf(first(free_energy(lens, DiracBelief(u₀), inputs, ps, st)))
            @test first(free_energy(lens, DiracBelief(solved), inputs, ps, st)) ≈ 0 atol=1e-16
            # r=z/2-x gives Jz=1/2, Jx=-1; r=z1-z0 gives Jend=1,
            # Jstart=-1. Thus ½ Δu'J'J Δu is 1/8 (DEQ), 2 (ODE).
            J = energy_jacobian(f, solved_xy..., ps, st; wrt = dst == names[2] ? :y : :x)
            @test only(J) ≈ (f isa DEQFactor && dst == names[2] ? 0.5 : dst == names[2] ? 1.0 : -1.0)
            Λ = J' * J
            soft = GaussianBelief(Λ * u₀, Λ)
            anchored, _ = free_energy(lens, soft, inputs, ps, st)
            @test anchored ≈ expected_energy
            @test first(scalar_free_energy(lens, soft, inputs, ps, st)) ≈ expected_energy
            @test first(free_energy(f, DiracBelief([1.0]), DiracBelief([3.0]), ps, st)) ≈ anchored
            @test anchored - unanchored ≈ dot(u₀ - solved, Λ * (u₀ - solved)) / 2
            # The bare Dirac observation and named-input forms have identical meaning.
            @test first(free_energy(lens, soft, DiracBelief(v), ps, st)) ≈ anchored
            for bad in (TrivialBelief(), Gaussian(0.0, 1.0), nothing)
                @test_throws ArgumentError free_energy(lens, soft, bad, ps, st)
                @test_throws ArgumentError LenticulumCore.entropy(lens, soft, bad, ps, st)
            end
            @test_throws ArgumentError free_energy(lens, LenticulumCore.uninformative(1), inputs, ps, st)
            @test_throws ArgumentError free_energy(lens, DiracBelief([1.0, 2.0]), inputs, ps, st)
            softp = Polarity(p.assignment, map(_ -> 1.0, p.precisions))
            softlens, _ = assemble(f, softp, ps, st)
            @test_throws ArgumentError free_energy(softlens, soft, inputs, ps, st)
        end
        for (π, y) in ((TrivialBelief(), DiracBelief([3.0])),
                       (Gaussian(1.0, 1.0), DiracBelief([3.0])),
                       (DiracBelief([1.0]), Gaussian(3.0, 1.0)),
                       (DiracBelief([1.0]), nothing))
            @test_throws ArgumentError free_energy(f, π, y, ps, st)
            @test_throws ArgumentError LenticulumCore.entropy(f, π, y, ps, st)
        end
    end
end

@testset "Matrix anchors and custom channel names" begin
    # For the existing linear oracle r=(I-W₁)z-U₁x-b₁, Jz=I-W₁, Jx=-U₁.
    f = DEQFactor(CELL, (drive=2,state=2); channels=(:drive,:state),
                  solver=BroydenSolver(tol=1e-13))
    ps, st = LuxCore.setup(Xoshiro(42), f)
    x = [1.0, -0.5]
    z = zstar(x)
    δ = [1.0, -2.0]
    for p in supported_polarities(f)
        forward = only(unobserved_channels(p)) == :state
        Jexact = forward ? I-W₁ : -U₁
        u₀ = (forward ? z : x) + δ
        x₀, z₀ = forward ? (x,u₀) : (u₀,z)
        incoming = first(free_energy(f, DiracBelief(x₀), DiracBelief(z₀), ps, st))
        @test incoming ≈ sum(abs2, Jexact * δ)/2
        J = energy_jacobian(f, x, nothing, z, ps, st; wrt=forward ? :y : :x)
        @test J ≈ Jexact
        Λ = J'J
        @test abs(Λ[1,2]) > 0 # exercises the off-diagonal penalty
        lens, _ = assemble(f, p, ps, st)
        src = only(observed_channels(p))
        inputs = NamedTuple{(src,)}((DiracBelief(forward ? x : z),))
        @test first(free_energy(lens, GaussianBelief(Λ*u₀, Λ), inputs, ps, st)) ≈ incoming
    end
    @test_throws ArgumentError free_energy(f, DiracBelief([1.0]), DiracBelief(z), ps, st)
end

struct CountingGameCell <: LuxCore.AbstractLuxLayer end
LuxCore.initialstates(::AbstractRNG, ::CountingGameCell) = (calls=0,)
(::CountingGameCell)(inp, ps, st) = (inp[1]./2 .+ inp[2], (calls=st.calls+1,))

@testset "Lens games retain solve error and thread state" begin
    # An early-stopped solver returns its initial point z=0: r=0/2-1=-1.
    f = DEQFactor(LinearCell(fill(0.5,1,1), ones(1,1), zeros(1)), (x=1,z=1);
                  solver=BroydenSolver(maxiters=0))
    ps, st = LuxCore.setup(Xoshiro(42), f)
    lens, _ = assemble(f, first(supported_polarities(f)), ps, st)
    @test first(free_energy(lens, TrivialBelief(), DiracBelief([1.0]), ps, st)) ≈ 1/2
    # Starting at z=3 gives r=1/2, cost=1/8, with zero anchor displacement.
    @test first(free_energy(lens, Gaussian(3.0, 4.0), DiracBelief([1.0]), ps, st)) ≈ 1/8

    # One Euler step for dz/dt=z/2 has forward multiplier 3/2, reverse 1/2.
    # Given end=3, reverse gives start=3/2, forward gives 9/4: residual=3/4.
    ode = NeuralODEFactor(LinearDynamics(fill(0.5,1,1)), 1;
                          tspan=(0.0,1.0), integrator=EulerIntegrator(steps=1))
    ps, st = LuxCore.setup(Xoshiro(42), ode)
    lens, _ = assemble(ode, last(supported_polarities(ode)), ps, st)
    @test first(free_energy(lens, TrivialBelief(), DiracBelief([3.0]), ps, st)) ≈ 9/32
    # Gaussian mean=1, precision=2 adds ½*2*(3/2-1)²=1/4.
    @test first(free_energy(lens, Gaussian(1.0, 0.5), DiracBelief([3.0]), ps, st)) ≈ 17/32

    counted = DEQFactor(CountingGameCell(), (x=1,z=1))
    ps, st = LuxCore.setup(Xoshiro(42), counted)
    lens, _ = assemble(counted, first(supported_polarities(counted)), ps, st)
    F, next = free_energy(lens, DiracBelief([2.0]), DiracBelief([1.0]), ps, st)
    @test F == 0
    @test next == (calls=2,) # one root check, then one energy evaluation
    @test last(free_energy(counted, DiracBelief([1.0]), DiracBelief([3.0]), ps, st)) == (calls=1,)
    @test last(LenticulumCore.entropy(lens, TrivialBelief(), DiracBelief([1.0]), ps, st)) == st
end

# r(x,z)=z+z²/4-x: at x=3 the positive root is z*=2, Jz=2.
# For z₀=2+h, r=2h+h²/4, hence incoming energy=2h²+h³/2+h⁴/32.
# The solved-point game with precision Jz'Jz=4 is 2h²; the gap is cubic.
struct QuadraticGameCell <: LuxCore.AbstractLuxLayer end
(::QuadraticGameCell)(inp, ps, st) = (inp[2] .- inp[1].^2 ./ 4, st)

@testset "Nonlinear DEQ: Gauss–Newton identity has cubic energy error" begin
    f = DEQFactor(QuadraticGameCell(), (x=1,z=1); solver=BroydenSolver(tol=1e-13))
    ps, st = LuxCore.setup(Xoshiro(42), f)
    lens, _ = assemble(f, first(supported_polarities(f)), ps, st)
    @test first(invert(lens, TrivialBelief(), (x=DiracBelief([3.0]),), ps, st)).value ≈ [2.0]
    J = energy_jacobian(f, [3.0], nothing, [2.0], ps, st; wrt=:y)
    @test only(J) ≈ 2.0
    Λ = J' * J
    gaps = Float64[]
    for h in (1/4, 1/8, 1/16, 1/32)
        u₀ = [2+h]
        incoming, _ = free_energy(f, DiracBelief([3.0]), DiracBelief(u₀), ps, st)
        solved, _ = free_energy(lens, GaussianBelief(Λ*u₀, Λ), (x=DiracBelief([3.0]),), ps, st)
        @test incoming ≈ 2h^2 + h^3/2 + h^4/32
        @test solved ≈ 2h^2
        gap = incoming - solved
        @test gap ≈ h^3/2 + h^4/32 rtol=1e-6
        @test 0.49 < gap/h^3 < 0.51
        push!(gaps, gap)
    end
    @test all(7.9 .< gaps[1:end-1] ./ gaps[2:end] .< 8.1)
end
