using Random
const LuxCore = LenticulumCore.LuxCore

# A scalar quadratic game, deliberately independent of the Gaussian factor implementation.
struct P36Game{Exact} <: AbstractLenticulumFactor end
LuxCore.parameterlength(::P36Game) = 1
LuxCore.statelength(::P36Game) = 1
LuxCore.parameterlength(::CouplingGuardFactor) = 0
LuxCore.initialparameters(::AbstractRNG, ::P36Game) = (w = [0.7],)
LuxCore.initialstates(::AbstractRNG, ::P36Game) = (counter = [2],)
LenticulumCore.energyspace(::P36Game) = ScalarEnergySpace()
LenticulumCore.scalarisation(::P36Game) = IdentityScalarisation()
LenticulumCore.pointwise_energyspace(::P36Game) = ScalarEnergySpace()
LenticulumCore.pointwise_scalarisation(::P36Game) = IdentityScalarisation()
LenticulumCore.exact_product_game(::P36Game{E}) where E = E
LenticulumCore.free_energy(::P36Game, π, y, ps, st) = ((ps.w[1] - y)^2 / 2, st)
LenticulumCore.entropy(::P36Game, π, y, ps, st) = (π, st)
# Give this test family an explicit composite game: the generic container only supplies plumbing.
function LenticulumCore.free_energy(f::ComposedFactor{<:P36Game,<:P36Game}, π, y, ps, st)
    a, sa = free_energy(f.first, π, y, ps.first, st.first)
    b, sb = free_energy(f.second, π, y, ps.second, st.second)
    return GradedEnergy((first = a, second = b)), (first = sa, second = sb)
end

@testset "P36 statistical games: traits, composition and independent products" begin
    rng = Xoshiro(36); f = P36Game{true}(); no = CouplingGuardFactor(:empty)
    @test islearnable(f) && !islearnable(no)
    @test !isfrozen(f)
    @test straight_through(round, x -> 3x, 1.7) == 2
    c = compose(f, f); t = TensorFactor((a = f, b = f))
    for container in (c, t)
        ps, st = setup(rng, container)
        @test LuxCore.parameterlength(container) == 2
        @test LuxCore.statelength(container) == 2
        @test dimension(energyspace(container)) == 2
        @test dimension(pointwise_energyspace(container)) == 2
        e = container === c ? GradedEnergy((first = 2.0, second = 3.0)) : GradedEnergy((a = 2.0, b = 3.0))
        @test scalarise(scalarisation(container), e) == 5
        @test scalarise(pointwise_scalarisation(container), e) == 5
    end
    ps, st = setup(rng, c)
    gr = free_energy_gradient(c, nothing, 1.2, ps, st)
    @test gr.first.w ≈ [-0.5] atol = 1e-8
    @test gr.second.w ≈ [-0.5] atol = 1e-8
    @test_throws ArgumentError free_energy_gradient(c, nothing, 1.2, ps, st; unsupported = true)
    @test_throws ArgumentError free_energy_gradient(compose(f, f; coupling = ExactCoupling()), nothing, 1.2, ps, st)
    @test_throws ArgumentError free_energy(compose(no, no), nothing, 1.2, (;), (;))
    @test_throws ArgumentError entropy(compose(no, no), nothing, 1.2, (;), (;))
    ps, st = setup(rng, t)
    F, sout = free_energy(t, (a = 2.0, b = 3.0), (a = 1.2, b = -0.3), ps, st)
    @test F.parts.a ≈ 0.125 && F.parts.b ≈ 0.5
    @test sout == st
    H, _ = entropy(t, (a = 2.0, b = 3.0), (a = 0, b = 0), ps, st)
    @test H.parts == (a = 2.0, b = 3.0)
    @test LenticulumCore.exact_product_game(t)
    @test !LenticulumCore.exact_product_game(no)
    bad = TensorFactor((a = no,))
    @test !LenticulumCore.exact_product_game(bad)
    for op in (free_energy, entropy)
        @test_throws ArgumentError op(t, nothing, nothing, ps, st)
        @test_throws ArgumentError op(t, (b = 1, a = 2), (a = 1, b = 2), ps, st)
        @test_throws ArgumentError op(bad, (a = 1,), (a = 1,), (a = (;),), (a = (;),))
    end
end
