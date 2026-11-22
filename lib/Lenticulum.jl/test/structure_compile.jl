# I1: the graph drivers' ForwardDiff calls take one dual type whatever the structure
# (`LenticulumCoreForwardDiffDIExt`, `lib/Mycelium.jl/src/passing.md`, "Compilation per structure").
# The values are those of the default tag; a caller's derivative through a factor's constant still nests.
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

# The position-velocity chain of `bench/kalman_chain.jl`, with every edge coupled, and the data
# value `y` as a parameter of the test (a caller's dual in a factor's constant).
function _i1_chain(T; y = 0.3, coupling = ExactCoupling())
    b = GraphBuilder()
    for t in 0:T; variable!(b, Symbol(:x, t), 2); end
    factor!(b, :prior, GaussianPrior(:x, zeros(2), Matrix(1.0I, 2, 2))); connect!(b, :prior, :x, :x0; coupling)
    for t in 1:T
        factor!(b, Symbol(:f, t), GaussianFactor(2 => 2; noise = 0.1, channels = (:a, :b)))
        connect!(b, Symbol(:f, t), :a, Symbol(:x, t - 1); coupling); connect!(b, Symbol(:f, t), :b, Symbol(:x, t); coupling)
        variable!(b, Symbol(:y, t), 2)
        factor!(b, Symbol(:h, t), GaussianFactor(2 => 2; noise = 0.5, channels = (:a, :b)))
        connect!(b, Symbol(:h, t), :a, Symbol(:x, t); coupling); connect!(b, Symbol(:h, t), :b, Symbol(:y, t); coupling)
        factor!(b, Symbol(:d, t), DataFactor(:v, [y * t, -y]))
        connect!(b, Symbol(:d, t), :v, Symbol(:y, t); direction = Emitting(), coupling)
    end
    g = build(b)
    rng = Xoshiro(3)
    ps = LuxCore.initialparameters(rng, g); st = LuxCore.initialstates(rng, g)
    ps = map(p -> haskey(p, :A) ? (A = p.A .+ 0.1 .* randn(rng, size(p.A)), b = 0.1 .* randn(rng, length(p.b))) : p, ps)
    return g, ps, st
end

@testset "I1: one dual type across structures, same gradients" begin
    X = Base.get_extension(LenticulumCore, :LenticulumCoreForwardDiffDIExt)
    backends = map((3, 4)) do T
        g, ps, st = _i1_chain(T)
        sched = tree_schedule(g)
        G(ad) = free_energy_gradient(MessageStore(g), g, ps, st; ad, schedule = sched)
        gf, gz, gr = G(AutoForwardDiff()), G(AutoZygote()), G(nothing)
        @test LenticulumCore._flatten(gf) ≈ LenticulumCore._flatten(gz) rtol = 1e-10
        @test LenticulumCore._flatten(gf) ≈ LenticulumCore._flatten(gr) rtol = 1e-5
        # The implicit adjoint takes the same backends.
        gi = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoForwardDiff(), schedule = sched, adjoint = :implicit)
        gj = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, adjoint = :implicit)
        @test LenticulumCore._flatten(gi) ≈ LenticulumCore._flatten(gj) rtol = 1e-8
        # A real graph has no dual for the scan to find, so the fixed tag is taken.
        θ = LenticulumCore._flatten(ps)
        LenticulumCore._graph_ad(AutoForwardDiff(), θ, (ps, g, st, sched))
    end
    @test typeof(backends[1]) === typeof(backends[2])
    @test backends[1].tag === ForwardDiff.Tag(X._GraphTag(), Float64)
end

@testset "I1: a caller's ForwardDiff through a factor's constant and a replay" begin
    function grad(y)
        g, ps, st = _i1_chain(3; y)
        return LenticulumCore._flatten(free_energy_gradient(MessageStore(g), g, ps, st;
            ad = AutoForwardDiff(), schedule = tree_schedule(g)))
    end
    grad(0.3)                                    # the fixed tag exists before the caller's
    D = ForwardDiff.derivative(grad, 0.3)
    h = 1e-5
    @test D ≈ (grad(0.3 + h) - grad(0.3 - h)) / 2h rtol = 1e-6 atol = 1e-8
    @test any(!iszero, D)
end
