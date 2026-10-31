using Test
isdefined(@__MODULE__, :FactorialInference) ||
    include(joinpath(@__DIR__, "..", "examples", "factorial_inference.jl"))

@testset "Factorial inference: exact, collapsed BP in Mycelium, mean field (examples/factorial_inference.jl)" begin
    # a cheap version of the script's run
    rows = FactorialInference.run(; Ls = (4, 8), ntrain = 1000, ntest = 1000, starts = 2, iters = 100, ncheck = 5)
    r(L, m) = only(x for x in rows if x.L == L && startswith(x.method, m))
    # few switches: every E-step's fixed point is close to the truth
    @test all(r(4, m).gap_fixed < 0.1 for m in ("exact", "graph", "mean field"))
    # overlapping switches: the factorised posteriors are biased, exact EM is not
    @test r(8, "exact").gap_fixed < 0.1
    @test r(8, "graph").gap_fixed > r(8, "exact").gap_fixed + 0.3
    @test r(8, "mean field").gap_fixed > r(8, "exact").gap_fixed + 0.3
end

@testset "A mixture input to a linear factor is collapsed to its moments (src/gaussian.jl)" begin
    using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random
    import LuxCore
    f = LinearConstraintFactor((a = 2, b = 2), 2; noise = 0.3)
    ps = LuxCore.initialparameters(Xoshiro(1), f)
    mix = LenticulumCore.MixtureBelief([Gaussian([1.0, 0.0], 0.2 * Matrix(1.0I, 2, 2)), Gaussian([-1.0, 2.0], [0.5 0.1; 0.1 0.4])], [0.3, 0.7])
    m1, _ = Mycelium.factor_message(f, :b, nothing, (a = mix,), nothing, ps, NamedTuple())
    m2, _ = Mycelium.factor_message(f, :b, nothing, (a = Lenticulum.moment_match(mix),), nothing, ps, NamedTuple())
    @test m1.η ≈ m2.η && m1.Λ ≈ m2.Λ
end
