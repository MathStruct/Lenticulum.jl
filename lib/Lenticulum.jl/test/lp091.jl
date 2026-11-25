# LP-091: findings of the review of Q4 (element type from the input). Each testset fails on the
# code Codex delivered and passes with the fix; the oracle is the Float64 call.
module LP091
using Test, Lenticulum, LenticulumCore, Mycelium, LuxCore, Random
const LC = LenticulumCore

# The element types of every floating-point payload of a result (flags and labels left out).
numeric_types(x::AbstractArray{<:Real}) = eltype(x) <: Integer ? Set{Type}() : Set{Type}([eltype(x)])
numeric_types(x::AbstractFloat) = Set{Type}([typeof(x)])
numeric_types(x::Union{Tuple,NamedTuple,AbstractArray}) = reduce(union, map(numeric_types, collect(x)); init = Set{Type}())
numeric_types(x::LC.AbstractBelief) = reduce(union, (numeric_types(getfield(x, n)) for n in fieldnames(typeof(x))); init = Set{Type}())
numeric_types(x::LC.GradedEnergy) = numeric_types(x.parts)
numeric_types(x) = Set{Type}()
f32(x::AbstractArray) = Float32.(x)
f32(x::NamedTuple) = map(f32, x)
f32(x::Real) = Float32(x)

@testset "LP-091: review of Q4" begin
    pol = LC.Polarity((x = LC.Unobserved(),))
    @testset "finding 1: GaussianPrior's messages convert the stored mean" begin
        g = GaussianPrior(:x, [0.3], [0.8;;]; learn_noise = true)
        ps = LuxCore.initialparameters(Xoshiro(0), g)
        m64, _ = Mycelium.factor_message(g, :x, pol, (;), nothing, ps, (;))
        m32, _ = Mycelium.factor_message(g, :x, pol, (;), nothing, f32(ps), (;))
        @test numeric_types(m32) == Set{Type}([Float32])
        @test belief_mean(m32) ≈ belief_mean(m64) rtol = 1e-5
    end
    @testset "finding 2: GaussianPrior's energy with a learned covariance" begin
        g = GaussianPrior(:x, [0.3], [0.8;;]; learn_noise = true)
        ps = LuxCore.initialparameters(Xoshiro(0), g)
        e64, _ = LC.energy(g, nothing, nothing, [0.5], ps, (;))
        e32, _ = LC.energy(g, nothing, nothing, [0.5f0], f32(ps), (;))
        @test numeric_types(e32) == Set{Type}([Float32])
        @test e32.parts.fit ≈ e64.parts.fit rtol = 1e-5
        @test e32.parts.complexity ≈ e64.parts.complexity rtol = 1e-5
    end
    @testset "finding 3: integer offsets do not widen a Float32 call" begin
        f = LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.4)
        ps = (A = (x = fill(-1.25f0, 1, 1), y = fill(1.0f0, 1, 1)), c = [1])
        p = LC.Polarity((x = LC.Observed(), y = LC.Unobserved()))
        m, _ = Mycelium.factor_message(f, :y, p, (; x = Gaussian(0.5f0, 0.8f0)), nothing, ps, (;))
        @test numeric_types(m) == Set{Type}([Float32])
        g = GaussianFactor(1 => 1; noise = 0.4)
        pg = (A = fill(1.25f0, 1, 1), b = [1])
        pf = LC.Polarity((x = LC.Observed(), y = LC.Unobserved()))
        mg, _ = Mycelium.factor_message(g, :y, pf, (; x = Gaussian(0.5f0, 0.8f0)), nothing, pg, (;))
        @test numeric_types(mg) == Set{Type}([Float32])
    end
    @testset "SwitchingConstraintFactor's mixture_message follows the input (found by the session)" begin
        f = SwitchingConstraintFactor((x = 1, u = 1), 2, 1; noise = 0.4)
        ps, st = LuxCore.setup(Xoshiro(1), f)
        umix(T) = MixtureBelief([Gaussian(T(0.2), T(0.5)), Gaussian(T(-0.4), T(0.3))], T[0.6, 0.4])
        kb(T) = CategoricalBelief(T[0.3, 0.7])
        p = LC.Polarity(; x = LC.Unobserved(), u = LC.Observed(), k = LC.Observed())
        m64, _ = mixture_message(f, :x, p, (u = umix(Float64), k = kb(Float64)), nothing, ps, st)
        m32, _ = mixture_message(f, :x, p, (u = umix(Float32), k = kb(Float32)), nothing, f32(ps), st)
        @test numeric_types(m32) == Set{Type}([Float32])
        @test Mycelium.belief_logdensity(m32, [0.1f0]) ≈ Mycelium.belief_logdensity(m64, [0.1]) rtol = 1e-4
        pk = LC.Polarity(; x = LC.Observed(), u = LC.Observed(), k = LC.Unobserved())
        mk, _ = mixture_message(f, :k, pk, (x = DiracBelief([0.1f0]), u = umix(Float32)), nothing, f32(ps), st)
        @test numeric_types(mk) == Set{Type}([Float32])
    end
end
end # module
