# I20: the batched per-replica message kernel materialises Float64 coefficients and accumulators,
# so it runs only when every array it touches is a plain host float array; dual (AD) coefficients
# take the per-replica path, which preserves their scalar types. Oracles: the batched and the
# per-replica messages agree term for term, and ForwardDiff through the batched call matches the
# finite difference.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, ForwardDiff
using LenticulumCore: TrivialBelief

@testset "I20: batched constraint messages preserve dual coefficients" begin
    f = LinearConstraintFactor((x = 1, y = 1), 1; noise = 1.0)
    fs = [f, f]
    # one Gaussian block input with a precision per replica
    inputs = (y = GaussianBlock(reshape([1.0, 2.0], 1, 2), [fill(1.0, 1, 1), fill(2.0, 1, 1)], [false, false]),)
    priors = [TrivialBelief(), TrivialBelief()]
    p = (A = (x = reshape([1.0], 1, 1), y = fill(1.0, 1, 1)), c = [0.0])
    fun(t) = begin
        pp = (A = (x = reshape([t], 1, 1), y = fill(1.0, 1, 1)), c = [0.0])
        out, _ = Mycelium.factor_messages(fs, :x, nothing, inputs, priors, pp, (;))
        sum(out.η)
    end
    v = fun(1.0)
    fd = (fun(1.00001) - fun(0.99999)) / 0.00002
    @test isfinite(v) && isfinite(fd)
    @test ForwardDiff.derivative(fun, 1.0) ≈ fd rtol = 1e-4
    # the batched kernel (plain Float64) and the per-replica path agree term for term
    outb, _ = Mycelium.factor_messages(fs, :x, nothing, inputs, priors, p, (;))
    cols = [NamedTuple{keys(inputs)}(map(b -> Mycelium.replica(b, i), values(inputs))) for i in 1:2]
    outu, _ = Mycelium.factor_messages(fs, :x, nothing, cols, priors, p, (;))
    for i in 1:2
        @test outb.η[:, i] ≈ outu[i].η
        Λi = outb.Λ isa AbstractMatrix ? outb.Λ : outb.Λ[i]
        @test Matrix(Λi) ≈ Matrix(outu[i].Λ)
        @test outb.improper[i] == outu[i].improper
    end
    # a shared-precision block differentiates through the other batched path too
    inputs2 = (y = GaussianBlock(reshape([1.0, 2.0], 1, 2), fill(1.0, 1, 1), [false, false]),)
    fun2(t) = begin
        pp = (A = (x = reshape([t], 1, 1), y = fill(1.0, 1, 1)), c = [0.0])
        out, _ = Mycelium.factor_messages(fs, :x, nothing, inputs2, priors, pp, (;))
        sum(out.η)
    end
    fd2 = (fun2(1.00001) - fun2(0.99999)) / 0.00002
    @test ForwardDiff.derivative(fun2, 1.0) ≈ fd2 rtol = 1e-4
    # Float32 blocks take the same batched kernel and give the same messages as Float64
    inputs32 = (y = GaussianBlock(reshape(Float32[1, 2], 1, 2), [fill(1f0, 1, 1), fill(2f0, 1, 1)], [false, false]),)
    p32 = (A = (x = reshape(Float32[1], 1, 1), y = fill(1f0, 1, 1)), c = Float32[0])
    out32, _ = Mycelium.factor_messages(fs, :x, nothing, inputs32, priors, p32, (;))
    @test out32.η ≈ outb.η atol = 1e-6
    for i in 1:2
        Λi = out32.Λ isa AbstractMatrix ? out32.Λ : out32.Λ[i]
        @test Matrix(Λi) ≈ (outb.Λ isa AbstractMatrix ? outb.Λ : outb.Λ[i]) atol = 1e-6
    end
end
