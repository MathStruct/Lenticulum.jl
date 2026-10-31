using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, Test
using LenticulumCore: DiracBelief
isdefined(@__MODULE__, :MeasurementFactors) ||
    include(joinpath(@__DIR__, "..", "examples", "measurement_factor.jl"))

@testset "Measurement factor (examples/measurement_factor.jl)" begin
    M = MeasurementFactors
    # 1. orthogonal lines y = ±x with weights 0.3/0.7: eigen-lines are the lines, the Born
    # weights are the mixture weights (equal x-variances), and the held-out conditional
    # likelihood is that of the true model
    A = M.experiment([π / 4, -π / 4], [0.3, 0.7])
    @test maximum(A.err) < 0.5
    w, m = M.born(A.f, [1.0])
    @test w ≈ [0.3, 0.7] atol = 0.02
    @test only.(m) ≈ [1.0, -1.0] atol = 0.02
    @test abs(A.ll_measure - A.ll_oracle) < 0.02
    @test A.ll_measure > A.ll_gauss + 1
    # one-shot measurements land on each line with its Born weight
    draws = [only(M.measure(A.f, [1.0]; rng = Xoshiro(i))) for i in 1:4000]
    @test count(>(0), draws) / 4000 ≈ w[1] atol = 0.03
    # as a graph factor: data x = 1 → measurement → y, the marginal is the mixture
    b = GraphBuilder()
    variable!(b, :x, 1); variable!(b, :y, 1)
    factor!(b, :data, DataFactor(:x, [1.0])); factor!(b, :meas, A.f)
    connect!(b, :data, :x, :x; direction = Emitting())
    connect!(b, :meas, :x, :x; direction = Absorbing())
    connect!(b, :meas, :y, :y; direction = Emitting())
    g = build(b)
    marg, _, _, _ = infer!(g, tree_schedule(g), NamedTuple(), NamedTuple())
    @test mixture_weights(marg.y) ≈ w
    # 2. lines 60° apart: eigenvectors are orthogonal, so they cannot be the lines. This
    # documents the limit, it does not pass by accident: both eigen-lines are ~30° off and the
    # conditional likelihood is far below the true model's.
    B = M.experiment([0.0, π / 3], [0.5, 0.5])
    @test minimum(B.err) > 25
    @test B.ll_measure < B.ll_oracle - 1
end
