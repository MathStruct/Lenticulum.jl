using Adapt, JLArrays, LinearAlgebra, Random, Test, LenticulumCore
const DeviceArray = isdefined(Main, :LENTICULUM_DEVICE_ARRAY) ? Main.LENTICULUM_DEVICE_ARRAY : JLArray  # CuArray under gpu/runtests.jl (Q8)

# Q21: the suites run JLArrays 0.5 with `test/support/strict_jlarrays.jl`, so a host array in a
# device broadcast fails here as it does on CUDA (LP-095 finding 3 passed on JLArrays without it).
@testset "a host array in a device broadcast is refused" begin
    JLArrays.allowscalar(false)
    x = DeviceArray(ones(3))
    @test_throws Exception x .+ ones(3)
    @test_throws Exception x .* zeros(Bool, 3)
    @test Array(x .+ 1.0) == fill(2.0, 3)          # scalars, ranges and device arrays still broadcast
    @test Array(x .+ (1:3)) == [2.0, 3.0, 4.0]
    @test Array(x .+ x) == fill(2.0, 3)
end

@testset "Device Gaussian algebra and adaptation" begin
    JLArrays.allowscalar(false)
    rng = MersenneTwister(917)
    for T in (Float32, Float64), n in 1:4
        A = randn(rng, T, n, n)
        Σ = A * A' + I
        μ = randn(rng, T, n)
        reference, device = Gaussian(μ, Σ), Gaussian(DeviceArray(μ), DeviceArray(Σ))
        rtol = T === Float32 ? 1e-4 : 1e-10
        @test device.η isa DeviceArray{T}
        @test device.Λ isa DeviceArray{T}
        @test isproper(device)
        @test Array(device.Λ) ≈ reference.Λ rtol=rtol
        @test Array(belief_mean(device)) ≈ μ rtol=rtol
        @test Array(belief_cov(device)) ≈ Σ rtol=rtol
        @test logpartition(device) ≈ logpartition(reference) rtol=rtol
        C = LenticulumCore._chol(DeviceArray(Σ), "Σ")
        @test Array(C \ DeviceArray(A)) ≈ Σ \ A rtol=rtol
    end
    indefinite = [1.0 2; 2 1]
    for to in (Array, DeviceArray)
        Σ = adapt(to, indefinite)
        @test_throws ArgumentError Gaussian(adapt(to, [0.0, 1]), Σ)
        @test !isproper(GaussianBelief(adapt(to, [0.0, 1]), Σ))
    end
    e = adapt(DeviceArray, ExtendedGaussian([1.0, 2.0], [2.0 0; 0 0], zeros(2, 0)))
    @test e.μ isa DeviceArray && e.Σ isa DeviceArray && e.D isa DeviceArray
    @test Array(hard_directions(e) * hard_directions(e)') ≈ [0.0 0; 0 1]
    a = adapt(DeviceArray, CategoricalBelief([0.25, 0.75]; labels = [:a, :b]))
    @test a.logp isa DeviceArray && a.labels isa Vector
    @test Array(probabilities(a)) ≈ [0.25, 0.75]
end

isdefined(@__MODULE__, :EltypeMatrix) || include("support/eltype_matrix.jl")
include("support/device_matrix.jl")

@testset "Core belief storage matrix" begin
    normal(P, T) = Gaussian(T[0.2, -0.1], T[1.0 0.2; 0.2 0.8])
    samples(P, T) = SampleBelief(T[1 0 2; 2 3 2], T[0.1, 0.2, 0.7])
    extended_normal(P, T) = ExtendedGaussian(T[1, 2], T[0.8 0; 0 0], zeros(T, 2, 0))
    categorical(P, T) = CategoricalBelief(T[1, 3]; labels = [:a, :b])
    fixtures = [
        EltypeMatrix.EltypeCase(:GaussianBelief, normal, [
            :mean => belief_mean, :covariance => belief_cov, :logpartition => logpartition]),
        EltypeMatrix.EltypeCase(:SampleBelief, samples, [:mean => belief_mean, :covariance => belief_cov]),
        EltypeMatrix.EltypeCase(:ExtendedGaussian, extended_normal, [
            :hard_projector => b -> (H = hard_directions(b); H * H'),
            :entropy => support_entropy]),
        EltypeMatrix.EltypeCase(:CategoricalBelief, categorical, [:probabilities => probabilities]),
    ]
    DeviceMatrix.check_device_matrix(fixtures)
end

# Q7 review (LP-092): a device matrix inside a LinearAlgebra wrapper (Symmetric, Adjoint,
# Diagonal) must be factorised on the device, not copied silently to the host by the
# CPU path's `Matrix(M)`, and the results must stay on the device.
@testset "LP-092: wrapped device matrices stay on the device" begin
    JLArrays.allowscalar(false)
    M = [2.0 0.3; 0.3 1.5]
    μ = [1.0, 2.0]
    wraps = (Symmetric(DeviceArray(M)), Symmetric(DeviceArray(M), :L), DeviceArray(M)', transpose(DeviceArray(M)))
    for W in wraps
        b, ref = GaussianBelief(DeviceArray(μ), W), GaussianBelief(μ, M)
        @test isproper(b)
        @test belief_mean(b) isa DeviceArray && Array(belief_mean(b)) ≈ belief_mean(ref)
        @test belief_cov(b) isa DeviceArray && Array(belief_cov(b)) ≈ belief_cov(ref)
        @test logpartition(b) ≈ logpartition(ref)
        @test LenticulumCore._lower_matrix(LenticulumCore._chol(W, "M")) isa DeviceArray
        @test LenticulumCore._lower_matrix(LenticulumCore._precision_factor(W)) isa DeviceArray
        @test LenticulumCore._lower_matrix(LenticulumCore._symmetric_cholesky(W)) isa DeviceArray
        @test LenticulumCore._symmetric_part(W) isa DeviceArray
    end
    D = Diagonal(DeviceArray([2.0, 3.0]))
    C = LenticulumCore._symmetric_cholesky(D)
    @test Array(LenticulumCore._lower_matrix(C)) ≈ Diagonal(sqrt.([2.0, 3.0]))
end

# Q8: the generic device Cholesky against LAPACK where a pivot is infinite (it computed Inf/√Inf = NaN
# for the diagonal entry; LAPACK and cuSOLVER give Inf), and on random SPD matrices.
@testset "Q8: generic device Cholesky agrees with LAPACK" begin
    JLArrays.allowscalar(false)
    rng = MersenneTwister(808)
    for T in (Float32, Float64), n in (1, 2, 5)
        R = randn(rng, T, n, n)
        A = R * R' + T(n) * I
        B = copy(A); B[n, n] = T(Inf)
        for M in (A, B)
            C, ref = LenticulumCore._symmetric_cholesky(DeviceArray(M)), cholesky(Symmetric(M))
            @test isequal(Array(LenticulumCore._lower_matrix(C)) .== Inf, ref.L .== Inf)
            finite = isfinite.(ref.L)
            @test Array(LenticulumCore._lower_matrix(C))[finite] ≈ ref.L[finite] rtol = (T === Float32 ? 1e-5 : 1e-12)
        end
    end
end
