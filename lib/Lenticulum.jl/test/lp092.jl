# LP-092 (Q7 review): two device paths that the one-dimensional fixtures missed.
# A nonlinear factor whose Jacobian is a device `Diagonal` reached `Array(::Diagonal{<:JLArray})`
# in the host spectral helper (scalar indexing); Poisson EP with a two-dimensional latent
# broadcast `_count` over device counts, whose error message cannot compile into a kernel.
using Test, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :DeviceMatrix) || include("../../LenticulumCore.jl/test/support/device_matrix.jl")
isdefined(@__MODULE__, :LenticulumWideFixtures) || include("support/wide_fixtures.jl")

@testset "LP-092: wide device fixtures that failed" begin
    fixtures = filter(f -> f.name in (:Nonlinear22, :PoissonEP2), LenticulumWideFixtures.fixtures())
    @test length(fixtures) == 2
    DeviceMatrix.check_device_matrix(fixtures)
end

# LP-092, Gemini's review: factor configurations the fixtures do not build.
using Adapt, JLArrays, LinearAlgebra
const DeviceArray = isdefined(Main, :LENTICULUM_DEVICE_ARRAY) ? Main.LENTICULUM_DEVICE_ARRAY : JLArray  # CuArray under gpu/runtests.jl (Q8)
@testset "LP-092: device paths of default and edge configurations" begin
    JLArrays.allowscalar(false)
    dev(x) = adapt(DeviceArray, x)
    polarity(f, target, observed) = only(p for p in supported_polarities(f)
        if Set(observed_channels(p)) == Set(observed) && target in unobserved_channels(p))
    msg(f, target, inputs, prior, ps) =
        first(Mycelium.factor_message(f, target, polarity(f, target, keys(inputs)), inputs, prior, ps, (;)))
    G2 = Gaussian([0.5, -0.2], [0.9 0.15; 0.15 0.7])
    @testset "default difference Jacobian" begin
    # NonlinearFactor's default Jacobian (central differences) indexed x[i] and wrote e[i]
    nl = NonlinearFactor(x -> x .^ 2 .+ x, 2 => 2; noise = [0.5 0.1; 0.1 0.4])
    for (target, inputs) in ((:y, (; x = G2)), (:x, (; y = G2)))
        ref = msg(nl, target, inputs, G2, (;))
        out = msg(dev(nl), target, map(dev, inputs), dev(G2), (;))
        @test DeviceMatrix.device_agrees(out, ref, 1e-8)
        @test DeviceMatrix.on_device(DeviceMatrix.device_payload(out))
    end
    end
    @testset "likelihood with z fixed" begin
    # A likelihood whose predictor fixes z returned a host `uninformative`
    lf = LikelihoodFactor(OrdinalProbitLikelihood([-1.0, 1.0]), 2; A = [0.0 0.0], b = [0.2])
    out = msg(dev(lf), :x, (; y = dev(DiracBelief([2]))), dev(G2), (;))
    @test DeviceMatrix.device_agrees(out, msg(lf, :x, (; y = DiracBelief([2])), G2, (;)), 1e-10)
    @test DeviceMatrix.on_device(DeviceMatrix.device_payload(out))
    end
    @testset "bivariate Poisson entropy" begin
    # Bivariate Poisson's conditional entropy read λ[1], λ[2], λ[3]
    z = [0.1, 0.2, -0.3]
    @test Lenticulum._conditional_entropy(BivariatePoissonLikelihood(), DeviceArray(z)) ≈
          Lenticulum._conditional_entropy(BivariatePoissonLikelihood(), z) rtol = 1e-10
    end
    @testset "categorical variable entropy" begin
    # The Bethe entropy of a categorical variable iterated its device log-probabilities
    c = CategoricalBelief([0.6, 0.3, 0.1])
    @test Mycelium.variable_entropy(dev(c)) ≈ Mycelium.variable_entropy(c) rtol = 1e-12
    end
    @testset "Cholesky triangle" begin
    # The device Cholesky read the lower triangle; the CPU one (Symmetric, :U) reads the upper
    A = [2.0 3.0; 0.0 2.0]
    @test issuccess(LenticulumCore._symmetric_cholesky(DeviceArray(A); check = false)) ==
          issuccess(LenticulumCore._symmetric_cholesky(A; check = false))
    B = [2.0 0.5; 0.3 2.0]
    @test Array(LenticulumCore._lower_matrix(LenticulumCore._symmetric_cholesky(DeviceArray(B)))) ≈
          LenticulumCore._lower_matrix(LenticulumCore._symmetric_cholesky(B))
    end
end

# LP-092, the benchmark: on the CPU, Q7's closures for `_host_spectral` assigned to a name of
# the enclosing function (`F = _host_spectral(A -> (F = svd(A); …))`), which boxes it, so the
# rest of the function ran uninferred (`message/nonlinear` 1.65× slower). `_first_device` looped
# over heterogeneous tuples, and `_covariance_at` did not specialise on its varargs.
@testset "LP-092: CPU paths stay inferred" begin
    nl = NonlinearFactor(x -> [x[1]^2 + x[2], sin(x[2])], 2 => 2; noise = 0.25)
    m, S = [0.3, -0.4], [0.5 0.1; 0.1 0.4]
    @test (@inferred Lenticulum._jacobian(nl, m)) isa Matrix{Float64}
    @test (@inferred Lenticulum.linearize(nl, m, S)) isa Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64}}
    @test (@inferred Lenticulum._range_svd([1.0 0.5; 0.0 1.0; 2.0 1.0])) isa Tuple{NamedTuple,Int}
    @test (@inferred LenticulumCore._first_device((m, S, (a = m, b = Gaussian(m, S))))) === nothing
end
