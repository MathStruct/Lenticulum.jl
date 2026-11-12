using LenticulumCore, LinearAlgebra, Random, Test
using Zygote, ForwardDiff
const LC18 = LenticulumCore

# I18: per-message work of a sweep made cheaper. Each fast path is compared with the code it
# replaces, on generated inputs (gaussian_belief.jl).

# The generic `_nearly_symmetric`, which the method for dense float matrices replaces.
_nearly_symmetric_ref(M) = invoke(LC18._nearly_symmetric, Tuple{Any}, M)

@testset "I18 _nearly_symmetric: the loop agrees with the generic test" begin
    rng = Xoshiro(18)
    tol = sqrt(eps(Float32))
    specials = (NaN, Inf, -Inf, 0.0, -0.0, floatmin(Float64), floatmax(Float64), nextfloat(0.0))
    for T in (Float64, Float32), trial in 1:3000
        n = rand(rng, 0:5)
        S = randn(rng, n, n); A = T.(S + S')
        if n > 0
            kind = trial % 6
            i, j = rand(rng, 1:n), rand(rng, 1:n)
            kind == 1 && (A[i, j] += T(rand(rng, (0.5, 1.0, 2.0)) * tol * maximum(abs, A)))  # near the threshold
            kind == 2 && (A[i, j] = T(rand(rng, specials)))
            kind == 3 && (A[i, j] = A[j, i] = T(rand(rng, specials)))
            kind == 4 && (A .*= T(rand(rng, (1e-300, 1e300, 1e-30, 1e30))))
            kind == 5 && (A = T.(randn(rng, n, n)))                                     # asymmetric
        end
        @test LC18._nearly_symmetric(A) == _nearly_symmetric_ref(A)
    end
    # Non-square: both throw (LP-026 F1; the loop returned false).
    @test_throws DimensionMismatch _nearly_symmetric_ref(zeros(2, 3))
    @test_throws DimensionMismatch LC18._nearly_symmetric(zeros(2, 3))
    @test_throws DimensionMismatch LC18._nearly_symmetric(zeros(Float32, 3, 2))
end

@testset "I18 _symmetric_part: the fused broadcast is bit-identical" begin
    rng = Xoshiro(19)
    for trial in 1:500
        n = rand(rng, 1:5)
        A = randn(rng, n, n) .* 10.0^rand(rng, -300:300)
        @test LC18._symmetric_part(A) == Symmetric(A + (A' - A) / 2)
        @test LC18._symmetric_part(A) !== A
    end
    A = [floatmin(Float64)/4 1.0; 1.0 2.0]       # a subnormal diagonal survives
    @test LC18._symmetric_part(A)[1, 1] == floatmin(Float64)/4
end

@testset "I18 _mean_cov from one factorisation" begin
    rng = Xoshiro(20)
    for trial in 1:300
        n = rand(rng, 1:6)
        L = randn(rng, n, n); Λ = L * L' + 0.1I
        b = GaussianBelief(randn(rng, n), Λ)
        μ, Σ = LC18._mean_cov(b)
        @test μ == belief_mean(b)
        @test Σ == belief_cov(b)
    end
    # Improper beliefs throw what `belief_mean` throws, the first call of the old `_moments`
    # (LP-026 F2: the message named `belief_cov`).
    errmsg(f, b) = try f(b); "" catch e; sprint(showerror, e) end
    for b in (GaussianBelief([1.0, 0.0], [1.0 0.0; 0.0 0.0]),
              GaussianBelief([1.0, 0.0], [1.0 0.0; 0.0 1.0], true),
              GaussianBelief(zeros(2), [1.0 0.2; 10.0 1.0]))
        @test_throws ArgumentError LC18._mean_cov(b)
        @test errmsg(LC18._mean_cov, b) == errmsg(belief_mean, b)
    end
    # A non-`Array` belief takes the generic method.
    bs = GaussianBelief([1.0, 2.0], Symmetric([2.0 0.5; 0.5 1.0]))
    @test LC18._mean_cov(bs) == (belief_mean(bs), belief_cov(bs))
end

@testset "I18 _mean_cov differentiates as its two parts" begin
    rng = Xoshiro(21)
    for trial in 1:20
        n = rand(rng, 1:4)
        L = randn(rng, n, n); Λ0 = L * L' + 0.5I; η0 = randn(rng, n)
        W, w = randn(rng, n, n), randn(rng, n)
        fused(η, Λ) = (mc = LC18._mean_cov(GaussianBelief(η, Λ)); dot(w, mc[1]) + sum(W .* mc[2]))
        split(η, Λ) = (b = GaussianBelief(η, Λ); dot(w, belief_mean(b)) + sum(W .* belief_cov(b)))
        only_mean(η, Λ) = dot(w, LC18._mean_cov(GaussianBelief(η, Λ))[1])
        only_cov(η, Λ) = sum(W .* LC18._mean_cov(GaussianBelief(η, Λ))[2])
        gf, gs = Zygote.gradient(fused, η0, Λ0), Zygote.gradient(split, η0, Λ0)
        @test gf[1] ≈ gs[1]
        @test gf[2] ≈ gs[2]
        gm = Zygote.gradient(only_mean, η0, Λ0)
        @test gm[1] ≈ Zygote.gradient((η, Λ) -> dot(w, belief_mean(GaussianBelief(η, Λ))), η0, Λ0)[1]
        gc = Zygote.gradient(only_cov, η0, Λ0)
        @test gc[2] ≈ Zygote.gradient((η, Λ) -> sum(W .* belief_cov(GaussianBelief(η, Λ))), η0, Λ0)[2]
        # Forward mode goes through the fused code itself.
        x0 = vcat(η0, vec(Λ0))
        unpack(x) = (x[1:n], reshape(x[n+1:end], n, n))
        @test ForwardDiff.gradient(x -> fused(unpack(x)...), x0) ≈ ForwardDiff.gradient(x -> split(unpack(x)...), x0)
    end
end
