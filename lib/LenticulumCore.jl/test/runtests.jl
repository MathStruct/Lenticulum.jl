using LenticulumCore
using LenticulumCore: Channel
using Statistics: mean, cov
using Test
using LinearAlgebra

struct UnsupportedCoreCoupling <: AbstractGradientCoupling end
struct CouplingGuardFactor <: AbstractLenticulumFactor
    name::Symbol
end

@testset "LP-011 composition coupling guard" begin
    a, b = CouplingGuardFactor(:a), CouplingGuardFactor(:b)
    for f in (ComposedFactor(a, b), ComposedFactor(a, b; coupling = DiagonalCoupling()),
              ComposedFactor(a, b, DiagonalCoupling()),
              ComposedFactor{typeof(a),typeof(b),DiagonalCoupling}(a, b, DiagonalCoupling()))
        @test f.first === a
        @test f.second === b
        @test f.coupling === DiagonalCoupling()
    end
    # All three non-default built-in families, including both score-function baselines,
    # plus a user-defined subtype: the guard is a whitelist, not a blacklist.
    for coupling in (PathwiseCoupling(), ScoreFunctionCoupling(),
                     ScoreFunctionCoupling(0.5), ExactCoupling(), UnsupportedCoreCoupling())
        @test_throws ArgumentError ComposedFactor(a, b; coupling)
        @test_throws ArgumentError ComposedFactor(a, b, coupling)
        @test_throws ArgumentError ComposedFactor{typeof(a),typeof(b),typeof(coupling)}(a, b, coupling)
    end
    @test_throws ArgumentError ComposedFactor(a, b; coupling = nothing)
end

# Test-only residuals; the circle uses the reference finite-difference Jacobian.
struct CircleResidual <: AbstractLenticulumFactor end
LenticulumCore.energy(::CircleResidual, x, a, y, ps, st) = ([sum(abs2, y) - one(eltype(y))], st)
LenticulumCore.pointwise_energyspace(::CircleResidual) = EuclideanEnergySpace(1)
LenticulumCore.pointwise_scalarisation(::CircleResidual) = SampsonNorm()

# Detect accidental localisation at the outgoing rather than incoming state.
struct StatefulResidual <: AbstractLenticulumFactor end
LenticulumCore.energy(::StatefulResidual, x, a, y, ps, st) =
    (st.scale .* y, (scale = 2st.scale,))

struct BilinearResidual <: AbstractLenticulumFactor end
LenticulumCore.energy(::BilinearResidual, x, a, y, ps, st) = (x .* y, st)

@testset "Point-dependent scalarisation localisation" begin
    f, nt = CircleResidual(), NamedTuple()
    ctx = (f, nothing, nothing, [1.2, 0.0], nt, nt)
    fixed = (IdentityScalarisation(), WeightedSum([2.0]), SquaredNorm(),
             SquaredNorm(reshape([3.0], 1, 1)),
             GradedScalarisation((a = SquaredNorm(), b = IdentityScalarisation())))
    for σ in fixed
        @test localise(σ, ctx...) === σ
    end
    # Fixed scalar_energy results and states retain their old meaning.
    r = first(energy(ctx...))
    for σ in fixed[1:4]
        value, st = scalar_energy(ctx...; σ)
        @test value ≈ scalarise(σ, r)
        @test st === nt
    end
    @test !islinear(SampsonNorm())
    @test_throws ArgumentError scalarise(SampsonNorm(), [1.0])
    @test_throws ArgumentError jensen_gap(SampsonNorm(), [[1.0], [2.0]])
    @test_throws ArgumentError jensen_gap(SampsonNorm(), [])
    graded = GradedScalarisation((outer = GradedScalarisation((inner = SampsonNorm(),)),))
    @test !islinear(graded)
    @test_throws ArgumentError localise(graded, ctx...)
    @test_throws ArgumentError jensen_gap(graded, [])
    @test_throws ArgumentError scalar_energy(ctx...; σ = graded)
    for composite in (TensorFactor((circle = f,)), ComposedFactor(f, f))
        @test_throws ArgumentError localise(pointwise_scalarisation(composite), composite,
                                           nothing, nothing, [1.2, 0.0], nt, nt)
    end
    @test_throws ArgumentError SampsonNorm(wrt = :a)
    for λ in (-1, Inf, NaN)
        @test_throws ArgumentError SampsonNorm(; λ)
    end
    @test_throws DimensionMismatch SampsonNorm(Σ = ones(2, 3))
    for Σ in ([1.0 2; 0 1], [1.0 0; 0 -1], Diagonal([1.0, -1]), fill(NaN, 2, 2))
        @test_throws ArgumentError SampsonNorm(; Σ)
    end
    @test SampsonNorm(Σ = Diagonal([1.0, 0])).Σ == Diagonal([1.0, 0])
    @test_throws DimensionMismatch localise(SampsonNorm(Σ = ones(1, 1)), ctx...)

    for ρ in (0.4, 0.9, 1.1, 2.0)
        y = ρ .* [0.6, 0.8]
        value, _ = scalar_energy(f, nothing, nothing, y, nt, nt)
        @test value ≈ (ρ^2 - 1)^2 / (8ρ^2) rtol = 2e-10
        for s in (0.5, 3.0)
            scaled, _ = scalar_energy(f, nothing, nothing, y, nt, nt;
                                     σ = SampsonNorm(Σ = Diagonal(fill(s^2, 2))))
            @test scaled ≈ value / s^2 rtol = 2e-10
        end
    end
    # The relative error is O(|ρ-1|), on both sides of the circle.
    for sign in (-1, 1)
        errors = Float64[]
        for δ in (1e-2, 1e-3, 1e-4, 1e-5)
            ρ = 1 + sign * δ
            value = first(scalar_energy(f, nothing, nothing, [ρ, 0.0], nt, nt))
            ratio = value / ((ρ - 1)^2 / 2)
            @test abs(ratio - 1) < 1.1δ
            push!(errors, abs(ratio - 1))
        end
        @test all(diff(errors) .< 0)
        @test last(errors) < 1.1e-5
    end
    @test_throws ArgumentError scalar_energy(f, nothing, nothing, zeros(2), nt, nt)
    @test isfinite(first(scalar_energy(f, nothing, nothing, zeros(2), nt, nt; σ = SampsonNorm(λ = 0.1))))
    # A PSD but singular input covariance is allowed; singular pushforward is rejected.
    @test_throws ArgumentError localise(SampsonNorm(Σ = zeros(2, 2)), ctx...)
    # Numerical SPD policy: reject a tiny positive eigenvalue relative to the largest.
    sf = StatefulResidual()
    sc = (sf, nothing, nothing, [1.0, 2.0], nt, (scale = 1.0,))
    @test_throws ArgumentError localise(SampsonNorm(Σ = Diagonal([1.0, eps(Float64)/2])), sc...)
    # Avoid overflow from symmetrising A via A+A', and reject overflow of inv(A).
    huge = localise(SampsonNorm(Σ = Diagonal(fill(0.75floatmax(Float64), 2))), sc...)
    @test all(isfinite, huge.M)
    @test all(diag(huge.M) .> 0)
    @test_throws ArgumentError localise(SampsonNorm(Σ = Diagonal(fill(nextfloat(0.0), 2))), sc...)
    v, next = scalar_energy(sc...; σ = SampsonNorm())
    @test v ≈ 2.5 rtol = 2e-10
    @test next == (scale = 2.0,)
    J = energy_jacobian(f, nothing, nothing, [1.2, 0.3], nt, nt; wrt = :y)
    σ = SampsonNorm(Σ = Diagonal([2.0, 3.0]), λ = 0.2)
    M = localise(σ, f, nothing, nothing, [1.2, 0.3], nt, nt).M
    @test gauss_newton_metric(f, nothing, nothing, [1.2, 0.3], nt, nt; wrt = :y, σ) ≈ J' * M * J
    # Noise may be on x while the pull-back still differentiates y.
    xc = (sf, [0.0, 0.0], nothing, [1.0, 2.0], nt, (scale = 1.0,))
    @test localise(SampsonNorm(wrt = :x, λ = 1), xc...).M ≈ I
    bc = (BilinearResidual(), [2.0, 3.0], nothing, [4.0, 5.0], nt, nt)
    noise_x = SampsonNorm(wrt = :x)
    @test localise(noise_x, bc...).M ≈ Diagonal([1/16, 1/25]) rtol = 2e-10
    @test gauss_newton_metric(bc...; wrt = :y, σ = noise_x) ≈
          Diagonal([4/16, 9/25]) rtol = 2e-10
    for T in (Float32, Float64)
        loc = localise(SampsonNorm(), f, nothing, nothing, T[1.2, 0.3], nt, nt)
        @test eltype(loc.M) === T
        @test isfinite(scalarise(loc, T[0.53]))
    end
end

diag_of(C) = [C[i, i] for i in axes(C, 1)]

@testset "LenticulumCore" begin
    @testset "channels and polarity" begin
        # P_in + P_out + P_latent = Id, pairwise products zero.
        p = Polarity(; x = Observed(), y = Unobserved(), h = Latent())
        @test observed_channels(p) == (:x,)
        @test unobserved_channels(p) == (:y,)
        @test latent_channels(p) == (:h,)
        @test length(observed_channels(p)) + length(unobserved_channels(p)) +
              length(latent_channels(p)) == length(keys(p))
        @test ispartition(p)

        # ρ_in defaults to a hard clamp (a categorical cup); the rest are free.
        @test channel_precision(p, :x) == Inf
        @test channel_precision(p, :y) == 1.0

        c = Channel(:x, 3)
        @test channelname(c) === :x
        @test channelspace(c) == 3
    end

    @testset "energy spaces compose by direct sum" begin
        Ec = EuclideanEnergySpace(2)
        Ed = ScalarEnergySpace()
        E = GradedEnergySpace((; first = Ec, second = Ed))
        @test dimension(E) == 3
        @test dimension(Ec ⊕ Ed) == 3
    end

    @testset "graded energy is a vector space" begin
        a = GradedEnergy((; first = [1.0, 2.0], second = 3.0))
        b = GradedEnergy((; first = [0.5, 0.5], second = 1.0))
        @test (a + b)[:first] == [1.5, 2.5]
        @test (a - b)[:second] == 2.0
        @test (2.0 * a)[:first] == [2.0, 4.0]
        @test zero(a)[:second] == 0.0
    end

    @testset "scalarisation recovers AutoBayes Definition 22" begin
        # σ_dc(e_c, e_d) = σ_c(e_c) + σ_d(e_d): the paper's `+` is the scalarisation of ⊕.
        σ = GradedScalarisation((; first = SquaredNorm(), second = IdentityScalarisation()))
        e = compose_energy([3.0, 4.0], 2.0)
        @test scalarise(σ, e) ≈ 0.5 * 25 + 2.0
        @test !islinear(σ)

        σlin = GradedScalarisation((;
            first = WeightedSum([1.0, 2.0]), second = IdentityScalarisation()
        ))
        @test islinear(σlin)
        @test scalarise(σlin, e) ≈ (3.0 + 8.0) + 2.0
    end

    @testset "linear σ ⇒ strict chain rule; convex σ ⇒ lax by the Jensen gap" begin
        # Samples of 𝐅^c(π, y) at (y,b) drawn from the downstream inversion.
        Fc = [[1.0, 0.0], [-1.0, 2.0], [0.0, 1.0], [2.0, -1.0]]

        # Linear scalarisation: σ(E[·]) == E[σ(·)], so the two chain rules agree exactly.
        σl = WeightedSum([1.0, -0.5])
        @test islinear(σl)
        @test jensen_gap(σl, Fc) == 0.0
        @test scalarise(σl, mean(Fc)) ≈ mean(scalarise.(Ref(σl), Fc))

        # Convex scalarisation: the gap is exactly ½ tr Cov (population covariance).
        σc = SquaredNorm()
        gap = jensen_gap(σc, Fc)
        M = reduce(hcat, Fc)'                       # 4×2, samples in rows
        C = cov(M; corrected = false)
        @test gap ≈ 0.5 * sum(diag_of(C)) atol = 1e-12
        @test gap > 0

        # AutoBayes Thm 23 (scalar) = multivariate chain rule + defect.
        Fd = 7.0
        σ = GradedScalarisation((; first = σc, second = IdentityScalarisation()))
        F_multi = scalarise(σ, compose_free_energy(Fc, Fd))
        F_scalar = mean(scalarise.(Ref(σc), Fc)) + Fd
        @test F_scalar ≈ F_multi + chain_rule_defect(σc, Fc)
        @test F_multi <= F_scalar        # Jensen: the multivariate composite is a lower bound
    end

    @testset "entropy composition averages under the downstream inversion" begin
        Hc = [1.0, 2.0, 3.0]
        H = compose_entropy(Hc, 10.0)
        @test H[:first] ≈ 2.0
        @test H[:second] == 10.0
    end

    @testset "SquaredNorm validation, oplus, and Gauss-Newton metric" begin
        @test_throws ArgumentError SquaredNorm([1.0 2.0; 0.0 1.0])
        @test_throws ArgumentError SquaredNorm([1.0 0.0; 0.0 -1.0])
        for M in (Diagonal([1.0, -1.0]), Symmetric([1.0 0.0; 0.0 -1.0]),
                  Hermitian(ComplexF64[1 0; 0 -1]), view([1.0 0.0; 0.0 -1.0], [1, 2], :))
            @test_throws ArgumentError SquaredNorm(M)
            @test scalarise(SquaredNorm(M; check = false), [0.0, 1.0]) ≈ -0.5
        end
        for M in (Diagonal([1.0, 0.0]), Symmetric([1.0 1.0; 1.0 1.0]),
                  Hermitian(ComplexF64[1 im; -im 1]), view([1.0 0.0; 0.0 0.0], [1, 2], :))
            @test SquaredNorm(M).M === M
        end
        σ_m = SquaredNorm([2.0 0.0; 0.0 4.0])
        @test scalarise(σ_m, [1.0, 1.0]) ≈ 0.5 * (2.0 + 4.0)
        @test scalarise(IdentityScalarisation(), [3.5]) ≈ 3.5

        # Non-graded oplus & disjoint graded oplus
        s_pair = SquaredNorm() ⊕ IdentityScalarisation()
        @test s_pair isa GradedScalarisation
        @test_throws ArgumentError oplus(
            GradedEnergySpace((; a = ScalarEnergySpace())),
            GradedEnergySpace((; a = ScalarEnergySpace())),
        )

        # Gauss-Newton metric G = J^T M J
        J = [1.0 2.0; 3.0 4.0]
        G_id = gauss_newton_metric(SquaredNorm(), J)
        @test G_id ≈ J' * J
        G_m = gauss_newton_metric(σ_m, J)
        @test G_m ≈ J' * [2.0 0.0; 0.0 4.0] * J
    end

    @testset "uninformative, isuninformative, and SampleBelief validation" begin
        u32 = uninformative(Float32, 2)
        @test eltype(u32.η) == Float32
        @test isuninformative(u32)
        @test isuninformative(TrivialBelief())
        @test !isuninformative(Gaussian([0.0], fill(1.0, 1, 1)))
        @test !isuninformative(DiracBelief([1.0]))

        @test_throws ArgumentError SampleBelief([[1.0], [2.0]], [1.0])
        @test_throws ArgumentError SampleBelief([[1.0], [2.0]], [-0.5, 1.5])
        @test_throws ArgumentError SampleBelief([[1.0], [2.0]], [NaN, 1.0])
    end
end

@testset "categorical and mixture beliefs: construction" begin
    b = CategoricalBelief([1, 3]; labels = [:a, :b])
    @test probabilities(b) ≈ [0.25, 0.75]
    @test b.labels == [:a, :b]
    @test probabilities(bernoulli(0.2)) ≈ [0.8, 0.2]
    @test_throws ArgumentError CategoricalBelief([-1, 2])
    @test_throws ArgumentError CategoricalBelief([0, 0])
    @test_throws ArgumentError CategoricalBelief([1, 2]; labels = [:a])
    m = MixtureBelief([b, b], [1, 3])                  # raw weights, normalised
    @test mixture_weights(m) ≈ [0.25, 0.75]
    @test mixture_weights(MixtureBelief([b, b])) ≈ [0.5, 0.5]
    @test_throws ArgumentError MixtureBelief(typeof(b)[])
    @test_throws ArgumentError MixtureBelief([b, b], [1.0])
end

@testset "weight normalization is finite and scale invariant" begin
    components = [DiracBelief(0), DiracBelief(1)]
    for T in (Float32, Float64)
        large = floatmax(T)
        small = nextfloat(zero(T))
        for weights in ([large, large], [small, small], T[1, 1])
            @test probabilities(CategoricalBelief(weights)) ≈ T[0.5, 0.5]
            @test mixture_weights(MixtureBelief(components, weights)) ≈ T[0.5, 0.5]
        end
        @test probabilities(CategoricalBelief(T[0, large])) == T[0, 1]
        @test mixture_weights(MixtureBelief(components, T[large, 0])) == T[1, 0]
        @test all(isfinite, CategoricalBelief(T[small, large]).logp)
        for weights in (T[Inf, 1], T[NaN, 1], T[-1, 1], T[0, 0])
            @test_throws ArgumentError CategoricalBelief(weights)
            @test_throws ArgumentError MixtureBelief(components, weights)
        end
    end
    @test_throws ArgumentError CategoricalBelief(Float64[])
end

@testset "Container conformance scope (LP-008/009)" begin
    a, b = CouplingGuardFactor(:a), CouplingGuardFactor(:b)
    for f in (ComposedFactor(a,b), TensorFactor((a=a,b=b)))
        # These describe parameter/energy-space containers, not wired factors. No
        # channels or assembled inversion exist, so the common checker cannot apply.
        @test !applicable(LenticulumCore.channels, f)
        @test !applicable(assemble, f, Polarity((x=Observed(),y=Unobserved())), (;), (;))
        @test isempty(supported_polarities(f))
        @test !applicable(energy, f, nothing, nothing, nothing, (;), (;))
        @test_throws ArgumentError free_energy(f, TrivialBelief(), nothing, (;), (;))
        @test_throws ArgumentError LenticulumCore.entropy(f, TrivialBelief(), nothing, (;), (;))
    end
end

struct LazyPolarityFactor{N} <: AbstractLenticulumFactor
    calls::Base.RefValue{Int}
end
LenticulumCore.channels(::LazyPolarityFactor{N}) where {N} =
    ntuple(i -> Channel(Symbol(:c, i), nothing), N)
function LenticulumCore.supports_polarity(f::LazyPolarityFactor, p::Polarity)
    f.calls[] += 1
    return !isempty(unobserved_channels(p))
end
LenticulumCore.supported_polarities(::LazyPolarityFactor) =
    throw(ArgumentError("use polarities or supports_polarity"))
LenticulumCore.hasnopolarity(::LazyPolarityFactor{N}) where {N} = N == 0
LenticulumCore.isunidirectional(::LazyPolarityFactor{N}) where {N} = N == 1

@testset "Lazy polarity enumeration (LP-017)" begin
    f = LazyPolarityFactor{2}(Ref(0))
    iter = polarities(f)
    @test f.calls[] == 0
    listed = collect(iter)
    @test f.calls[] == 9
    O, U, L = Observed(), Unobserved(), Latent()
    expected = [Polarity((c1=a,c2=b)) for b in (O,U,L) for a in (O,U,L)
                if a isa Unobserved || b isa Unobserved]
    @test listed == expected
    @test all(p -> supports_polarity(f,p), listed)
    @test all(p -> p.precisions == Polarity(p.assignment).precisions, listed)
    @test_throws ArgumentError supported_polarities(f)
    @test !hasnopolarity(f)
    @test !isunidirectional(f)
    @test isunidirectional(LazyPolarityFactor{1}(Ref(0)))
    @test isempty(collect(polarities(LazyPolarityFactor{0}(Ref(0)))))
    large = LazyPolarityFactor{20}(Ref(0))
    @test first(polarities(large))[:c1] isa Unobserved
    @test large.calls[] == 2  # traversal stopped after the first accepted assignment
    @test hasnopolarity(CouplingGuardFactor(:x))  # existing list-based defaults
    @test !isunidirectional(CouplingGuardFactor(:x))
end

@testset "likelihood potential representation" begin
    logφ = x -> 0.5 - x
    b = LogPotentialBelief(logφ)
    @test b isa AbstractPotentialBelief
    @test b isa AbstractBelief
    @test !isproper(b)
    @test b.logφ === logφ
    @test !applicable(belief_mean, b)
    @test !applicable(belief_cov, b)
end

include("parameter_ad.jl")
