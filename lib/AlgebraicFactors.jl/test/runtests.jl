using AlgebraicFactors
using LenticulumCore
using Mycelium
using LinearAlgebra
using LuxCore
using Random
using Test

# §2.3 obligation only: these fixtures are not the §3 polynomial-factor API.
# Monomial order: (1, z₁, z₂, z₁², z₁z₂, z₂²).
struct GaugeResidual{Damped} <: AbstractLenticulumFactor end
monomials2(z) = [one(eltype(z)), z[1], z[2], z[1]^2, z[1]*z[2], z[2]^2]
monomial_jacobian2(z) = [0 0; 1 0; 0 1; 2z[1] 0; z[2] z[1]; 0 2z[2]]
LenticulumCore.energy(::GaugeResidual, x, a, y, ps, st) = (ps.Θ * monomials2(y), st)
LenticulumCore.pointwise_energyspace(::GaugeResidual) = EuclideanEnergySpace(2)
LenticulumCore.pointwise_scalarisation(::GaugeResidual) = SampsonNorm()
function LenticulumCore.energy_jacobian(::GaugeResidual, x, a, y, ps, st; wrt = :y, ε = 1e-6)
    wrt === :y || throw(ArgumentError("test residual only differentiates y"))
    return ps.Θ * monomial_jacobian2(y)
end
LenticulumCore.sampson_damping(::GaugeResidual{true}, x, a, y, ps, st) = ps.Θ * ps.Θ'

@testset "Sampson gauge obligation (§2.3)" begin
    rng = Xoshiro(2307)
    f, identity_f, nt = GaugeResidual{true}(), GaugeResidual{false}(), NamedTuple()
    Θ = randn(rng, 2, 6)
    # B has singular values 0.7 and 1.8, independent of the random orthogonal bases.
    Q = Matrix(qr(randn(rng, 2, 2)).Q)
    U = Matrix(qr(randn(rng, 2, 2)).Q)
    B = Q * Diagonal([0.7, 1.8]) * U'
    @test cond(B) ≈ 1.8 / 0.7
    score(f, Θ, z; σ = SampsonNorm()) = first(scalar_energy(f, nothing, nothing, z, (; Θ), nt; σ))
    algebraic_differences, identity_differences = Float64[], Float64[]
    for _ in 1:20
        z = randn(rng, 2)
        J = energy_jacobian(f, nothing, nothing, z, (; Θ), nt; wrt = :y)
        Jfd = invoke(energy_jacobian,
            Tuple{AbstractLenticulumFactor,Any,Any,Any,Any,Any},
            f, nothing, nothing, z, (; Θ), nt; wrt = :y)
        @test J ≈ Jfd rtol = 2e-10 atol = 2e-9
        for λ in (0.0, 0.3), Σ in (nothing, [2.0 0.2; 0.2 0.7])
            σ = SampsonNorm(; λ, Σ)
            @test score(f, Θ, z; σ) ≈ score(f, B*Θ, z; σ) rtol = 1e-10
        end
        push!(identity_differences, abs(score(identity_f, Θ, z; σ = SampsonNorm(λ = 0.3)) -
                                       score(identity_f, B*Θ, z; σ = SampsonNorm(λ = 0.3))))
        push!(algebraic_differences, abs(score(f, Θ, z; σ = SquaredNorm()) -
                                        score(f, B*Θ, z; σ = SquaredNorm())))
        # Retraction at zero orthonormalises a full-row-rank representative.
        Θorth = retract(Grassmann(2, 6), Θ, zero(Θ))
        R = Θorth * Θ'  # Θ' = Θorth' * R, positive-diagonal QR convention
        Bqr = inv(R')
        @test Θorth ≈ Bqr * Θ rtol = 2e-14
        for λ in (0.0, 0.3)
            @test score(f, Θ, z; σ = SampsonNorm(; λ)) ≈
                  score(f, Θorth, z; σ = SampsonNorm(; λ)) rtol = 1e-10
        end
    end
    # For seed 2307 and these 20 points, each contrast must exceed 1e-3 somewhere.
    @test maximum(identity_differences) > 1e-3
    @test maximum(algebraic_differences) > 1e-3
    # Two separable quadratics at z=0, full row rank Θ but rank-zero J.
    singular = [-1.0 0 0 1 0 0; -2.0 0 0 0 0 1]
    z = zeros(2)
    @test iszero(energy_jacobian(f, nothing, nothing, z, (Θ = singular,), nt))
    @test_throws ArgumentError score(f, singular, z)
    @test_throws ArgumentError score(f, B*singular, z)
    for Σ in (nothing, [2.0 0.2; 0.2 0.7])
        σ = SampsonNorm(λ = 0.3, Σ = Σ)
        value = score(f, singular, z; σ)
        @test isfinite(value)
        @test value ≈ score(f, B*singular, z; σ) rtol = 1e-10
    end
end

struct UnsupportedGrassmannRule <: AbstractUpdateRule end

# Use the common checker with independent direct-message site/posterior fixtures.
include(joinpath(@__DIR__, "../../../test/factor_conformance.jl"))

# Independent enumeration oracles: explicit integer-state loops and ordinary products.
# Moderate parameters keep these probability-domain references well conditioned.
states(sizes) = collect(Iterators.product((1:n for n in sizes)...))[:]
function oracle_weights(A, η, sizes, incoming; skip = 0)
    xs = states(sizes)
    w = [exp(sum(η[r] * A[r, k] for r in axes(A, 1))) *
         prod(i == skip ? 1.0 : incoming[i][x[i]] for i in eachindex(sizes))
         for (k, x) in enumerate(xs)]
    return xs, w
end
function oracle_message(A, η, sizes, incoming, t)
    xs, w = oracle_weights(A, η, sizes, incoming; skip = t)
    out = [sum(w[k] for (k, x) in enumerate(xs) if x[t] == j) for j in 1:sizes[t]]
    return out / sum(out)
end
function oracle_statistics(A, η, sizes, incoming; normalised = false)
    xs, w = oracle_weights(A, η, sizes, incoming)
    q = w / sum(w)
    logits = [sum(η[r] * A[r, k] for r in axes(A, 1)) for k in eachindex(xs)]
    offset = normalised ? log(sum(exp.(logits))) : 0.0
    U = sum(q[k] * (offset - logits[k]) for k in eachindex(q))
    H = -sum(p == 0 ? 0.0 : p * log(p) for p in q)
    return U - H, H
end
design(sizes) = [mod(3r + 5k + r*k, 7) - 3 for r in 1:4, k in 1:prod(sizes)]
polarity(names, t, observed) = Polarity(NamedTuple{names}(map(eachindex(names)) do i
    i == t ? Unobserved() : i in observed ? Observed() : Latent()
end |> Tuple))

@testset "AlgebraicFactors" begin
@testset "Grassmann geometry" begin
    rng = Xoshiro(2401)
    for (k, m) in ((1, 1), (1, 5), (2, 6), (4, 4))
        M = Grassmann(k, m)
        Θ = rand(rng, M)
        G, Δ = randn(rng, k, m), randn(rng, k, m)
        @test ispoint(M, Θ; atol = 2e-14)
        @test Θ * Θ' ≈ I atol = 2e-14
        H = project_tangent(M, Θ, G)
        @test norm(H * Θ') < 2e-14
        @test project_tangent(M, Θ, H) ≈ H atol = 2e-14
        @test retract(M, Θ, Δ) * retract(M, Θ, Δ)' ≈ I atol = 2e-14
        @test retract(M, Θ, zero(Θ)) ≈ Θ atol = 2e-14
        # Reconstruct R independently: A = Q*R, Q = result'. This also tests
        # the positive diagonal on a generic step, beyond zero-step identity.
        R = retract(M, Θ, Δ) * (Θ + Δ)'
        @test all(diag(R) .> 0)
        @test norm(tril(R, -1)) < 2e-14
    end
    # The sampler uses a Gaussian m × k matrix and the same positive-R convention.
    M = Grassmann(2, 6)
    A = randn(Xoshiro(2402), 6, 2)
    Θ = rand(Xoshiro(2402), M)
    R = Θ * A
    @test all(diag(R) .> 0)
    @test norm(tril(R, -1)) < 2e-14
    @test Θ' * R ≈ A
    @test rand(Xoshiro(2402), M) == Θ
    @test ispoint(M, rand(M))
    @test !ispoint(M, zeros(2, 6))
    @test !ispoint(M, zeros(3, 6))
    @test !ispoint(M, fill(NaN, 2, 6))
    @test_throws ArgumentError ispoint(M, Θ; atol = -1)
    @test_throws ArgumentError ispoint(M, Θ; atol = Inf)
    @test_throws ArgumentError retract(M, Θ, -Θ)
    deficient = copy(Θ); deficient[2, :] .= deficient[1, :]
    @test_throws ArgumentError retract(M, deficient, zero(deficient))
    @test_throws ArgumentError retract(M, Θ, fill(Inf, 2, 6))
    @test_throws DimensionMismatch retract(M, Θ, zeros(6, 2))
    @test_throws DimensionMismatch project_tangent(M, Θ, zeros(6, 2))
    @test_throws DimensionMismatch project_tangent(M, Θ', Θ)
    for (k, m) in ((0, 3), (-1, 3), (4, 3), (1, 0), (1, -1))
        @test_throws ArgumentError Grassmann(k, m)
    end
    Θ32 = Float32.(Θ)
    @test eltype(retract(M, Θ32, zero(Θ32))) == Float32
    @test ispoint(M, retract(M, Θ32, zero(Θ32)); atol = 1e-6)
end

@testset "Retracted rules and eigenplane descent" begin
    rng = Xoshiro(2403)
    M = Grassmann(2, 6)
    # A fixed seeded symmetric S with known spectrum and gap λ₃ - λ₂ = 2.
    U = Matrix(qr(randn(rng, 6, 6)).Q)
    λ = [-2.0, -1.0, 1.0, 2.0, 3.0, 5.0]
    S = Symmetric(U * Diagonal(λ) * U')
    target = U[:, 1:2]'
    F(Θ) = tr(Θ * S * Θ')
    Θ0 = rand(rng, M)
    G = 2Θ0 * S
    grad = project_tangent(M, Θ0, G)
    @test norm(grad * Θ0') < 2e-14
    O = Matrix(qr(randn(rng, 2, 2)).Q)
    @test F(O * Θ0) ≈ F(Θ0)
    @test project_tangent(M, O * Θ0, O * G) ≈ O * grad
    # dF[-grad] = -‖grad‖² for the horizontal Frobenius metric.
    # At η=1e-6 the first-order decrease agrees within relative tolerance 2e-5.
    ηsmall = 1e-6
    decrease = F(Θ0) - F(retract(M, Θ0, -ηsmall * grad))
    @test decrease > 0
    @test decrease ≈ ηsmall * sum(abs2, grad) rtol = 2e-5

    @test_throws ArgumentError Retracted(Nesterov(0.04, 0.6), M)
    @test_throws ArgumentError Retracted(UnsupportedGrassmannRule(), M)
    # η=0.04 < 1/(2‖S‖₂)=0.1. Both rules are checked for 400 steps;
    # momentum uses Mycelium's s′=γ*s-η*grad with γ=0.6.
    η = 0.04
    @test η < 1 / (2opnorm(S))
    for inner in (GradientDescent(η), Momentum(η, 0.6))
        r = Retracted(inner, M)
        Θ, factor_Θ = copy(Θ0), copy(Θ0)
        state = inner isa Momentum ? zero(Θ) : nothing
        factor_state = deepcopy(state)
        of = OptimiserFactor(:θ, r)
        @test r isa AbstractUpdateRule
        @test rule_get(r, state, Θ) === Θ
        @test !islearnable(of)
        for _ in 1:400
            # OptimiserFactor's state layout and emission contract, exactly as
            # factors.jl uses them; optimiser_step is a separate training action.
            st = (p = factor_Θ, state = factor_state)
            bel, st′ = factor_message(of, :θ, nothing, (;), nothing, (;), st)
            @test bel isa DiracBelief
            @test bel.value === factor_Θ
            @test st′ === st
            G = 2Θ * S
            # Independent candidate/state oracle also checks the inner recurrence
            # and transport. A nonzero vertical component in G must be removed.
            P = I - Θ' * Θ
            velocity = inner isa Momentum ? inner.γ * state - η * (G * P) : -η * (G * P)
            expected_Θ = retract(M, Θ, velocity * P)
            state, Θ = rule_put(r, state, Θ, G)
            @test Θ ≈ expected_Θ atol = 2e-14
            factor_state, factor_Θ = optimiser_step(of, factor_state, factor_Θ, 2factor_Θ * S)
            @test Θ == factor_Θ
            @test ispoint(M, Θ; atol = 2e-14)
            if inner isa Momentum
                @test size(state) == size(Θ)
                @test norm(state * Θ') < 2e-14
                @test state ≈ velocity * (I - Θ' * Θ) atol = 2e-14
                @test state == factor_state
            else
                @test state === nothing
                @test factor_state === nothing
            end
        end
        # Principal angles are an independent row-space comparison (basis-free).
        largest_angle = acos(clamp(minimum(svdvals(Θ * target')), 0, 1))
        @test largest_angle < 1e-6
        @test F(Θ) ≈ sum(λ[1:2]) atol = 2e-12 rtol = 0
    end
    # Stateless GD passes even a caller-owned sentinel through unchanged.
    sentinel = Ref(:unchanged)
    state, _ = rule_put(Retracted(GradientDescent(η), M), sentinel, Θ0, G)
    @test state === sentinel
    # γ=0 recovers the same Grassmann step as gradient descent.
    _, gdΘ = rule_put(Retracted(GradientDescent(η), M), nothing, Θ0, G)
    _, mΘ = rule_put(Retracted(Momentum(η, 0.0), M), zero(Θ0), Θ0, G)
    @test gdΘ == mΘ
end

@testset "construction and design matrices" begin
    names, sizes = (:x, :y), (2, 3)
    expected = [1 0 1 0 1 0;
                0 1 0 1 0 1;
                1 1 0 0 0 0;
                0 0 1 1 0 0;
                0 0 0 0 1 1]
    @test independence(names, sizes) == expected
    @test hierarchical(names, sizes, ((:x,), (:y,))) == expected
    @test hierarchical(names, sizes, ((1,), (2,))) == expected
    @test independence((:a, :b, :c), (2, 3, 2)) ==
          hierarchical((:a, :b, :c), (2, 3, 2), ((:a,), (:b,), (:c,)))
    @test hierarchical(names, sizes, ((:x, :y),)) == Matrix{Int}(I, 6, 6)
    # Reversing a generator changes its row order, never the joint column order.
    @test hierarchical(names, sizes, ((:y, :x),)) ==
          Matrix{Int}(I, 6, 6)[:, [1, 4, 2, 5, 3, 6]]
    @test hierarchical(names, sizes, ((),)) == ones(Int, 1, 6)
    @test size(hierarchical(names, sizes, ())) == (0, 6)
    f = ToricFactor(names, sizes, expected)
    @test Tuple(channelname(c) for c in LenticulumCore.channels(f)) == names
    @test Tuple(channelspace(c) for c in LenticulumCore.channels(f)) == sizes
    @test islearnable(f)
    @test LuxCore.initialparameters(Xoshiro(1), f) == (η = zeros(5),)
    @test LuxCore.initialstates(Xoshiro(1), f) == NamedTuple()
    @test LuxCore.parameterlength(f) == 5
    @test LuxCore.statelength(f) == 0
    @test energyspace(f) == ScalarEnergySpace()
    for (ns, sz, A) in (((), (), zeros(Int, 0, 1)),
                          ((:x, :x), sizes, expected), (("x", :y), sizes, expected),
                          (names, (2,), expected), (names, (0, 3), expected),
                          (names, (-2, 3), expected), (names, (2.0, 3), expected),
                          (names, (true, 6), expected), (names, sizes, zeros(Int, 2, 5)),
                          (names, (typemax(Int), 2), expected))
        @test_throws ArgumentError ToricFactor(ns, sz, A)
    end
    for labels in (([1, 1], [1, 2, 3]), ([1], [1, 2, 3]), ([1, 2],),
                   [[1, 2], [1, 2, 3]], ((1, 2), [1, 2, 3]))
        @test_throws ArgumentError ToricFactor(names, sizes, expected; labels)
    end
    for gens in (((:z,),), ((3,),), ((0,),), ((:x, :x),), (:x,))
        @test_throws ArgumentError hierarchical(names, sizes, gens)
    end
    # Defensive copies: constructor arguments cannot silently alter the factor.
    A = copy(expected); labels = ([10, 20], [:a, :b, :c])
    g = ToricFactor(names, sizes, A; labels)
    A[1, 1] = 99; labels[1][1] = 99
    @test g.A == expected
    @test g.labels[1] == [10, 20]
end

@testset "pointwise energy and column order" begin
    names, sizes = (:x, :y), (2, 3)
    A = [2 -1 0 3 1 -2; 0 1 2 -1 3 1]
    η = [0.4, -0.7]; ps = (; η); st = (token = 7,)
    labels = ([20, 10], [:red, :green, :blue])
    logits = [sum(η[r] * A[r, k] for r in 1:2) for k in 1:6]
    for normalised in (false, true), custom in (false, true)
        f = ToricFactor(names, sizes, A; normalised, labels = custom ? labels : nothing)
        for j in 1:3, i in 1:2
            x = custom ? (labels[1][i], labels[2][j]) : (i, j)
            expected = -logits[i + 2*(j-1)] + (normalised ? log(sum(exp.(logits))) : 0)
            E, next = energy(f, x, nothing, nothing, ps, st)
            @test E ≈ expected
            @test next === st
            @test first(energy(f, x, ps, st)) ≈ expected
        end
        @test_throws ArgumentError energy(f, (1,), ps, st)
        @test_throws ArgumentError energy(f, (99, 1), ps, st)
        @test_throws ArgumentError energy(f, custom ? (20, :red) : (1, 1), (η = [0.2],), st)
        @test_throws ArgumentError energy(f, custom ? (20, :red) : (1, 1), (η = [Inf, 0.0],), st)
    end
    # Gauge directions can shift every logit without changing a normalised model.
    f = ToricFactor(names, sizes, vcat(ones(Int, 1, 6), A); normalised = true)
    @test first(energy(f, (2, 3), (η = [4.0; η],), st)) ≈
          first(energy(f, (2, 3), (η = [0.0; η],), st))
end

@testset "messages: all targets, evidence kinds, and polarities" begin
    η = [0.3, -0.2, 0.15, 0.4]; ps = (; η); st = (token = 8,)
    for sizes in ((2, 3), (2, 3, 2)), custom in (false, true), normalised in (false, true)
        n = length(sizes); names = Tuple((:a, :b, :c)[1:n]); A = design(sizes)
        labs = Tuple([[10i + j for j in 1:sizes[i]] for i in 1:n])
        f = ToricFactor(names, sizes, A; normalised, labels = custom ? labs : nothing)
        @test length(supported_polarities(f)) == n * 2^(n - 1)
        @test all(p -> supports_polarity(f, p), supported_polarities(f))
        @test !supports_polarity(f, Polarity(NamedTuple{names}(ntuple(_ -> Observed(), n))))
        @test !supports_polarity(f, Polarity(NamedTuple{names}(ntuple(_ -> Unobserved(), n))))
        for t in 1:n, kinds in Iterators.product(ntuple(_ -> 1:7, n-1)...)
            others = filter(!=(t), 1:n)
            beliefs = Dict{Symbol,Any}(); weights = [ones(s) for s in sizes]
            observed = Int[]
            for (i, kind) in zip(others, kinds)
                if kind == 1
                    # Omitted latent.
                elseif kind <= 5
                    beliefs[names[i]] = (nothing, TrivialBelief(), Latent(), missing)[kind-1]
                elseif kind == 6
                    beliefs[names[i]] = DiracBelief(custom ? labs[i][2] : 2)
                    weights[i] = [j == 2 ? 1.0 : 0.0 for j in 1:sizes[i]]
                    push!(observed, i)
                else
                    w = Float64.(1:sizes[i]); w ./= sum(w)
                    # Reverse labelled supports to test matching rather than positions.
                    beliefs[names[i]] = custom ? CategoricalBelief(reverse(w); labels = reverse(labs[i])) :
                                                                CategoricalBelief(w)
                    weights[i] = w; push!(observed, i)
                end
            end
            inputs = (; (name => beliefs[name] for name in names if haskey(beliefs, name))...)
            p = polarity(names, t, observed)
            site, next = factor_message(f, names[t], p, inputs, DiracBelief(:ignored), ps, st)
            @test probabilities(site) ≈ oracle_message(A, η, sizes, weights, t)
            @test site.labels == (custom ? labs[t] : nothing)
            @test next === st
            # Even an invalid target input must be excluded, as must any target prior.
            contaminated = merge(inputs, NamedTuple{(names[t],)}((DiracBelief(:ignored),)))
            again, _ = factor_message(f, names[t], p, contaminated, TrivialBelief(), ps, st)
            @test probabilities(again) ≈ probabilities(site)
        end
        for t in 1:n
            p = polarity(names, t, ())
            latent, _ = factor_message(f, names[t], p, (;), nothing, ps, st)
            inputs = NamedTuple{names}(map(i -> CategoricalBelief(ones(sizes[i])), 1:n) |> Tuple)
            uniform, _ = factor_message(f, names[t], p, inputs, nothing, ps, st)
            @test probabilities(uniform) ≈ probabilities(latent)
        end
    end
    f = ToricFactor((:a, :b), (2, 2), [1000 -1000 999 -999])
    site, _ = factor_message(f, :a, Polarity(a = Unobserved(), b = Latent()), (;), nothing,
                             (η = [1.0],), st)
    @test all(isfinite, probabilities(site))
    @test probabilities(site) ≈ [1.0, 0.0]
    # A huge common logit must preserve finite incoming evidence and log partition
    # differences, even when adding log(K) to that logit would round back to it.
    huge = ToricFactor((:a, :b), (2, 2), ones(Int, 1, 4); normalised = true)
    hugeps = (η = [1e300],)
    @test first(energy(huge, (1, 2), hugeps, st)) ≈ log(4)
    @test first(free_energy(huge, nothing, nothing, hugeps, st)) ≈ 0 atol = 1e-12
    evidence = (a = CategoricalBelief([0.2, 0.8]),)
    @test first(local_free_energy(huge, evidence, hugeps, st)) ≈
          log(2) + sum(p * log(p) for p in [0.2, 0.8])
    g = ToricFactor((:a, :b), (2, 2), zeros(Int, 0, 4))
    @test probabilities(first(factor_message(g, :a, Polarity(a = Unobserved(), b = Latent()),
                                            (;), nothing, (η = Float64[],), st))) ≈ [0.5, 0.5]
    p = Polarity(a = Unobserved(), b = Observed())
    @test_throws ArgumentError factor_message(g, :b, p, (;), nothing, (η = Float64[],), st)
    for bad in (DiracBelief(9), CategoricalBelief([1, 2, 3]),
                CategoricalBelief([1, 2]; labels = [1, 1]),
                CategoricalBelief([1, 2]; labels = [1, 9]), 2)
        @test_throws ArgumentError factor_message(g, :a, p, (b = bad,), nothing, (η = Float64[],), st)
    end
    # Label 1 must not silently select the first entry of a custom alphabet.
    g = ToricFactor((:a, :b), (2, 2), zeros(Int, 0, 4); labels = ([10, 20], [:x, :y]))
    @test_throws ArgumentError factor_message(g, :b, Polarity(a = Observed(), b = Unobserved()),
                                             (a = DiracBelief(1),), nothing, (η = Float64[],), st)
end

@testset "local and game free energy and entropy" begin
    names, sizes = (:a, :b, :c), (2, 3, 2); A = design(sizes)
    η = [0.3, -0.2, 0.15, 0.4]; ps = (; η); st = (token = 9,)
    for normalised in (false, true)
        f = ToricFactor(names, sizes, A; normalised)
        for a in (nothing, CategoricalBelief([0.4, 0.6]), DiracBelief(2)),
            b in (TrivialBelief(), CategoricalBelief([0, 0.3, 0.7]), DiracBelief(1)),
            c in (nothing, CategoricalBelief([0.8, 0.2]), DiracBelief(1))
            msgs = (; a, b, c)
            weights = map(zip(sizes, (a, b, c))) do (n, bel)
                bel isa CategoricalBelief ? probabilities(bel) : bel isa DiracBelief ?
                    [j == bel.value ? 1.0 : 0.0 for j in 1:n] : ones(n)
            end
            expected, H = oracle_statistics(A, η, sizes, weights; normalised)
            F, next = local_free_energy(f, msgs, ps, st)
            @test F ≈ expected atol = 1e-12
            @test next === st
            if normalised
                @test first(free_energy(f, (a = a, c = c), (b = b,), ps, st)) ≈ expected atol = 1e-12
                @test first(LenticulumCore.entropy(f, (a = a, c = c), (b = b,), ps, st)) ≈ H atol = 1e-12
            end
        end
        if normalised
            @test first(free_energy(f, nothing, nothing, ps, st)) ≈ 0 atol = 1e-12
            @test first(free_energy(f, CategoricalBelief([0.4, 0.6]), (b = 1, c = 2), ps, st)) ≈
                  first(local_free_energy(f, (a = CategoricalBelief([0.4, 0.6]),
                                             b = DiracBelief(1), c = DiracBelief(2)), ps, st))
            @test_throws ArgumentError free_energy(f, CategoricalBelief([1, 1]), (;), ps, st)
            @test_throws ArgumentError free_energy(f, (a = nothing,), (a = 1,), ps, st)
            @test_throws ArgumentError free_energy(f, nothing, (z = 1,), ps, st)
        else
            @test_throws ArgumentError free_energy(f, nothing, (;), ps, st)
            @test_throws ArgumentError LenticulumCore.entropy(f, nothing, (;), ps, st)
        end
    end
    f = ToricFactor(names, sizes, A; labels = ([:x, :y], [3, 2, 1], [true, false]), normalised = true)
    incoming = ([0.2, 0.8], [0.0, 1.0, 0.0], ones(2))
    expected, H = oracle_statistics(A, η, sizes, incoming; normalised = true)
    π = (a = CategoricalBelief([0.8, 0.2]; labels = [:y, :x]),)
    @test first(free_energy(f, π, (b = 2,), ps, st)) ≈ expected
    @test first(LenticulumCore.entropy(f, π, (b = 2,), ps, st)) ≈ H
end

@testset "tree BP and Bethe partition function" begin
    # Five variables, three toric factors, and two priors; optionally clamp c.
    # Incidence graph: a--f1--b--f2--d--f3--e, with c attached to f2.
    sizes = (2, 3, 2, 2, 3); names = (:a, :b, :c, :d, :e)
    subsets = ((1, 2), (2, 3, 4), (4, 5)); fnames = (:ab, :bcd, :de)
    As = map(s -> design(Tuple(sizes[i] for i in s)), subsets)
    ηs = ([0.3, -0.2, 0.15, 0.4], [-0.1, 0.5, 0.2, -0.3], [0.6, 0.1, -0.2, 0.25])
    pa, pe = [0.25, 0.75], [0.2, 0.5, 0.3]
    for normalised in (false, true), clamp in (false, true)
        b = GraphBuilder()
        for name in names; variable!(b, name); end
        for (fname, subset, A) in zip(fnames, subsets, As)
            ns = Tuple(names[i] for i in subset); sz = Tuple(sizes[i] for i in subset)
            factor!(b, fname, ToricFactor(ns, sz, A; normalised))
            for name in ns; connect!(b, fname, name, name; direction = Bidirectional()); end
        end
        factor!(b, :pa, PriorFactor(:a, CategoricalBelief(pa), x -> -log(pa[x])))
        factor!(b, :pe, PriorFactor(:e, CategoricalBelief(pe), x -> -log(pe[x])))
        # Priors need cavity messages too, to form their posterior factor beliefs for F.
        connect!(b, :pa, :a, :a; direction = Bidirectional())
        connect!(b, :pe, :e, :e; direction = Bidirectional())
        if clamp
            factor!(b, :obs, DataFactor(:c, 2))
            connect!(b, :obs, :c, :c; direction = Emitting())
        end
        g = build(b)
        @test istree(g)
        @test validate(g) === g
        ps = NamedTuple{fnames}(map(η -> (; η), ηs))
        st = LuxCore.initialstates(Xoshiro(42), g)
        xs = states(sizes)
        w = map(xs) do x
            value = pa[x[1]] * pe[x[5]] * (clamp && x[3] != 2 ? 0.0 : 1.0)
            for (subset, A, η) in zip(subsets, As, ηs)
                # Compute column strides explicitly, independently of factor indexing.
                col = 1; stride = 1
                for i in subset
                    col += (x[i] - 1) * stride; stride *= sizes[i]
                end
                logits = [sum(η[r] * A[r, k] for r in 1:4) for k in axes(A, 2)]
                value *= exp(logits[col]) / (normalised ? sum(exp.(logits)) : 1.0)
            end
            value
        end
        Z = sum(w)
        for root in (:a, :b, :e)
            marginals, report, next, store = infer!(g, tree_schedule(g; root), ps, st)
            @test report.converged
            for (i, name) in enumerate(names)
                expected = [sum(w[k] for (k, x) in enumerate(xs) if x[i] == j)/Z for j in 1:sizes[i]]
                bel = marginals[name]
                actual = bel isa DiracBelief ? [j == bel.value ? 1.0 : 0.0 for j in 1:sizes[i]] : probabilities(bel)
                @test actual ≈ expected atol = 1e-12
            end
            F, _ = scalar_free_energy(store, g, ps, next)
            @test F ≈ -log(Z) atol = 1e-12
            B, _ = bethe_free_energy(store, g, ps, next)
            @test sum(B[:factors][n] for n in keys(B[:factors])) +
                  sum(B[:variables][n] for n in keys(B[:variables])) ≈ -log(Z) atol = 1e-12
        end
    end
end

@testset "LP-014: a prior on an Emitting edge" begin
    # Marginals stay exact; free energy rejects a prior edge that cannot carry the cavity.
    pa = [0.25, 0.75]; A = design((2, 3)); η = [0.3, -0.2, 0.15, 0.4]
    w = [pa[i] * exp(sum(η .* A[:, i + 2*(j-1)])) for i in 1:2, j in 1:3]
    Z = sum(w)
    for direction in (Bidirectional(), Emitting())
        b = GraphBuilder()
        variable!(b, :a); variable!(b, :b)
        factor!(b, :ab, ToricFactor((:a, :b), (2, 3), A))
        connect!(b, :ab, :a, :a; direction = Bidirectional())
        connect!(b, :ab, :b, :b; direction = Bidirectional())
        factor!(b, :pa, PriorFactor(:a, CategoricalBelief(pa), x -> -log(pa[x])))
        connect!(b, :pa, :a, :a; direction)
        g = build(b)
        ps = (ab = (; η),); st = LuxCore.initialstates(Xoshiro(1), g)
        marginals, _, next, store = infer!(g, tree_schedule(g; root = :a), ps, st)
        @test probabilities(marginals[:a]) ≈ vec(sum(w; dims = 2)) / Z
        @test probabilities(marginals[:b]) ≈ vec(sum(w; dims = 1)) / Z
        if direction isa Bidirectional
            F, _ = scalar_free_energy(store, g, ps, next)
            @test F ≈ -log(Z)
        else
            @test_throws ArgumentError scalar_free_energy(store, g, ps, next)
            @test_throws ArgumentError bethe_free_energy(store, g, ps, next)
        end
    end
end

@testset "direct-message factor conformance" begin
    for sizes in ((2,), (2, 3), (2, 3, 2))
        n = length(sizes); names = Tuple((:a, :b, :c)[1:n]); A = design(sizes)
        labs = Tuple([Symbol("v$(i)_$(j)") for j in 1:sizes[i]] for i in 1:n)
        f = ToricFactor(names, sizes, A; normalised = true, labels = labs)
        ps = (η = [0.3, -0.2, 0.15, 0.4],); st = NamedTuple()
        for p in supported_polarities(f)
            t = findfirst(isequal(only(unobserved_channels(p))), names)
            inputs = NamedTuple{observed_channels(p)}(Tuple(DiracBelief(labs[i][1])
                     for i in 1:n if names[i] in observed_channels(p)))
            π = CategoricalBelief(Float64.(1:sizes[t]); labels = labs[t])
            prior = NamedTuple{(names[t],)}((π,)); msgs = merge(prior, inputs)
            incoming = [names[i] in observed_channels(p) ? [j == 1 ? 1.0 : 0.0 for j in 1:sizes[i]] :
                        i == t ? probabilities(π) : ones(sizes[i]) for i in 1:n]
            site = oracle_message(A, ps.η, sizes, incoming, t)
            posterior = site .* probabilities(π); posterior ./= sum(posterior)
            check_factor_conformance(f, p; π, inputs, ps, st,
                pointwise = (Tuple(first.(labs)), nothing, nothing),
                game_prior = prior, game_y = inputs, msgs, direct_message = true,
                expected_site = CategoricalBelief(site; labels = labs[t]),
                expected_posterior = CategoricalBelief(posterior; labels = labs[t]))
        end
    end
end
end

# References below use empirical margins, direct probability-domain likelihoods,
# and independently enumerated minors; none calls the fitting helpers.
@testset "Toric fitting and membership" begin
    @testset "independence closed forms and Newton/IPF agreement" begin
        for counts in ([2.0 7 3; 5 1 9], [2.0 7 3 11; 5 1 9 4; 8 6 2 5])
            sizes = size(counts)
            A = independence((:a, :b), sizes)
            f = ToricFactor((:a, :b), sizes, A; normalised = true)
            expected = vec(sum(counts; dims = 2) * sum(counts; dims = 1)) / sum(counts)^2
            fits = [fit_mle(f, method == :newton ? counts : vec(counts); method)
                    for method in (:newton, :ipf)]
            for fit in fits
                @test fit.converged
                @test fit.existence == :exists
                @test fit.residual <= 1e-10
                @test fit.p ≈ expected atol = 3e-10 rtol = 0
                @test sum(fit.p) ≈ 1
                @test inmodel(f, reshape(fit.p, sizes))
                @test fit.iterations > 0
                reproduced = [exp(-first(energy(f, Tuple(x), (; η = fit.η), NamedTuple())))
                              for x in CartesianIndices(sizes)]
                @test vec(reproduced) ≈ fit.p atol = 1e-14
            end
            @test fits[1].p ≈ fits[2].p atol = 3e-10
        end
    end

    @testset "decomposable chain closed form" begin
        names, sizes = (:a, :b, :c), (2, 3, 2)
        counts = reshape([3.0, 7, 2, 8, 5, 1, 9, 4, 6, 2, 3, 11], sizes)
        A = hierarchical(names, sizes, ((:a, :b), (:b, :c)))
        f = ToricFactor(names, sizes, A)
        q = counts / sum(counts)
        ab = dropdims(sum(q; dims = 3); dims = 3)
        bc = dropdims(sum(q; dims = 1); dims = 1)
        b = vec(sum(q; dims = (1, 3)))
        expected = [ab[i,j] * bc[j,k] / b[j] for i in 1:2, j in 1:3, k in 1:2]
        fits = [fit_mle(f, counts; method) for method in (:newton, :ipf)]
        for fit in fits
            @test fit.converged
            @test fit.existence == :exists
            @test fit.p ≈ vec(expected) atol = 5e-10 rtol = 0
        end
        @test fits[1].p ≈ fits[2].p atol = 5e-10
    end

    @testset "binary features without an intercept" begin
        A = [0 1 1 0 1 0; 0 0 1 1 0 1]
        @test rank(vcat(ones(Int,1,6), A)) == rank(A) + 1
        f = ToricFactor((:x,), (6,), A)
        counts = [7, 1, 3, 9, 2, 8]
        newton = fit_mle(f, counts)
        ipf = fit_mle(f, counts; method = :ipf)
        @test newton.converged && ipf.converged
        @test newton.p ≈ ipf.p atol = 3e-10
        @test ipf.existence == :exists
        @test inmodel(f, ipf.p)
        for method in (:newton, :ipf)
            stopped = fit_mle(f, counts; method, maxiter = 1)
            @test !stopped.converged
            @test stopped.iterations == 1
            @test stopped.residual > 1e-10
        end
    end

    @testset "integer design, gradient and complete gauge" begin
        rng = Xoshiro(403)
        A = rand(rng, -3:4, 4, 12) # No constant in the row space, independently checked.
        @test rank(vcat(ones(Int, 1, 12), A)) == rank(A) + 1
        counts = rand(rng, 1:15, 12)
        q = counts / sum(counts)
        f = ToricFactor((:x,), (12,), A; normalised = true)
        fit = fit_mle(f, counts)
        @test fit.converged
        @test fit.existence == :exists
        @test norm(A * fit.p - A * q, Inf) <= 1e-10
        @test norm(A * q - A * fit.p, Inf) <= 1e-10 # normalised log L gradient
        @test inmodel(f, fit.p)
        # Independent likelihood/gradient in probability space at moderate parameters.
        η = 0.2 * randn(rng, 4)
        likelihood(v) = dot(q, transpose(A) * v) - log(sum(exp.(transpose(A) * v)))
        p = exp.(transpose(A) * η); p /= sum(p)
        gradient = A * q - A * p
        h = 1e-5
        differences = [(likelihood(η + h * I(4)[:,r]) -
                        likelihood(η - h * I(4)[:,r])) / (2h) for r in 1:4]
        @test gradient ≈ differences atol = 2e-9 rtol = 0
        for augmented in (vcat(A, A[1:1,:]), vcat(ones(Int, 1, 12), A),
                          vcat(ones(Int, 1, 12), A, A[1:1,:]))
            g = ToricFactor((:x,), (12,), augmented; normalised = true)
            fitted = fit_mle(g, counts)
            @test fitted.converged
            @test fitted.p ≈ fit.p atol = 2e-10
            # Independent gauge projector checks the minimum-norm representative.
            B = transpose(Float64.(augmented))
            B = B .- sum(B; dims = 1) / 12
            @test pinv(B) * (B * fitted.η) ≈ fitted.η atol = 1e-12
            @test [exp(-first(energy(g, (i,), (; η = fitted.η), NamedTuple())))
                   for i in 1:12] ≈ fitted.p atol = 1e-14
        end
        @test_throws ArgumentError fit_mle(f, counts; method = :ipf)
        # Normalisation really must be augmented: kernel(A) alone is insufficient.
        g = ToricFactor((:x,), (2,), reshape([1, 2], 1, 2))
        @test inmodel(g, [0.2, 0.8])
        @test_throws ArgumentError binomial_residuals(g, [0.2, 0.8], [[2, -1]])
    end

    @testset "extended MLE and truthful statuses" begin
        counts = [0.0 0 0; 3 7 2]
        A = independence((:a, :b), (2, 3))
        f = ToricFactor((:a, :b), (2, 3), A; normalised = true)
        expected = vec(sum(counts; dims = 2) * sum(counts; dims = 1)) / sum(counts)^2
        for method in (:newton, :ipf)
            fit = fit_mle(f, counts; method, tol = 1e-10, maxiter = 2000)
            @test fit.converged
            @test fit.existence == :suspected_boundary
            @test fit.p ≈ expected atol = 3e-10 rtol = 0
            @test all(isfinite, fit.p)
            @test all(isfinite, fit.η)
            @test fit.residual <= 1e-10
            @test [exp(-first(energy(f, Tuple(x), (; η = fit.η), NamedTuple())))
                   for x in states((2,3))] ≈ fit.p atol = 1e-14
            # Reversing the rows also exercises target moment one in IPF.
            reversefit = fit_mle(f, reverse(counts; dims = 1); method)
            @test reversefit.converged
            @test reversefit.existence == :suspected_boundary
            @test reversefit.p ≈ vec(reverse(reshape(expected, 2, 3); dims = 1)) atol = 3e-10
            stopped = fit_mle(f, counts; method, maxiter = 0)
            @test !stopped.converged
            @test stopped.iterations == 0
            @test stopped.existence == :unknown
        end
        # Zero cells do not themselves prove boundary MLE: these margins are positive.
        sparsecounts = [0.0 2 3; 4 5 0]
        for method in (:newton, :ipf)
            fit = fit_mle(f, sparsecounts; method)
            @test fit.converged
            @test fit.existence == :unknown
            @test minimum(fit.p) > 0.01
            stopped = fit_mle(f, ones(2,3) .+ sparsecounts; method, maxiter = 0)
            @test !stopped.converged
            @test stopped.existence == :exists # Certificate is independent of optimisation.
        end
    end

    @testset "input validation and degenerate models" begin
        f = ToricFactor((:a, :b), (2, 3), independence((:a, :b), (2, 3)))
        for counts in ([-1, 2, 3, 4, 5, 6], zeros(2,3), ones(5), ones(3,2),
                       ones(2,3,1), [Inf, 1, 1, 1, 1, 1], fill(NaN, 6),
                       fill(1+im, 6), (1,2,3,4,5,6), 1)
            @test_throws ArgumentError fit_mle(f, counts)
        end
        for kwargs in ((method = :other,), (tol = 0,), (tol = NaN,),
                       (maxiter = -1,), (maxiter = 1.5,), (maxiter = true,))
            @test_throws ArgumentError fit_mle(f, ones(6); kwargs...)
        end
        @test fit_mle(f, fill(floatmax(Float64), 6)).converged
        for A in (zeros(Int, 0, 6), ones(Int, 2, 6)), method in (:newton, :ipf)
            g = ToricFactor((:a, :b), (2, 3), A)
            fit = fit_mle(g, collect(1:6); method)
            @test fit.converged
            @test fit.iterations == 0
            @test fit.p ≈ fill(1/6, 6)
            @test fit.η == zeros(size(A, 1))
            @test inmodel(g, fit.p)
            @test !inmodel(g, collect(1:6) / 21)
        end
        g = ToricFactor((:x,), (1,), zeros(Int, 0, 1))
        @test fit_mle(g, [2]).p == [1.0]
        @test inmodel(g, [1.0])
    end

    @testset "membership and supplied independence minors" begin
        rng = Xoshiro(704)
        sizes = (3,4)
        f = ToricFactor((:a, :b), sizes, independence((:a, :b), sizes))
        moves = Vector{Int}[]
        for i in 1:2, k in (i+1):3, j in 1:3, l in (j+1):4
            u = zeros(Int, sizes)
            u[i,j] += 1; u[k,l] += 1; u[i,l] -= 1; u[k,j] -= 1
            push!(moves, vec(u))
        end
        for _ in 1:12
            a = rand(rng, 3) .+ 0.2; b = rand(rng, 4) .+ 0.2
            inside = a * transpose(b); inside /= sum(inside)
            outside = rand(rng, sizes...) .+ 0.2; outside /= sum(outside)
            for (p, expected) in ((inside, true), (outside, false))
                residuals = binomial_residuals(f, p, moves)
                direct = [p[i,j]*p[k,l] - p[i,l]*p[k,j]
                          for i in 1:2 for k in (i+1):3 for j in 1:3 for l in (j+1):4]
                @test maximum(abs, residuals - direct) <= 2e-17
                @test inmodel(f, p) == expected
                @test all(abs.(residuals) .<= 1e-12) == inmodel(f, p)
            end
        end
        p = fill(1/12, 12)
        @test isempty(binomial_residuals(f, p, []))
        @test binomial_residuals(f, p, [zeros(Int, 12)]) == [0.0]
        boundary = zeros(sizes); boundary[1,:] .= 1/4
        @test all(iszero, binomial_residuals(f, boundary, moves))
        @test_throws ArgumentError inmodel(f, boundary)
        @test_throws ArgumentError binomial_residuals(f, p, [ones(Int, 12)])
        bad = zeros(Int, 12); bad[1] = 1; bad[2] = -1
        @test_throws ArgumentError binomial_residuals(f, p, [bad])
        @test_throws ArgumentError binomial_residuals(f, p, [zeros(Int, 11)])
        @test_throws ArgumentError binomial_residuals(f, p, [zeros(12)])
        # Exact integer checks must not wrap a nonzero statistic to zero.
        g = ToricFactor((:x,), (2,), reshape([typemin(Int), 0], 1, 2))
        @test_throws ArgumentError binomial_residuals(g, [0.5,0.5], [[2,-2]])
        for badp in (fill(-1/12, 12), ones(12), fill(NaN,12), ones(11), ones(4,3))
            @test_throws ArgumentError inmodel(f, badp)
            @test_throws ArgumentError binomial_residuals(f, badp, moves)
        end
        @test_throws ArgumentError inmodel(f, p; atol = -1)
        @test_throws ArgumentError inmodel(f, p; atol = Inf)
        # Check the documented 2-norm tolerance using an independent LS projection.
        q = collect(1.0:12); q /= sum(q)
        B = transpose(Float64.(vcat(ones(Int,1,12), f.A)))
        residual = norm(log.(q) - B * (pinv(B) * log.(q)))
        @test inmodel(f, q; atol = 1.01residual)
        @test !inmodel(f, q; atol = 0.99residual)
    end
end

@testset "Plücker coordinates and relations" begin
    @test plucker_pairs(4) == [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
    @test plucker([1,2], [3,5]) == [-1]
    @test plucker([1,2,3], [4,5,6]) == [-3,-6,-3]
    @test plucker([1,2,3,4], [5,6,7,8]) == [-4,-8,-12,-4,-8,-4]
    @test eltype(plucker([1,2], [3,5])) == Int
    @test plucker([1//2,2//3], [3//4,5//6]) == [-1//12]
    @test plucker(ComplexF64[1,im], ComplexF64[im,1]) == [2]
    @test eltype(plucker(Float32[1,2], Float32[3,5])) == Float32
    @test_throws ArgumentError plucker_pairs(1)
    @test_throws ArgumentError plucker([1], [2])
    @test_throws DimensionMismatch plucker([1,2], [3,4,5])
    @test_throws DimensionMismatch plucker_jacobian([1,2], [3,4,5])
    @test_throws ArgumentError plucker_relations(1)
    @test_throws DimensionMismatch plucker_residuals(4, zeros(5))
    @test isempty(plucker_relations(2)) && isempty(plucker_relations(3))
    @test isempty(plucker_residuals(3, zeros(3)))
    rng = Xoshiro(5101)
    for r in 2:6
        z, w = randn(rng,r), randn(rng,r)
        p = plucker(z,w)
        @test plucker(w,z) == -p
        a,b,c,d = randn(rng,4)
        @test plucker(a*z+b*w, c*z+d*w) ≈ (a*d-b*c)*p atol = 1e-13
        @test sum(abs2,p) ≈ sum(abs2,z)*sum(abs2,w)-dot(z,w)^2 atol = 1e-13
        # Rectangular A also checks distinct input/output exterior-power dimensions.
        A = randn(rng,r+1,r)
        # Independent minors: do not use plucker or its Jacobian to build Λ²A.
        exterior = [det(A[[i,j],[k,l]]) for (i,j) in plucker_pairs(r+1), (k,l) in plucker_pairs(r)]
        @test plucker(A*z,A*w) ≈ exterior*p atol = 2e-13
        v, h = vcat(z,w), 1e-5
        fd = hcat([(plucker((v+h*I(2r)[:,a])[1:r], (v+h*I(2r)[:,a])[r+1:2r]) -
                    plucker((v-h*I(2r)[:,a])[1:r], (v-h*I(2r)[:,a])[r+1:2r]))/(2h) for a in 1:2r]...)
        @test size(plucker_jacobian(z,w)) == (binomial(r,2),2r)
        @test plucker_jacobian(z,w) ≈ fd atol = 2e-10 rtol = 0
        relations = plucker_relations(r)
        @test length(relations) == binomial(r,4)
        @test [q.indices for q in relations] == [(i,j,k,l) for i in 1:r for j in i+1:r for k in j+1:r for l in k+1:r]
        if r >= 4
            for _ in 1:10
                @test maximum(abs, plucker_residuals(r,plucker(randn(rng,r),randn(rng,r)))) < 3e-14
            end
            @test norm(plucker_residuals(r,randn(rng,binomial(r,2)))) > 1e-5
        end
    end
    klein = only(plucker_relations(4))
    @test klein.indices == (1,2,3,4)
    @test klein.monomials == ((1,6),(2,5),(3,4))
    @test klein.coefficients == (1,-1,1)
    p = [2,3,5,7,11,13]
    @test plucker_residuals(4,p) == [p[1]*p[6]-p[2]*p[5]+p[3]*p[4]]
    @test plucker_residuals(4,[1,0,0,0,0,1]) == [1] # e₁∧e₂ + e₃∧e₄
end

# Independent moment oracle: expand E[p_a p_b] as four signed products of
# E[z_i z_k] E[w_j w_l], then subtract E[p_a]E[p_b]. No Jacobian is used.
function plucker_moment_oracle(μ,S,ν,T)
    pairs = [(i,j) for i in 1:length(μ) for j in i+1:length(μ)]
    means = [μ[i]*ν[j]-μ[j]*ν[i] for (i,j) in pairs]
    second_z, second_w = S+μ*μ', T+ν*ν'
    C = [sum(s*t*second_z[u,x]*second_w[v,y]
             for (s,u,v) in ((1,i,j),(-1,j,i)), (t,x,y) in ((1,k,l),(-1,l,k)))
         for (i,j) in pairs, (k,l) in pairs] - means*means'
    return means,C
end

@testset "PluckerFactor forward beliefs" begin
    f = PluckerFactor(4)
    ps, st = LuxCore.setup(Xoshiro(5102),f)
    @test ps == st == NamedTuple()
    @test LuxCore.parameterlength(f) == LuxCore.statelength(f) == 0
    @test !islearnable(f)
    @test isunidirectional(f)
    @test Tuple(channelname(c) for c in LenticulumCore.channels(f)) == (:z,:w,:p)
    @test Tuple(c.space for c in LenticulumCore.channels(f)) == (4,4,6)
    forward = only(supported_polarities(f))
    @test forward == Polarity(z=Observed(),w=Observed(),p=Unobserved())
    for zpol in (Observed(),Latent(),Unobserved()), wpol in (Observed(),Latent(),Unobserved()),
        ppol in (Observed(),Latent(),Unobserved())
        pol = Polarity(z=zpol,w=wpol,p=ppol)
        @test supports_polarity(f,pol) == (pol == forward)
    end
    @test !supports_polarity(f,Polarity(a=Observed(),b=Observed(),c=Unobserved()))
    z,w = [1.,2,3,4],[5.,6,7,8]
    d, next = factor_message(f,:p,forward,(z=DiracBelief(z),w=DiracBelief(w)),nothing,ps,st)
    @test d isa DiracBelief && d.value == [-4,-8,-12,-4,-8,-4]
    @test next === st
    @test first(pushforward(f,(DiracBelief(z),DiracBelief(w)),ps,st)).value == d.value
    # Both target prior and target input are excluded from an extrinsic forward message.
    @test first(factor_message(f,:p,forward,(z=DiracBelief(z),w=DiracBelief(w),p=DiracBelief(zeros(6))),
                              Gaussian(zeros(6),I(6)),ps,st)).value == d.value
    custom = PluckerFactor(4;names=(:a,:b,:c))
    @test first(factor_message(custom,:c,only(supported_polarities(custom)),
                              (a=DiracBelief(z),b=DiracBelief(w)),nothing,ps,st)).value == d.value
    @test first(energy(f,nothing,nothing,nothing,ps,st)) == 0
    @test first(local_free_energy(f,NamedTuple(),ps,st)) == 0
    # The common conformance checker requires game-level free_energy/entropy. Like
    # GeneratorFactor these are absent (LP-009); do not claim full conformance.
    @test !applicable(free_energy,f,TrivialBelief(),NamedTuple(),ps,st)
    @test !applicable(LenticulumCore.entropy,f,TrivialBelief(),NamedTuple(),ps,st)

    zs, ws = [z,2z,-z], hcat(w,-w,3w)
    weights = [1.,2,3]
    for (a,b) in ((SampleBelief(zs,weights),SampleBelief(ws,2weights)),
                  (SampleBelief(zs),SampleBelief(ws)),
                  (SampleBelief(zs),SampleBelief(ws,ones(3))),
                  (SampleBelief(zs,weights),DiracBelief(w)),
                  (DiracBelief(z),SampleBelief(ws,weights)))
        out,next = factor_message(f,:p,forward,(z=a,w=b),nothing,ps,st)
        @test out isa SampleBelief && length(out.samples) == 3
        @test next === st
        @test out.weights === (a isa SampleBelief ? a.weights : b.weights)
        for i in 1:3
            @test out.samples[i] == plucker(a isa SampleBelief ? zs[i] : z,
                                            b isa SampleBelief ? ws[:,i] : w)
        end
    end
    for (a,b) in ((SampleBelief(zs),SampleBelief(ws[:,1:2])),
                  (SampleBelief(zs,weights),SampleBelief(ws,[3.,2,1])),
                  (SampleBelief(zs,zeros(3)),DiracBelief(w)),
                  (SampleBelief(Vector{Float64}[]),DiracBelief(w)),
                  (TrivialBelief(),DiracBelief(w)),
                  (nothing,DiracBelief(w)),
                  (SampleBelief(zs),Gaussian(w,I(4))),
                  (DiracBelief(complex.(z)),DiracBelief(w)))
        @test_throws ArgumentError factor_message(f,:p,forward,(z=a,w=b),nothing,ps,st)
    end
    @test_throws DimensionMismatch factor_message(f,:p,forward,(z=DiracBelief(z[1:3]),w=DiracBelief(w)),nothing,ps,st)
    @test_throws ArgumentError factor_message(f,:z,nothing,NamedTuple(),nothing,ps,st)
    @test_throws ArgumentError factor_message(f,:bad,nothing,NamedTuple(),nothing,ps,st)
    @test_throws ArgumentError factor_message(f,:p,Polarity(z=Latent(),w=Observed(),p=Unobserved()),
                                            (z=DiracBelief(z),w=DiracBelief(w)),nothing,ps,st)
    @test_throws ArgumentError factor_message(f,:p,forward,(z=DiracBelief(z),),nothing,ps,st)
    for kwargs in ((names=(:a,:a,:p),),(names=(:a,:b),),(names=(1,2,3),),
                   (covariance_jitter=-1,),(covariance_jitter=Inf,),(covariance_jitter=NaN,))
        @test_throws ArgumentError PluckerFactor(4;kwargs...)
    end
    @test_throws ArgumentError PluckerFactor(1)
end

@testset "Plücker exact moments and Gaussian messages" begin
    rng = Xoshiro(5103)
    for r in 2:5
        μ,ν = randn(rng,r),randn(rng,r)
        A,B = randn(rng,r,r),randn(rng,r,r)
        S,T = A*A'+I, B*B'+I
        for (Sz,Tw) in ((S,T),(zero(S),T),(S,zero(T)),(zero(S),zero(T)))
            m = plucker_moments(μ,Sz,ν,Tw)
            expected_mean, expected_cov = plucker_moment_oracle(μ,Sz,ν,Tw)
            @test m.mean ≈ expected_mean
            @test m.cov ≈ expected_cov atol = 3e-13 rtol = 2e-14
            @test m.cov ≈ m.cov'
        end
        f = PluckerFactor(r)
        out,_ = factor_message(f,:p,only(supported_polarities(f)),
                               (z=Gaussian(μ,S),w=Gaussian(ν,T)),nothing,NamedTuple(),NamedTuple())
        expected_mean, expected_cov = plucker_moment_oracle(μ,S,ν,T)
        @test out isa GaussianBelief
        @test belief_mean(out) ≈ expected_mean
        @test belief_cov(out) ≈ expected_cov atol = 5e-12
    end
    μ,ν = [0.4,-0.7,0.2,1.1],[-0.3,0.8,1.2,-0.5]
    A = [1.0 0 0 0; 0.3 0.8 0 0; -0.2 0.1 0.7 0; 0.1 -0.2 0.3 0.9]
    B = [0.7 0 0 0; -0.1 0.9 0 0; 0.2 0.4 0.8 0; -0.3 0.1 -0.2 0.6]
    S,T = A*A',B*B'
    exact = plucker_moments(μ,S,ν,T)
    # Monte Carlo seed 5104, 200000 independent pairs. Normalised mean error
    # < 0.015; covariance Frobenius relative error < 0.025 (no elementwise ratios).
    rng = Xoshiro(5104); n = 200_000
    Z,W = μ .+ A*randn(rng,4,n), ν .+ B*randn(rng,4,n)
    P = reduce(vcat,[permutedims(Z[i,:].*W[j,:]-Z[j,:].*W[i,:]) for (i,j) in plucker_pairs(4)])
    empirical_mean = vec(sum(P;dims=2))/n
    centered = P .- empirical_mean
    empirical_cov = centered*centered'/(n-1)
    mean_error = norm(empirical_mean-exact.mean)/sqrt(tr(exact.cov))
    cov_error = norm(empirical_cov-exact.cov)/norm(exact.cov)
    @test mean_error < 0.015
    @test cov_error < 0.025
    @info "Plücker Monte Carlo (seed 5104, n=200000)" mean_error cov_error
    # Centred isotropic inputs: output has 2I covariance even though it lies on
    # the nonlinear Plücker variety. A nonlinear constraint need not lower covariance rank.
    @test plucker_moments(zeros(4),I(4),zeros(4),I(4)).cov == 2I(6)

    # Dirac × Gaussian is an exact Gaussian law, but has rank r-1 for nonzero z.
    f = PluckerFactor(4); pol = only(supported_polarities(f)); empty = NamedTuple()
    for inputs in ((z=DiracBelief(μ),w=Gaussian(ν,T)), (z=Gaussian(μ,S),w=DiracBelief(ν)),
                   (z=DiracBelief(zeros(4)),w=Gaussian(ν,T)))
        @test_throws ArgumentError factor_message(f,:p,pol,inputs,nothing,empty,empty)
        reg = PluckerFactor(4;covariance_jitter=0.01)
        out,_ = factor_message(reg,:p,pol,inputs,nothing,empty,empty)
        Sz = inputs.z isa DiracBelief ? zero(S) : S
        Tw = inputs.w isa DiracBelief ? zero(T) : T
        mz = inputs.z isa DiracBelief ? inputs.z.value : μ
        m = plucker_moments(mz,Sz,ν,Tw)
        @test belief_mean(out) ≈ m.mean atol = 2e-13
        @test belief_cov(out) ≈ m.cov+0.01I atol = 3e-12
    end
    @test rank(plucker_moments(μ,zero(S),ν,T).cov;atol=1e-12) == 3
    for inputs in ((z=DiracBelief([1.,2]),w=Gaussian([3.,4],I(2))),
                   (z=Gaussian([3.,4],I(2)),w=DiracBelief([1.,2])))
        out,_ = factor_message(PluckerFactor(2),:p,nothing,inputs,nothing,empty,empty)
        @test belief_cov(out) ≈ fill(5.,1,1)
        @test belief_mean(out) ≈ (inputs.z isa DiracBelief ? [-2.] : [2.])
    end
    @test_throws ArgumentError factor_message(f,:p,pol,(z=uninformative(4),w=Gaussian(ν,T)),nothing,empty,empty)
    @test_throws DimensionMismatch plucker_moments(μ,ones(3,3),ν,T)

    # Mycelium DAG: independent Gaussian priors, absorbing input edges, emitting output.
    builder = GraphBuilder()
    for name in (:z,:w,:p); variable!(builder,name); end
    factor!(builder,:pz,PriorFactor(:z,Gaussian(μ,S),x->0.0))
    factor!(builder,:pw,PriorFactor(:w,Gaussian(ν,T),x->0.0))
    factor!(builder,:wedge,f)
    connect!(builder,:pz,:z,:z;direction=Emitting())
    connect!(builder,:pw,:w,:w;direction=Emitting())
    connect!(builder,:wedge,:z,:z;direction=Absorbing())
    connect!(builder,:wedge,:w,:w;direction=Absorbing())
    connect!(builder,:wedge,:p,:p;direction=Emitting())
    graph = build(builder)
    ps,st = LuxCore.setup(Xoshiro(5105),graph)
    marginals,_,_,store = infer!(graph,forward_backward_schedule(graph),ps,st)
    @test marginals[:p] isa GaussianBelief
    @test belief_mean(marginals[:p]) ≈ exact.mean
    @test belief_cov(marginals[:p]) ≈ exact.cov
    # Verify the stored outgoing site itself, independently of marginal pooling.
    edge = only(findall(e->e.channel == :p,graph.edges))
    @test belief_cov(store.to_variable[edge].belief) ≈ exact.cov
end

include("polynomial.jl")

include("homotopy.jl")
