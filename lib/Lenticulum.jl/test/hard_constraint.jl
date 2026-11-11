# HardConstraintFactor against two oracles (hard_constraint.md):
#
# 1. With a positive definite noise its messages are LinearConstraintFactor's, on random inputs
#    (points, proper and improper Gaussians, no information).
# 2. Without noise, the posterior of Gaussian priors under Σᵢ Aᵢ xᵢ = c is the Gaussian conditioned
#    on a linear equality: μ − ΣAᵀ(AΣAᵀ)⁺(Aμ − c), Σ − ΣAᵀ(AΣAᵀ)⁺AΣ. Checked factor by factor and
#    through `infer!` on a graph (a tree schedule, and damped flooding).

module HardConstraintOracle
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, LuxCore

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))
same(x::ExtendedGaussian, y::ExtendedGaussian; atol = 1e-8) =
    isapprox(x.μ, y.μ; atol) && isapprox(x.Σ, y.Σ; atol) && isapprox(x.D * x.D', y.D * y.D'; atol)

function random_input(rng, n, kind)
    kind === :dirac && return DiracBelief(randn(rng, n))
    kind === :proper && return Gaussian(randn(rng, n), spd(rng, n))
    kind === :trivial && return TrivialBelief()
    V = Matrix(qr(randn(rng, n, n)).Q)[:, 1:max(1, n - 1)]      # an improper site: flat along one direction
    Λ = V * spd(rng, size(V, 2)) * V'
    return GaussianBelief(Λ * randn(rng, n), (Λ + Λ') / 2, true)
end

# the dense conditional of independent Gaussian priors under Σᵢ Aᵢ xᵢ = c
function conditioned(μs, Σs, As, c)
    μ = reduce(vcat, μs)
    n = length(μ); Σ = zeros(n, n); o = 0
    for S in Σs
        k = size(S, 1); Σ[o+1:o+k, o+1:o+k] = S; o += k
    end
    A = reduce(hcat, As)
    K = Σ * A' * pinv(A * Σ * A')
    return μ - K * (A * μ - c), Σ - K * A * Σ
end
end # module

@testset "HardConstraintFactor with noise: LinearConstraintFactor's messages" begin
    HO = HardConstraintOracle
    rng = Xoshiro(41)
    for trial in 1:60
        k = rand(rng, 2:3)
        names = (:a, :b, :c)[1:k]
        dims = NamedTuple{names}(Tuple(rand(rng, 1:3) for _ in 1:k))
        m = rand(rng, 1:3)
        Q = HO.spd(rng, m)
        hard = HardConstraintFactor(dims, m; noise = Q)
        soft = LinearConstraintFactor(dims, m; noise = Q)
        ps = LuxCore.initialparameters(Xoshiro(trial), soft)
        for t in names
            kinds = Dict(n => rand(rng, (:dirac, :proper, :improper, :trivial)) for n in names if n !== t)
            inputs = NamedTuple{Tuple(n for n in names if n !== t)}(Tuple(HO.random_input(rng, dims[n], kinds[n]) for n in names if n !== t))
            ms, _ = Mycelium.factor_message(soft, t, nothing, inputs, nothing, ps, NamedTuple())
            mh, _ = Mycelium.factor_message(hard, t, nothing, inputs, nothing, ps, NamedTuple())
            @test HO.same(extended(ms, dims[t]), mh; atol = 1e-7)
        end
    end
end

@testset "HardConstraintFactor without noise: the Gaussian conditioned on the equality" begin
    HO = HardConstraintOracle
    rng = Xoshiro(42)
    for trial in 1:60
        k = rand(rng, 2:3)
        names = (:a, :b, :c)[1:k]
        dims = NamedTuple{names}(Tuple(rand(rng, 1:3) for _ in 1:k))
        m = rand(rng, 1:(sum(values(dims)) - 1))            # fewer equations than unknowns
        f = HardConstraintFactor(dims, m)
        ps = LuxCore.initialparameters(Xoshiro(100 + trial), f)
        μs = [randn(rng, dims[n]) for n in names]; Σs = [HO.spd(rng, dims[n]) for n in names]
        μp, Σp = HO.conditioned(μs, Σs, [ps.A[n] for n in names], ps.c)
        off = cumsum([0; collect(values(dims))])
        for (j, t) in enumerate(names)
            inputs = NamedTuple{Tuple(n for n in names if n !== t)}(Tuple(Gaussian(μs[i], Σs[i]) for (i, n) in enumerate(names) if n !== t))
            msg, _ = Mycelium.factor_message(f, t, nothing, inputs, nothing, ps, NamedTuple())
            post = Mycelium.combine(Gaussian(μs[j], Σs[j]), msg)
            r = (off[j] + 1):off[j + 1]
            @test size(post.D, 2) == 0
            @test post.μ ≈ μp[r] atol = 1e-8
            @test post.Σ ≈ Σp[r, r] atol = 1e-8
        end
    end
end

@testset "HardConstraintFactor in a graph: infer! against the conditioned Gaussian" begin
    HO = HardConstraintOracle
    rng = Xoshiro(43)
    for trial in 1:15
        dims = (a = rand(rng, 1:2), b = rand(rng, 1:3), c = rand(rng, 1:2))
        m = rand(rng, 1:(sum(values(dims)) - 1))
        gb = GraphBuilder()
        μs = Dict{Symbol,Vector{Float64}}(); Σs = Dict{Symbol,Matrix{Float64}}()
        for n in keys(dims)
            variable!(gb, n, dims[n])
            μs[n] = randn(rng, dims[n]); Σs[n] = HO.spd(rng, dims[n])
            p = Symbol(:prior_, n)
            factor!(gb, p, GaussianPrior(:x, μs[n], Σs[n])); connect!(gb, p, :x, n)
        end
        factor!(gb, :rel, HardConstraintFactor(dims, m))
        for n in keys(dims); connect!(gb, :rel, n, n); end
        g = validate(build(gb))
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        μp, Σp = HO.conditioned([μs[n] for n in keys(dims)], [Σs[n] for n in keys(dims)], [ps.rel.A[n] for n in keys(dims)], ps.rel.c)
        off = cumsum([0; collect(values(dims))])
        # tol 1e-8: the residual is absolute, and an ill-conditioned A leaves a rounding floor near 4e-10
        # (seed 43, trial 14: 3.8e-10), the known limit of an absolute bound (LP-024)
        for (sched, kw) in ((tree_schedule(g), (;)), (flooding_schedule(g), (damping = 0.5, maxsweeps = 500, tol = 1e-8)))
            marg, rep, _ = infer!(g, sched, ps, st; kw...)
            @test rep.converged
            for (j, n) in enumerate(keys(dims))
                r = (off[j] + 1):off[j + 1]
                b = marg[n]
                @test b isa ExtendedGaussian && size(b.D, 2) == 0
                @test b.μ ≈ μp[r] atol = 1e-7
                @test b.Σ ≈ Σp[r, r] atol = 1e-7
            end
        end
    end
end

@testset "HardConstraintFactor: refusals and the free energy" begin
    f = HardConstraintFactor((x = 2,), 1)
    @test_throws ArgumentError HardConstraintFactor((x = 2,), 1; noise = -1.0)
    @test_throws ArgumentError Mycelium.factor_message(HardConstraintFactor((x = 1, y = 1), 1), :y, nothing,
        (x = LenticulumCore.MixtureBelief([Gaussian([0.0], fill(1.0, 1, 1)), Gaussian([1.0], fill(1.0, 1, 1))], [0.5, 0.5]),), nothing,
        (A = (x = ones(1, 1), y = ones(1, 1)), c = [0.0]), NamedTuple())
    @test_throws ArgumentError Mycelium.local_free_energy(f, NamedTuple(), (A = (x = ones(1, 2),), c = [0.0]), NamedTuple())
    @test !LenticulumCore.islearnable(f)
end

@testset "HardConstraintFactor: the factor contract (conformance, no free energy)" begin
    isdefined(@__MODULE__, :check_factor_conformance) ||
        include(joinpath(@__DIR__, "../../LenticulumCore.jl/test/support/factor_conformance.jl"))
    f = HardConstraintFactor((x = 1, y = 1), 1)
    ps = (A = (x = ones(1, 1), y = ones(1, 1)), c = [1.0])
    p = LenticulumCore.Polarity((x = LenticulumCore.Observed(), y = LenticulumCore.Unobserved()))
    # x + y = 1 with x clamped at 0.25: y = 0.75 exactly, whatever the prior on y
    check_factor_conformance(f, p; inputs = (x = DiracBelief([0.25]),), π = Gaussian([0.0], fill(1.0, 1, 1)),
        ps, st = NamedTuple(), direct_message = true,
        expected_site = ExtendedGaussian([0.75], zeros(1, 1)), expected_posterior = ExtendedGaussian([0.75], zeros(1, 1)),
        pointwise_gap = "an exact relation has no pointwise energy",
        game_gap = "an exact relation has no game free energy",
        free_energy_gap = "an exact relation has no free energy in closed form")
end
