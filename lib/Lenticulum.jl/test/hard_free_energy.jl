# The free energy of graphs with hard constraints (hard_constraint.md, "Free energy").
#
# Oracles:
# 1. On a tree the Bethe free energy is −log Z. With a linear-Gaussian model z ~ N(μ, Σ) and the
#    relation C z = c imposed exactly, Z = ∫ N(z; μ, Σ) δ(Cz − c) dz = N(c; Cμ, CΣCᵀ), the density
#    of Cz at c. Checked on random trees where the relation fixes some variables in some
#    directions or entirely, so that hard messages reach GaussianFactor, GaussianPrior and the
#    variable entropies.
# 2. Factor by factor, HardConstraintFactor's term without noise is the limit of the noisy
#    relation's term with noise εI as ε → 0 (the divergent parts cancel), computed from its
#    formula in 512-bit BigFloat at ε = 1e-30. In Float64 (LinearConstraintFactor) the limit is
#    out of reach when the equations are ill-conditioned: the error decays like ε/σ_min², and at
#    ε = 1e-8 one random instance was off by 3.7 (and by 338 at 1e-6).

module HardFreeEnergyOracle
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, LuxCore

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# x ~ N(μ₀, Σ₀); u = A₁x + b₁ + ε₁; w = A₂u + b₂ + ε₂; v ~ N(μᵥ, Σᵥ); Cᵤu + Cᵥv = c exactly
function model(rng, n, nu, nv, nw, m)
    gb = GraphBuilder()
    variable!(gb, :x, n); variable!(gb, :u, nu); variable!(gb, :v, nv); variable!(gb, :w, nw)
    μ0, Σ0, μv, Σv = randn(rng, n), spd(rng, n), randn(rng, nv), spd(rng, nv)
    Q1, Q2 = spd(rng, nu), spd(rng, nw)
    factor!(gb, :px, GaussianPrior(:x, μ0, Σ0)); connect!(gb, :px, :x, :x)
    factor!(gb, :pv, GaussianPrior(:x, μv, Σv)); connect!(gb, :pv, :x, :v)
    factor!(gb, :g1, GaussianFactor(n => nu; noise = Q1)); connect!(gb, :g1, :x, :x); connect!(gb, :g1, :y, :u)
    factor!(gb, :g2, GaussianFactor(nu => nw; noise = Q2)); connect!(gb, :g2, :x, :u); connect!(gb, :g2, :y, :w)
    factor!(gb, :rel, HardConstraintFactor((u = nu, v = nv), m))
    connect!(gb, :rel, :u, :u); connect!(gb, :rel, :v, :v)
    g = validate(build(gb))
    return g, (; μ0, Σ0, μv, Σv, Q1, Q2)
end

# the dense joint of (x, u, v, w), the conditioned moments and −log Z
function dense(p, ps, n, nu, nv, nw, m)
    A1, b1, A2, b2 = ps.g1.A, ps.g1.b, ps.g2.A, ps.g2.b
    N = n + nu + nv + nw
    ix, iu, iv, iw = 1:n, (n + 1):(n + nu), (n + nu + 1):(n + nu + nv), (n + nu + nv + 1):N
    # z = L ξ + o with independent ξ = (x, ε₁, v, ε₂)
    L = zeros(N, N); o = zeros(N); S = zeros(N, N)
    L[ix, ix] = I(n); L[iu, ix] = A1; L[iu, iu] = I(nu); L[iv, iv] = I(nv)
    L[iw, ix] = A2 * A1; L[iw, iu] = A2; L[iw, iw] = I(nw)
    o[iu] = b1; o[iv] .= 0; o[iw] = A2 * b1 + b2
    S[ix, ix] = p.Σ0; S[iu, iu] = p.Q1; S[iv, iv] = p.Σv; S[iw, iw] = p.Q2
    μξ = zeros(N); μξ[ix] = p.μ0; μξ[iv] = p.μv
    μ = L * μξ + o; Σ = L * S * L'
    C = zeros(m, N); C[:, iu] = ps.rel.A.u; C[:, iv] = ps.rel.A.v
    Sc = Symmetric(C * Σ * C')
    d = ps.rel.c - C * μ
    nlogZ = (d' * (Sc \ d) + logdet(2π * Sc)) / 2
    K = Σ * C' / Sc
    return μ - K * (C * μ - ps.rel.c), Σ - K * C * Σ, nlogZ, (x = ix, u = iu, v = iv, w = iw)
end

function random_eg(rng, n; kf = rand(rng, 0:n), kh = rand(rng, 0:(n - kf)))
    B = Matrix(qr(randn(rng, n, n)).Q)
    V = B[:, (kf + kh + 1):n]
    Σ = V * spd(rng, size(V, 2)) * V'
    return ExtendedGaussian(randn(rng, n), (Σ + Σ') / 2, B[:, 1:kf])
end
end # module

@testset "HardConstraintFactor in a tree: the Bethe free energy is −log Z" begin
    HF = HardFreeEnergyOracle
    rng = Xoshiro(61)
    nfixed = 0
    for trial in 1:40
        n, nu, nv, nw = rand(rng, 1:3), rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:2)
        m = rand(rng, 1:(nu + nv))                    # up to fixing u and v entirely
        g, p = HF.model(rng, n, nu, nv, nw, m)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        μp, Σp, nlogZ, ix = HF.dense(p, ps, n, nu, nv, nw, m)
        marg, rep, _, s = infer!(g, tree_schedule(g), ps, st)
        @test rep.converged
        for v in (:x, :u, :v, :w)
            e = extended(marg[v], length(ix[v]))
            @test size(e.D, 2) == 0
            @test e.μ ≈ μp[ix[v]] atol = 1e-7 rtol = 1e-9
            @test e.Σ ≈ Σp[ix[v], ix[v]] atol = 1e-7 rtol = 1e-9
        end
        nfixed += size(hard_directions(extended(marg[:u], nu)), 2) > 0
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, st))
        @test F ≈ nlogZ atol = 1e-7 * (1 + abs(nlogZ))
    end
    @test nfixed >= 20                                 # u had hard directions in 25 of 40 (seed 61)
end

@testset "HardConstraintFactor's term is the limit of the noisy relation's as the noise vanishes" begin
    HF = HardFreeEnergyOracle
    rng = Xoshiro(62)
    errs = Float64[]
    for trial in 1:80
        k = rand(rng, 2:3)
        names = (:a, :b, :c)[1:k]
        dims = NamedTuple{names}(Tuple(rand(rng, 1:3) for _ in 1:k))
        # inputs: proper Gaussians, points and extended Gaussians with hard but no flat directions
        msgs = NamedTuple{names}(Tuple(begin
            kind = rand(rng, (:proper, :dirac, :hard))
            kind === :proper ? Gaussian(randn(rng, dims[n]), HF.spd(rng, dims[n])) :
            kind === :dirac ? DiracBelief(randn(rng, dims[n])) : HF.random_eg(rng, dims[n]; kf = 0)
        end for n in names))
        # the reduced relation G s = c on the inputs' free coordinates, and their precisions
        free = sum(n -> msgs[n] isa DiracBelief ? 0 :
                        msgs[n] isa ExtendedGaussian ? dims[n] - size(hard_directions(msgs[n]), 2) : dims[n], names)
        free == 0 && continue
        m = rand(rng, 1:free)                          # full row rank on the free coordinates
        hard = HardConstraintFactor(dims, m)
        ps = LuxCore.initialparameters(Xoshiro(trial), hard)
        F0 = LenticulumCore.scalarise(LenticulumCore.scalarisation(hard), first(Mycelium.local_free_energy(hard, msgs, ps, NamedTuple())))
        G, c, canon = Lenticulum._reduced_relation(ps.A, ps.c, msgs, dims, names)
        Fε = setprecision(BigFloat, 512) do
            ε = big"1e-30"
            Gb, cb = BigFloat.(G), BigFloat.(c)
            Λs = BigFloat.(Lenticulum._blockdiag(map(last, canon))); ηs = BigFloat.(reduce(vcat, map(first, canon)))
            Λ = Symmetric(Gb' * Gb / ε + Λs); η = Gb' * cb / ε + ηs
            Σ = inv(Λ); mb = Σ * η
            r = Gb * mb - cb
            fit = (r' * r + tr(Gb * Σ * Gb')) / (2ε)
            complexity = m * log(2 * BigFloat(π) * ε) / 2
            H = size(Gb, 2) * (1 + log(2 * BigFloat(π))) / 2 - logdet(cholesky(Λ)) / 2
            Float64(fit + complexity - H)
        end
        push!(errs, abs(Fε - F0) / (1 + abs(F0)))
    end
    @test length(errs) > 60
    @test maximum(errs) < 1e-9
end

@testset "HardConstraintFactor's free energy: refusals" begin
    f = HardConstraintFactor((x = 1, y = 1), 2)
    ps = (A = (x = [1.0; 1.0;;], y = [1.0; -1.0;;]), c = [0.0, 0.0])
    # two equations on one free coordinate (y clamped): rank 1 < 2, an infinite free energy
    @test_throws ArgumentError Mycelium.local_free_energy(f, (x = Gaussian([0.0], fill(1.0, 1, 1)), y = DiracBelief([0.5])), ps, NamedTuple())
    # a factor belief flat along the solution set
    f1 = HardConstraintFactor((x = 1, y = 1), 1)
    ps1 = (A = (x = ones(1, 1), y = ones(1, 1)), c = [1.0])
    @test_throws ArgumentError Mycelium.local_free_energy(f1, (x = TrivialBelief(), y = TrivialBelief()), ps1, NamedTuple())
    # a singular nonzero noise
    f2 = HardConstraintFactor((x = 2,), 2; noise = [1.0 0; 0 0])
    @test_throws ArgumentError Mycelium.local_free_energy(f2, (x = Gaussian(zeros(2), Matrix(1.0I, 2, 2)),), (A = (x = Matrix(1.0I, 2, 2),), c = zeros(2)), NamedTuple())
    # with a positive definite noise: LinearConstraintFactor's free energy
    Q = [1.0 0.2; 0.2 0.5]
    f3 = HardConstraintFactor((x = 2, y = 1), 2; noise = Q)
    l3 = LinearConstraintFactor((x = 2, y = 1), 2; noise = Q)
    ps3 = LuxCore.initialparameters(Xoshiro(3), l3)
    msgs = (x = Gaussian([0.3, -0.2], [2.0 0.1; 0.1 1.0]), y = Gaussian([1.0], fill(0.5, 1, 1)))
    @test LenticulumCore.scalarise(LenticulumCore.scalarisation(f3), first(Mycelium.local_free_energy(f3, msgs, ps3, NamedTuple()))) ≈
          LenticulumCore.scalarise(LenticulumCore.scalarisation(l3), first(Mycelium.local_free_energy(l3, msgs, ps3, NamedTuple())))
end
