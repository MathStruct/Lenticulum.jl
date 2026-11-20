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
function model(rng, n, nu, nv, nw, m; noise = 0)
    gb = GraphBuilder()
    variable!(gb, :x, n); variable!(gb, :u, nu); variable!(gb, :v, nv); variable!(gb, :w, nw)
    μ0, Σ0, μv, Σv = randn(rng, n), spd(rng, n), randn(rng, nv), spd(rng, nv)
    Q1, Q2 = spd(rng, nu), spd(rng, nw)
    factor!(gb, :px, GaussianPrior(:x, μ0, Σ0)); connect!(gb, :px, :x, :x)
    factor!(gb, :pv, GaussianPrior(:x, μv, Σv)); connect!(gb, :pv, :x, :v)
    factor!(gb, :g1, GaussianFactor(n => nu; noise = Q1)); connect!(gb, :g1, :x, :x); connect!(gb, :g1, :y, :u)
    factor!(gb, :g2, GaussianFactor(nu => nw; noise = Q2)); connect!(gb, :g2, :x, :u); connect!(gb, :g2, :y, :w)
    factor!(gb, :rel, HardConstraintFactor((u = nu, v = nv), m; noise))
    connect!(gb, :rel, :u, :u); connect!(gb, :rel, :v, :v)
    g = validate(build(gb))
    return g, (; μ0, Σ0, μv, Σv, Q1, Q2)
end

# the dense joint of (x, u, v, w), the conditioned moments and −log Z
function dense(p, ps, n, nu, nv, nw, m; R = zeros(m, m))
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
    Sc = Symmetric(C * Σ * C' + R)
    d = ps.rel.c - C * μ
    nlogZ = (d' * (Sc \ d) + logdet(2π * Sc)) / 2
    K = Σ * C' / Sc
    return μ - K * (C * μ - ps.rel.c), Σ - K * C * Σ, nlogZ, (x = ix, u = iu, v = iv, w = iw)
end

# a random positive semidefinite m × m matrix of rank r
psd(rng, m, r) = (B = randn(rng, m, r); Matrix(Symmetric(B * B')))

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
    # a singular nonzero noise whose exact part is redundant (y clamped, x₁ = 0 twice over): refused
    f2 = HardConstraintFactor((x = 1, y = 1), 3; noise = [1.0 0 0; 0 0 0; 0 0 0])
    ps2 = (A = (x = [1.0; 1.0; 2.0;;], y = [0.0; 0.0; 0.0;;]), c = zeros(3))
    @test_throws ArgumentError Mycelium.local_free_energy(f2, (x = Gaussian([0.0], fill(1.0, 1, 1)), y = DiracBelief([0.5])), ps2, NamedTuple())
    # with a positive definite noise: LinearConstraintFactor's free energy
    Q = [1.0 0.2; 0.2 0.5]
    f3 = HardConstraintFactor((x = 2, y = 1), 2; noise = Q)
    l3 = LinearConstraintFactor((x = 2, y = 1), 2; noise = Q)
    ps3 = LuxCore.initialparameters(Xoshiro(3), l3)
    msgs = (x = Gaussian([0.3, -0.2], [2.0 0.1; 0.1 1.0]), y = Gaussian([1.0], fill(0.5, 1, 1)))
    @test LenticulumCore.scalarise(LenticulumCore.scalarisation(f3), first(Mycelium.local_free_energy(f3, msgs, ps3, NamedTuple()))) ≈
          LenticulumCore.scalarise(LenticulumCore.scalarisation(l3), first(Mycelium.local_free_energy(l3, msgs, ps3, NamedTuple())))
end

# P5: a singular nonzero noise, exact along its null space and soft along its range
# (hard_constraint.md, "Free energy with a singular noise").
@testset "P5: HardConstraintFactor with a singular noise in a tree: the Bethe free energy is −log Z" begin
    HF = HardFreeEnergyOracle
    rng = Xoshiro(63)
    nfixed = 0
    for trial in 1:40
        n, nu, nv, nw = rand(rng, 1:3), rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:2)
        m = rand(rng, 2:(nu + nv + 1))                 # the soft equations may exceed the unknowns
        r = rand(rng, max(1, m - nu - nv):(m - 1))     # rank of the noise, 0 < r < m
        R = HF.psd(rng, m, r)
        g, p = HF.model(rng, n, nu, nv, nw, m; noise = R)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        μp, Σp, nlogZ, ix = HF.dense(p, ps, n, nu, nv, nw, m; R)
        marg, rep, _, s = infer!(g, tree_schedule(g), ps, st)
        @test rep.converged
        for v in (:x, :u, :v, :w)
            e = extended(marg[v], length(ix[v]))
            @test e.μ ≈ μp[ix[v]] atol = 1e-7 rtol = 1e-9
            @test e.Σ ≈ Σp[ix[v], ix[v]] atol = 1e-7 rtol = 1e-9
        end
        nfixed += size(hard_directions(extended(marg[:u], nu)), 2) > 0
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, st))
        @test F ≈ nlogZ atol = 1e-7 * (1 + abs(nlogZ))
    end
    @test nfixed >= 10
end

@testset "P5: the term with a singular noise is the limit of the noisy relation's (noise R + εQₕQₕᵀ)" begin
    HF = HardFreeEnergyOracle
    rng = Xoshiro(64)
    errs = Float64[]
    for trial in 1:80
        k = rand(rng, 2:3)
        names = (:a, :b, :c)[1:k]
        dims = NamedTuple{names}(Tuple(rand(rng, 1:3) for _ in 1:k))
        msgs = NamedTuple{names}(Tuple(begin
            kind = rand(rng, (:proper, :dirac, :hard))
            kind === :proper ? Gaussian(randn(rng, dims[n]), HF.spd(rng, dims[n])) :
            kind === :dirac ? DiracBelief(randn(rng, dims[n])) : HF.random_eg(rng, dims[n]; kf = 0)
        end for n in names))
        free = sum(n -> msgs[n] isa DiracBelief ? 0 :
                        msgs[n] isa ExtendedGaussian ? dims[n] - size(hard_directions(msgs[n]), 2) : dims[n], names)
        free == 0 && continue
        m = rand(rng, 2:(free + 1))
        r = rand(rng, max(1, m - free):(m - 1))        # m − r ≤ free exact equations
        R = HF.psd(rng, m, r)
        hard = HardConstraintFactor(dims, m; noise = R)
        ps = LuxCore.initialparameters(Xoshiro(trial), hard)
        F0 = LenticulumCore.scalarise(LenticulumCore.scalarisation(hard), first(Mycelium.local_free_energy(hard, msgs, ps, NamedTuple())))
        G, c, canon = Lenticulum._reduced_relation(ps.A, ps.c, msgs, dims, names)
        # the noise rebuilt from its split: Q's null eigenvalues are ±1e-17 in Float64, far above ε
        Qh, Qs, d = Lenticulum._noise_split(hard.Q)
        @test Qs * Diagonal(d) * Qs' ≈ R
        Fε = setprecision(BigFloat, 512) do
            ε = big"1e-30"
            Qhb, Qsb = BigFloat.(Qh), BigFloat.(Qs)
            Rε = Symmetric(Qsb * Diagonal(BigFloat.(d)) * Qsb' + ε * Qhb * Qhb')
            Ri = inv(Rε)
            Gb, cb = BigFloat.(G), BigFloat.(c)
            Λs = BigFloat.(Lenticulum._blockdiag(map(last, canon))); ηs = BigFloat.(reduce(vcat, map(first, canon)))
            Λ = Symmetric(Gb' * Ri * Gb + Λs); η = Gb' * Ri * cb + ηs
            Σ = inv(Λ); mb = Σ * η
            res = Gb * mb - cb
            fit = (res' * Ri * res + tr(Ri * Gb * Σ * Gb')) / 2
            complexity = logdet(cholesky(Symmetric(2 * BigFloat(π) * Rε))) / 2
            H = size(Gb, 2) * (1 + log(2 * BigFloat(π))) / 2 - logdet(cholesky(Λ)) / 2
            Float64(fit + complexity - H)
        end
        push!(errs, abs(Fε - F0) / (1 + abs(F0)))
    end
    @test length(errs) > 60
    @test maximum(errs) < 1e-9
end

@testset "P5: a singular noise, by hand" begin
    # x ~ N(0, I₂); x = ε with ε ~ N(0, diag(1, 0)): Z = N(0; 0, 2) N(0; 0, 1)
    gb = GraphBuilder()
    variable!(gb, :x, 2)
    factor!(gb, :p, GaussianPrior(:x, zeros(2), Matrix(1.0I, 2, 2))); connect!(gb, :p, :x, :x)
    factor!(gb, :rel, HardConstraintFactor((x = 2,), 2; noise = [1.0 0; 0 0])); connect!(gb, :rel, :x, :x)
    g = validate(build(gb))
    ps = (p = LuxCore.initialparameters(Xoshiro(1), g).p, rel = (A = (x = Matrix(1.0I, 2, 2),), c = zeros(2)))
    st = LuxCore.initialstates(Xoshiro(1), g)
    marg, _, _, s = infer!(g, tree_schedule(g), ps, st)
    e = extended(marg[:x], 2)
    @test e.μ ≈ zeros(2) atol = 1e-12
    @test e.Σ ≈ [0.5 0; 0 0] atol = 1e-12
    @test first(LenticulumCore.scalar_free_energy(s, g, ps, st)) ≈ (log(4π) + log(2π)) / 2
end

@testset "Review of P5, F1: a flat factor belief with a positive definite noise is refused" begin
    # x, y uninformative: the factor belief's precision GᵀQ⁻¹G has rank 1 of 3. Rounding left an
    # eigenvalue of about 1e-16 in 25 of these 400 draws, and the term came out finite.
    rng = Xoshiro(1)
    for t in 1:200, noise in (1.0, 0.7)
        a = randn(rng, 1, 3)
        ps = (A = (x = a[:, 1:2], y = a[:, 3:3]), c = [0.3])
        msgs = (x = TrivialBelief(), y = TrivialBelief())
        @test_throws ArgumentError Mycelium.local_free_energy(HardConstraintFactor((x = 2, y = 1), 1; noise), msgs, ps, NamedTuple())
        @test_throws ArgumentError Mycelium.local_free_energy(LinearConstraintFactor((x = 2, y = 1), 1; noise), msgs, ps, NamedTuple())
    end
end

# Found by the review of U3 (LP-070, finding 1), not fixed: a Gaussian z ∈ ℝ³ observed exactly
# through two oblique scalar projections, each its own HardConstraintFactor (k·z = y₁, h·z = y₂).
# The tree's marginals are right, its Bethe free energy is not: each factor's ½ log det(G̃G̃ᵀ) is
# taken on a cavity support the other observation has already cut, and the two log-sine terms do
# not cancel. Dense: −log N((y₁, y₂); Lμ, LCLᵀ), L = [k; h].
@testset "LP-070: two oblique exact projections of one variable" begin
    μ, C = [0.8, -0.2, 1.1], [1.6 0.4 -0.1; 0.4 0.95 0.25; -0.1 0.25 1.2]
    k, h, y = [0.8, -0.1, 0.9], [0.9, 0.3, 0.7], [0.25, 1.4]
    gb = GraphBuilder()
    variable!(gb, :z, 3)
    factor!(gb, :prior, GaussianPrior(:out, μ, C)); connect!(gb, :prior, :out, :z)
    for (i, row) in enumerate((k, h))
        o, rel, dat = Symbol(:y, i), Symbol(:rel, i), Symbol(:data, i)
        variable!(gb, o, 1)
        factor!(gb, rel, HardConstraintFactor((u = 3, y = 1), 1)); connect!(gb, rel, :u, :z); connect!(gb, rel, :y, o)
        factor!(gb, dat, DataFactor(:out, [y[i]])); connect!(gb, dat, :out, o)
    end
    g = validate(build(gb))
    ps, st = LuxCore.setup(Xoshiro(0), g)
    ps = merge(ps, (rel1 = (A = (u = reshape(k, 1, 3), y = fill(-1.0, 1, 1)), c = [0.0]),
                    rel2 = (A = (u = reshape(h, 1, 3), y = fill(-1.0, 1, 1)), c = [0.0])))
    marg, _, st, s = infer!(g, tree_schedule(g), ps, st)
    L = [k'; h']
    V = L * C * L'
    δ = y - L * μ
    K = C * L' / V
    @test extended(marg[:z], 3).μ ≈ μ + K * δ atol = 1e-8
    @test extended(marg[:z], 3).Σ ≈ C - K * L * C atol = 1e-8
    nlogZ = (δ' * (V \ δ) + logdet(2π * V)) / 2
    @test_broken first(LenticulumCore.scalar_free_energy(s, g, ps, st)) ≈ nlogZ atol = 1e-8
end
