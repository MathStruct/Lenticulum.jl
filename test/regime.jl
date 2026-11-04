# RegimeFactor: per-outcome coefficients and noise (L-SPEC-009, `cluster.md` "Per-outcome
# coefficients and noise"). Oracles: the closed forms of a mixture of factor analysers
# (p(z) = Σ_k w_k N(z; μ_k, W_k W_kᵀ + Q_k), the posterior of x given z and k), of a Gaussian
# mixture with one covariance per component, and finite differences of −log p(z).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote
using LenticulumCore: MixtureBelief, CategoricalBelief, DiracBelief

const RG_K, RG_DX, RG_DZ = 3, 2, 3
const RG_W = [[1.0 0.2; -0.5 1.0; 0.3 0.4], [0.2 -1.0; 1.5 0.1; 0.0 0.7], [0.8 0.8; 0.0 -0.6; 1.2 0.0]]
const RG_MU = [0.0 2.0 -1.5; 1.0 -1.0 0.5; 0.0 0.5 2.0]
const RG_Q = [Diagonal([0.1, 0.2, 0.05]) |> Matrix, [0.3 0.1 0.0; 0.1 0.2 0.0; 0.0 0.0 0.4], Matrix(0.15I, 3, 3)]
const RG_LW = log.([0.5, 0.3, 0.2])
rg_ps(; b = RG_MU, logw = RG_LW, W = RG_W) = (A = (z = cat(fill(Matrix(1.0I, RG_DZ, RG_DZ), RG_K)...; dims = 3),
    x = cat((-Wk for Wk in W)...; dims = 3)), b = b, logw = logw)

# −log p(z) of the mixture of factor analysers, x ~ N(0, I)
rg_lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
rg_logN(z, μ, S) = -(dot(z - μ, S \ (z - μ)) + logdet(2π * S)) / 2
rg_nlp(z; b = RG_MU, logw = RG_LW, Q = RG_Q, W = RG_W) =
    -rg_lse([logw[k] - rg_lse(logw) + rg_logN(z, b[:, k], W[k] * W[k]' + Q[k]) for k in 1:RG_K])

function rg_graph(z; learn_noise = false, cl = true)
    b = GraphBuilder()
    variable!(b, :z, RG_DZ); variable!(b, :x, RG_DX); variable!(b, :k, RG_K)
    factor!(b, :data, DataFactor(:z, z)); connect!(b, :data, :z, :z; direction = Emitting())
    factor!(b, :pr, GaussianPrior(:x, zeros(RG_DX), Matrix(1.0I, RG_DX, RG_DX))); connect!(b, :pr, :x, :x)
    factor!(b, :reg, RegimeFactor((z = RG_DZ, x = RG_DX), RG_K, RG_DZ; noise = RG_Q, learn_noise))
    connect!(b, :reg, :z, :z); connect!(b, :reg, :x, :x); connect!(b, :reg, :k, :k)
    g = build(b)
    return cl ? cluster(g, :c, [:pr, :reg]) : g
end

@testset "RegimeFactor: a mixture of factor analysers in a cluster is exact" begin
    z = [0.7, -0.4, 1.1]
    gc = rg_graph(z)
    ps = cluster_parameters(gc, :c, (data = NamedTuple(), pr = NamedTuple(), reg = rg_ps()))
    _, rep, _, s = infer!(gc, tree_schedule(gc), ps, NamedTuple())
    @test rep.converged
    F = first(scalar_free_energy(s, gc, ps, NamedTuple()))
    @test F ≈ rg_nlp(z) rtol = 1e-12
    # the posterior over regimes and of x given each: three precisions
    lr = [RG_LW[k] + rg_logN(z, RG_MU[:, k], RG_W[k] * RG_W[k]' + RG_Q[k]) for k in 1:RG_K]
    qk = cluster_marginal(s, gc, :c, :k, ps)
    @test exp.(qk.logp) ≈ exp.(lr .- rg_lse(lr)) atol = 1e-12
    qx = cluster_marginal(s, gc, :c, :x, ps)
    @test qx isa MixtureBelief && length(qx.components) == RG_K
    for k in 1:RG_K
        S = inv(I + RG_W[k]' * (RG_Q[k] \ RG_W[k]))
        m = S * RG_W[k]' * (RG_Q[k] \ (z - RG_MU[:, k]))
        @test belief_mean(qx.components[k]) ≈ m atol = 1e-12
        @test belief_cov(qx.components[k]) ≈ S atol = 1e-12
    end
    @test length(unique(belief_cov.(qx.components))) == RG_K
end

@testset "RegimeFactor: free-energy gradient in A, b, logw and the noise" begin
    z = [0.7, -0.4, 1.1]
    gc = rg_graph(z; learn_noise = true)
    f = RegimeFactor((z = RG_DZ, x = RG_DX), RG_K, RG_DZ; noise = RG_Q, learn_noise = true)
    L0 = LuxCore.initialparameters(Xoshiro(1), f).L
    @test regime_covariances(f, (L = L0,)) ≈ cat(RG_Q...; dims = 3) atol = 1e-12
    psr = merge(rg_ps(), (L = L0,))
    ps = cluster_parameters(gc, :c, (data = NamedTuple(), pr = NamedTuple(), reg = psr))
    _, _, _, s = infer!(gc, tree_schedule(gc), ps, NamedTuple())
    nlp(p) = (Q = regime_covariances(f, p); rg_nlp(z; b = p.b, logw = p.logw, Q = [Q[:, :, k] for k in 1:RG_K],
                                                   W = [-p.A.x[:, :, k] for k in 1:RG_K]))
    h = 1e-6
    fd(get, set) = [(nlp(set(psr, j, h)) - nlp(set(psr, j, -h))) / 2h for j in eachindex(get(psr))]
    bump(a, j, δ) = (a = copy(a); a[j] += δ; a)
    fdb = fd(p -> p.b, (p, j, δ) -> merge(p, (b = bump(p.b, j, δ),)))
    fdL = fd(p -> p.L, (p, j, δ) -> merge(p, (L = bump(p.L, j, δ),)))
    fdA = fd(p -> p.A.x, (p, j, δ) -> merge(p, (A = (z = p.A.z, x = bump(p.A.x, j, δ)),)))
    fdw = fd(p -> p.logw, (p, j, δ) -> merge(p, (logw = bump(p.logw, j, δ),)))
    for ad in (AutoForwardDiff(), AutoZygote())
        G = free_energy_gradient(s, gc, ps, NamedTuple(); ad).c.reg
        @test isapprox(vec(G.b), fdb; atol = 1e-6) && isapprox(vec(G.logw), fdw; atol = 1e-6)
        @test isapprox(vec(G.A.x), fdA; atol = 1e-6)
        # the strict upper triangle of L is not used: zero gradient there, and in the differences
        @test isapprox(vec(G.L), fdL; atol = 1e-6)
    end
end

@testset "RegimeFactor outside a cluster: one-factor messages and free energy" begin
    # a Gaussian mixture with one covariance per component: z = μ_k + ε_k, observed
    z = [0.4, 0.9, -0.2]
    f = RegimeFactor((z = RG_DZ,), RG_K, RG_DZ; noise = RG_Q)
    psf = (A = (z = cat(fill(Matrix(1.0I, 3, 3), RG_K)...; dims = 3),), b = RG_MU, logw = RG_LW)
    b = GraphBuilder()
    variable!(b, :z, RG_DZ); variable!(b, :k, RG_K)
    factor!(b, :data, DataFactor(:z, z)); connect!(b, :data, :z, :z; direction = Emitting())
    factor!(b, :reg, f); connect!(b, :reg, :z, :z); connect!(b, :reg, :k, :k)
    g = build(b)
    ps = (data = NamedTuple(), reg = psf)
    _, _, _, s = infer!(g, tree_schedule(g), ps, NamedTuple())
    lr = [RG_LW[k] + rg_logN(z, RG_MU[:, k], RG_Q[k]) for k in 1:RG_K]
    @test first(scalar_free_energy(s, g, ps, NamedTuple())) ≈ -rg_lse(lr) rtol = 1e-12
    mk, _ = factor_message(f, :k, nothing, (z = DiracBelief(z),), nothing, psf, NamedTuple())
    @test exp.(mk.logp) ≈ exp.(lr .- rg_lse(lr)) atol = 1e-12
    for ad in (AutoForwardDiff(), AutoZygote())
        G = free_energy_gradient(s, g, ps, NamedTuple(); ad).reg
        # ∂/∂μ_k of −log p(z) = −r_k Q_k⁻¹ (z − μ_k)
        rk = exp.(lr .- rg_lse(lr))
        @test G.b ≈ reduce(hcat, [-rk[k] * (RG_Q[k] \ (z - RG_MU[:, k])) for k in 1:RG_K]) atol = 1e-9
    end
    # the message to a continuous channel: components with their own precision Λ_k = W_kᵀQ_k⁻¹W_k
    fx = RegimeFactor((z = RG_DZ, x = RG_DX), RG_K, RG_DZ; noise = RG_Q)
    mx, _ = factor_message(fx, :x, nothing, (z = DiracBelief(z),), nothing, rg_ps(), NamedTuple())
    @test mx isa MixtureBelief && length(mx.components) == RG_K
    lm = map(1:RG_K) do k
        Λ = RG_W[k]' * (RG_Q[k] \ RG_W[k]); η = RG_W[k]' * (RG_Q[k] \ (z - RG_MU[:, k]))
        @test mx.components[k].Λ ≈ Λ atol = 1e-12
        @test mx.components[k].η ≈ η atol = 1e-12
        # the weight: w_k ∫ N(z; μ_k + W_k x, Q_k) dx
        RG_LW[k] - dot(z - RG_MU[:, k], RG_Q[k] \ (z - RG_MU[:, k])) / 2 - logdet(2π * RG_Q[k]) / 2 +
            dot(η, Λ \ η) / 2 + RG_DX * log(2π) / 2 - logdet(Λ) / 2
    end
    @test exp.(mx.logw) ≈ exp.(lm .- rg_lse(lm)) atol = 1e-12
    # improper components with different precisions are refused: m = 1 < dim x
    f1 = RegimeFactor((z = 1, x = 2), 2, 1; noise = [fill(0.1, 1, 1), fill(0.3, 1, 1)])
    p1 = (A = (z = ones(1, 1, 2), x = cat([-1.0 0.5], [0.3 -2.0]; dims = 3)), b = [0.0 1.0], logw = zeros(2))
    @test_throws ArgumentError factor_message(f1, :x, nothing, (z = DiracBelief([0.2]),), nothing, p1, NamedTuple())
    # with z unobserved and x given, the message to z is a mixture with per-regime noise
    mz, _ = factor_message(f1, :z, nothing, (x = DiracBelief([1.0, 2.0]),), nothing, p1, NamedTuple())
    @test mz isa MixtureBelief && [c.Λ[1] for c in mz.components] ≈ [10.0, 1 / 0.3]
end

@testset "RegimeFactor: construction" begin
    f = RegimeFactor((z = 2, x = 1), 3, 2; noise = [Matrix(s * I, 2, 2) for s in (0.1, 0.2, 0.3)], learn_noise = true)
    p = LuxCore.initialparameters(Xoshiro(2), f)
    @test size(p.A.z) == (2, 2, 3) && size(p.A.x) == (2, 1, 3) && size(p.b) == (2, 3) && size(p.L) == (2, 2, 3)
    @test LuxCore.parameterlength(f) == 3 * (2 * 3 + 2 + 1) + 3 * 3
    @test regime_covariances(f, p)[:, :, 2] ≈ Matrix(0.2I, 2, 2)
    @test_throws ArgumentError RegimeFactor((z = 2,), 2, 2; noise = [0.1, 0.2, 0.3])          # a diagonal of the wrong size
    @test_throws ArgumentError RegimeFactor((z = 2,), 2, 2; noise = [Matrix(0.1I, 2, 2)])      # one covariance for two regimes
    @test RegimeFactor((z = 2,), 2, 2; noise = cat(Matrix(0.1I, 2, 2), Matrix(0.2I, 2, 2); dims = 3)).Q[2, 2, 2] == 0.2
    @test_throws ArgumentError RegimeFactor((k = 2,), 2, 2)
    @test_throws DimensionMismatch Mycelium.ClusterForm((z = zeros(2, 2, 3),), zeros(2, 2), zeros(2, 2), zeros(2), :k)
    # per-outcome forms are not batched over a plate: the per-replica path is taken
    @test Mycelium._varies(cluster_form(f, p)) && !Mycelium._varies(cluster_form(SwitchingConstraintFactor((z = 2,), 2, 2),
        (A = (z = Matrix(1.0I, 2, 2),), b = zeros(2, 2), logw = zeros(2))))
end

@testset "RegimeFactor over a plate: per-replica cluster nodes, exact" begin
    Z = [0.7 -1.0 2.1 0.0; -0.4 0.3 -0.8 1.5; 1.1 0.9 2.0 -0.5]
    N = size(Z, 2)
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:z), RG_DZ); variable!(b, r(:x), RG_DX); variable!(b, r(:k), RG_K)
        factor!(b, r(:data), DataFactor(:z, Z[:, i])); connect!(b, r(:data), :z, r(:z); direction = Emitting())
        factor!(b, r(:pr), GaussianPrior(:x, zeros(RG_DX), Matrix(1.0I, RG_DX, RG_DX)); params = :pr)
        factor!(b, r(:reg), RegimeFactor((z = RG_DZ, x = RG_DX), RG_K, RG_DZ; noise = RG_Q); params = :reg)
        connect!(b, r(:pr), :x, r(:x)); connect!(b, r(:reg), :z, r(:z)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    gp = cluster(as_plate(build(b), :pts, N), :pts, :c, [:pr, :reg]; params = :mix)
    ps = merge(NamedTuple{Tuple(replica_name(:data, i) for i in 1:N)}(ntuple(_ -> NamedTuple(), N)),
               (mix = (pr = NamedTuple(), reg = rg_ps()),))
    s = MessageStore(gp)
    for _ in 1:2; sweep!(s, gp, plate_schedule(gp), ps, NamedTuple()); end
    @test first(scalar_free_energy(s, gp, ps, NamedTuple())) ≈ sum(rg_nlp, eachcol(Z)) rtol = 1e-12
    pl = only(plates(gp))
    cs = [gp.factors[pl.factors[1, i]].factor for i in 1:N]
    @test !Mycelium._cluster_batchable(cs, [NamedTuple() for _ in 1:N], nothing, ps.mix)
    G = free_energy_gradient(s, gp, ps, NamedTuple(); ad = AutoForwardDiff()).mix.reg
    h = 1e-6
    fdb = map(1:length(RG_MU)) do j
        e = zeros(size(RG_MU)); e[j] = h
        (sum(z -> rg_nlp(z; b = RG_MU + e), eachcol(Z)) - sum(z -> rg_nlp(z; b = RG_MU - e), eachcol(Z))) / 2h
    end
    @test isapprox(vec(G.b), fdb; atol = 1e-6)
end
