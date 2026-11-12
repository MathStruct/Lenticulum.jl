# RegimeFactor over a plate, batched (L-SPEC-009 §8–9, Feature Backlog I6): tied replicas whose
# coefficients and noise depend on the outcome share their per-class precisions, and the batched
# messages, beliefs, free energies, gradients and EM statistics equal the per-replica ones.
# Oracles: the per-replica path (itself checked against closed forms in `regime.jl`), the closed
# form −log p of a mixture of factor analysers and of a mixture of regressions, and central
# differences.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using MyceliumLearning
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote
using LenticulumCore: MixtureBelief, CategoricalBelief, DiracBelief, GradedEnergy

# fieldwise closeness of two beliefs (or of the pieces of one)
rp_close(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}; atol) = size(a) == size(b) && isapprox(a, b; atol, rtol = 0)
rp_close(a::Number, b::Number; atol) = isapprox(a, b; atol, rtol = 0)
rp_close(a::AbstractVector, b::AbstractVector; atol) = length(a) == length(b) && all(rp_close(x, y; atol) for (x, y) in zip(a, b))
rp_close(a, b; atol) = typeof(a) == typeof(b) &&
    (isempty(fieldnames(typeof(a))) ? isequal(a, b) : all(f -> rp_close(getfield(a, f), getfield(b, f); atol), fieldnames(typeof(a))))
rp_total(e::GradedEnergy) = e[:fit] + e[:complexity] + e[:negentropy]

# A random regime factor with per-outcome coefficients on both channels and learned noise.
function rp_factor(rng, dz, dx, K)
    f = RegimeFactor((z = dz, x = dx), K, dz; learn_noise = true)
    p0 = LuxCore.initialparameters(rng, f)
    p = (A = (z = cat((Matrix(1.0I, dz, dz) .+ 0.4 .* randn(rng, dz, dz) for _ in 1:K)...; dims = 3), x = randn(rng, dz, dx, K)),
         b = randn(rng, dz, K), logw = randn(rng, K), L = p0.L .+ 0.3 .* randn(rng, dz, dz, K))
    return f, p
end
rp_gauss(rng, d, Λ) = GaussianBelief(randn(rng, d), Λ)

@testset "RegimeFactor plate: batched messages and energies equal per replica (random)" begin
    for seed in 1:6
        rng = Xoshiro(seed)
        dz, dx, K, N = rand(rng, 1:3), rand(rng, 1:2), rand(rng, 2:3), 5
        f, p = rp_factor(rng, dz, dx, K)
        fs = [RegimeFactor((z = dz, x = dx), K, dz; learn_noise = true) for _ in 1:N]   # separate, equal objects
        Λx = (S = randn(rng, dx, dx); S * S' + I)
        Λz = (S = randn(rng, dz, dz); S * S' + I)
        lk = CategoricalBelief(Val(:log), (l = randn(rng, K); l .- log(sum(exp.(l)))), nothing)
        cases = [
            (:k, [(z = DiracBelief(randn(rng, dz)), x = rp_gauss(rng, dx, Λx)) for _ in 1:N]),
            (:x, [(z = DiracBelief(randn(rng, dz)),) for _ in 1:N]),
            (:x, [(z = DiracBelief(randn(rng, dz)), k = lk) for _ in 1:N]),
            (:z, [(x = DiracBelief(randn(rng, dx)), k = lk) for _ in 1:N]),
            (:z, [(x = rp_gauss(rng, dx, Λx),) for _ in 1:N]),
            (:x, [(z = rp_gauss(rng, dz, Λz),) for _ in 1:N])]
        for (target, inputs) in cases
            @test Mycelium._cluster_batchable(Lenticulum._as_clusters(fs), inputs, target, (factor = p,))
            mb, _ = factor_messages(fs, target, nothing, inputs, fill(nothing, N), p, NamedTuple())
            ms = [first(factor_message(fs[i], target, nothing, inputs[i], nothing, p, NamedTuple())) for i in 1:N]
            @test all(rp_close(mb[i], ms[i]; atol = 1e-10) for i in 1:N)
        end
        # beliefs, free energies and their gradient: z a point, x a Gaussian message; and both points
        for msgs in ((z = belief_block([DiracBelief(randn(rng, dz)) for _ in 1:N]), x = belief_block([rp_gauss(rng, dx, Λx) for _ in 1:N])),
                     (z = belief_block([DiracBelief(randn(rng, dz)) for _ in 1:N]), x = belief_block([DiracBelief(randn(rng, dx)) for _ in 1:N])))
            B = factor_beliefs(fs, msgs, p, NamedTuple())
            @test B isa FrozenClusterBeliefs && length(B.Σ) == K
            bs = [factor_belief(fs[i], Mycelium._column(msgs, i), p, NamedTuple()) for i in 1:N]
            Eb, _ = belief_free_energies(fs, B, p, NamedTuple())
            Es = [first(belief_free_energy(fs[i], bs[i], p, NamedTuple())) for i in 1:N]
            @test all(rp_close(collect(values(Eb[i].parts)), collect(values(Es[i].parts)); atol = 1e-10) for i in 1:N)
            El, _ = local_free_energies(fs, msgs, p, NamedTuple())
            @test rp_total.(El) ≈ rp_total.(Es) atol = 1e-10
            Fb(q) = sum(rp_total, first(belief_free_energies(fs, B, q, NamedTuple())))
            Fs(q) = sum(i -> rp_total(first(belief_free_energy(fs[i], bs[i], q, NamedTuple()))), 1:N)
            gb, gs = only(Zygote.gradient(Fb, p)), only(Zygote.gradient(Fs, p))
            @test all(k -> isapprox(getfield(gb, k), getfield(gs, k); atol = 1e-9), (:b, :logw, :L))
            @test gb.A.z ≈ gs.A.z atol = 1e-9
            @test gb.A.x ≈ gs.A.x atol = 1e-9
            fd = ForwardDiff.gradient(b -> Fb(merge(p, (b = b,))), p.b)
            @test fd ≈ gs.b atol = 1e-9
        end
    end
end

# A mixture of factor analysers over a plate, the regime factor and the prior of x in one cluster
# node per point, random parameters: free energy and gradient against the closed form.
rp_lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
rp_logN(z, μ, S) = -(dot(z - μ, S \ (z - μ)) + logdet(2π * S)) / 2
function rp_nlp(z, A, b, logw, Q)          # A.z[:,:,k] z + A.x[:,:,k] x = b_k + ε, x ~ N(0, I)
    K = length(logw)
    -rp_lse([logw[k] - rp_lse(logw) + begin
        Az, Ax = A.z[:, :, k], A.x[:, :, k]
        # the factor is a density in its residual (no output declared), so ∫ f(z, x, k) N(x; 0, I) dx
        # = w_k N(A_z z; b_k, A_x A_xᵀ + Q_k), without the Jacobian of z
        rp_logN(Az * z, b[:, k], Ax * Ax' + Q[:, :, k])
    end for k in 1:K])
end

@testset "RegimeFactor plate: clustered mixture of factor analysers, batched, against the closed form (random)" begin
    for seed in 1:3
        rng = Xoshiro(100 + seed)
        dz, dx, K, N = 3, 2, 3, 7
        f, p = rp_factor(rng, dz, dx, K)
        Z = randn(rng, dz, N)
        b = GraphBuilder()
        for i in 1:N
            r(s) = replica_name(s, i)
            variable!(b, r(:z), dz); variable!(b, r(:x), dx); variable!(b, r(:k), K)
            factor!(b, r(:data), DataFactor(:z, Z[:, i])); connect!(b, r(:data), :z, r(:z); direction = Emitting())
            factor!(b, r(:pr), GaussianPrior(:x, zeros(dx), Matrix(1.0I, dx, dx)); params = :pr)
            factor!(b, r(:reg), RegimeFactor((z = dz, x = dx), K, dz; learn_noise = true); params = :reg)
            connect!(b, r(:pr), :x, r(:x)); connect!(b, r(:reg), :z, r(:z)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
        end
        gp = cluster(as_plate(build(b), :pts, N), :pts, :c, [:pr, :reg]; params = :mix)
        ps = merge(NamedTuple{Tuple(replica_name(:data, i) for i in 1:N)}(ntuple(_ -> NamedTuple(), N)),
                   (mix = (pr = NamedTuple(), reg = p),))
        s = MessageStore(gp)
        for _ in 1:2; sweep!(s, gp, plate_schedule(gp), ps, NamedTuple()); end
        Q = regime_covariances(f, p)
        @test first(scalar_free_energy(s, gp, ps, NamedTuple())) ≈ sum(z -> rp_nlp(z, p.A, p.b, p.logw, Q), eachcol(Z)) rtol = 1e-11
        pl = only(plates(gp))
        row = only(r for r in Mycelium._energy_rows(s, gp) if first(r.fids) == pl.factors[findfirst(i -> gp.factors[pl.factors[i, 1]].factor isa ClusterFactor, axes(pl.factors, 1)), 1])
        cs = [gp.factors[i].factor for i in row.fids]
        @test factor_beliefs(cs, Mycelium._row_messages(s, row), ps.mix, NamedTuple()) isa FrozenClusterBeliefs
        nlp(q) = sum(z -> rp_nlp(z, q.A, q.b, q.logw, regime_covariances(f, q)), eachcol(Z))
        ref = only(Zygote.gradient(nlp, p))
        for ad in (AutoForwardDiff(), AutoZygote())
            G = free_energy_gradient(s, gp, ps, NamedTuple(); ad).mix.reg
            @test G.b ≈ ref.b atol = 1e-8
            @test G.A.x ≈ ref.A.x atol = 1e-8
            @test G.A.z ≈ ref.A.z atol = 1e-8
            @test G.L ≈ ref.L atol = 1e-8
            @test G.logw ≈ ref.logw atol = 1e-8
        end
    end
end

# The benchmark's model (`bench/rxinfer_baseline.jl`): a mixture of regressions, one RegimeFactor
# per point outside any cluster, as a plate and without one.
function rp_mixreg_graph(x, y; plated)
    b = GraphBuilder()
    for i in eachindex(x)
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 1); variable!(b, r(:y), 1); variable!(b, r(:k), 2)
        factor!(b, r(:dx), DataFactor(:x, [x[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
        factor!(b, r(:dy), DataFactor(:y, [y[i]])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
        factor!(b, r(:reg), RegimeFactor((y = 1, x = 1), 2, 1; learn_noise = true, output = :y); params = :reg)
        connect!(b, r(:reg), :y, r(:y)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    g = build(b)
    return plated ? as_plate(g, :pts, length(x)) : g
end
rp_mixreg_ps(N, a, c, σ, w) = merge(NamedTuple(s => NamedTuple() for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i))),
    (reg = (A = (x = reshape(-a, 1, 1, 2),), b = reshape(c, 1, 2), logw = log.(w), L = reshape(log.(σ), 1, 1, 2)),))
rp_mixreg_nlp(x, y, a, c, σ, w) = -sum(i -> rp_lse([log(w[k] / sum(w)) + rp_logN([y[i]], [a[k] * x[i] + c[k]], fill(σ[k]^2, 1, 1)) for k in 1:2]), eachindex(x))

@testset "RegimeFactor plate: a mixture of regressions outside clusters, batched EM and gradient" begin
    rng = Xoshiro(7)
    N = 40
    x = randn(rng, N)
    y = [rand(rng) < 0.6 ? x[i] + 0.2randn(rng) : -x[i] + 0.5 + 0.5randn(rng) for i in 1:N]
    a, c, σ, w = [0.8, -0.7], [0.1, 0.4], [0.3, 0.6], [0.5, 0.5]
    ps = rp_mixreg_ps(N, a, c, σ, w)
    gp, gu = rp_mixreg_graph(x, y; plated = true), rp_mixreg_graph(x, y; plated = false)
    sp = MessageStore(gp); sweep!(sp, gp, plate_schedule(gp), ps, NamedTuple())
    su = MessageStore(gu); sweep!(su, gu, SequentialSchedule(all_tasks(gu)), ps, NamedTuple())
    @test first(scalar_free_energy(sp, gp, ps, NamedTuple())) ≈ rp_mixreg_nlp(x, y, a, c, σ, w) rtol = 1e-12
    @test first(scalar_free_energy(su, gu, ps, NamedTuple())) ≈ rp_mixreg_nlp(x, y, a, c, σ, w) rtol = 1e-12
    # the plate row of regime factors is read from stacked beliefs
    pl = only(plates(gp))
    kr = findfirst(i -> gp.factors[pl.factors[i, 1]].factor isa RegimeFactor, axes(pl.factors, 1))
    row = only(r for r in Mycelium._energy_rows(sp, gp) if first(r.fids) == pl.factors[kr, 1])
    @test factor_beliefs([gp.factors[i].factor for i in row.fids], Mycelium._row_messages(sp, row), ps.reg, NamedTuple()) isa FrozenClusterBeliefs
    sb = moment_statistics(sp, gp, ps, NamedTuple())
    sn = moment_statistics(sp, gp, ps, NamedTuple(); batched = false)
    @test keys(sb) == keys(sn) == Set([(:reg, nothing)])
    st_b, st_n = sb[(:reg, nothing)][2], sn[(:reg, nothing)][2]
    @test st_b.n ≈ st_n.n rtol = 1e-12
    @test st_b.m ≈ st_n.m rtol = 1e-12
    @test st_b.S ≈ st_n.S rtol = 1e-12
    for ad in (AutoForwardDiff(), AutoZygote())
        Gp = free_energy_gradient(sp, gp, ps, NamedTuple(); ad).reg
        Gu = free_energy_gradient(su, gu, ps, NamedTuple(); ad).reg
        @test all(k -> isapprox(getfield(Gp, k), getfield(Gu, k); atol = 1e-9), (:b, :logw, :L))
        @test Gp.A.x ≈ Gu.A.x atol = 1e-9
    end
    # EM on the plate reaches the unplated fit
    pp, Fp, _ = fit_em(gp, ps, NamedTuple(); schedule = plate_schedule(gp), iters = 200, tol = 1e-12)
    pu, Fu, _ = fit_em(gu, ps, NamedTuple(); schedule = SequentialSchedule(all_tasks(gu)), iters = 200, tol = 1e-12)
    @test all(diff(Fp) .<= 1e-8)
    @test Fp ≈ Fu rtol = 1e-10
    @test pp.reg.b ≈ pu.reg.b atol = 1e-8
    @test pp.reg.A.x ≈ pu.reg.A.x atol = 1e-8
end

# Review of I6 (LP-032, finding 1): a categorical input with reordered labels on a replica other
# than the first was read positionally by the batched path, which checked only replica 1's
# inputs; the per-replica path refuses it. The labels are part of the shape now.
@testset "RegimeFactor plate: a mislabeled categorical input on a later replica is refused" begin
    rng = Xoshiro(31)
    f, p = rp_factor(rng, 2, 1, 2)
    fs = [RegimeFactor((z = 2, x = 1), 2, 2; learn_noise = true) for _ in 1:3]
    l = log.([0.3, 0.7])
    ins = [(z = DiracBelief(randn(rng, 2)), k = CategoricalBelief(Val(:log), l, i == 2 ? (2, 1) : nothing)) for i in 1:3]
    @test !Mycelium._cluster_batchable(Lenticulum._as_clusters(fs), ins, :x, (factor = p,))
    @test_throws ArgumentError factor_messages(fs, :x, nothing, ins, fill(nothing, 3), p, NamedTuple())
    msgs = (z = belief_block([i.z for i in ins]), k = Mycelium.VectorBlock(Any[i.k for i in ins]))
    @test_throws ArgumentError factor_beliefs(fs, msgs, p, NamedTuple())
end
