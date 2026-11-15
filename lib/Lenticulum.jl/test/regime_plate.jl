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
    # as blocks (I20, Mistral's review): a categorical block holds one set of labels, so inputs with
    # different labels stack into a vector block, which the block path sends through the columns
    @test belief_block([i.k for i in ins]) isa Mycelium.VectorBlock
    @test !Mycelium._block_batchable(Lenticulum._as_clusters(fs), msgs, :x)
    @test_throws ArgumentError factor_messages(fs, :x, nothing, msgs, fill(nothing, 3), p, NamedTuple())
end

# A row whose factors are not all equal gets one cluster per replica and is not batched; the
# replicas of equal factors share one cluster (`_as_clusters`, I19; from Mistral's probe p2,
# LP-034).
@testset "RegimeFactor plate: unequal factors in a row go replica by replica" begin
    rng = Xoshiro(21)
    dz, dx, K, N = 2, 1, 2, 3
    f, p = rp_factor(rng, dz, dx, K)
    fs = [RegimeFactor((z = dz, x = dx), K, dz; learn_noise = true) for _ in 1:N]
    cs = Lenticulum._as_clusters(fs)
    @test all(c -> c === cs[1], cs)
    fs2 = copy(fs)
    fs2[2] = RegimeFactor((z = dz, x = dx), K, dz; learn_noise = true, noise = 0.5)
    cs2 = Lenticulum._as_clusters(fs2)
    @test cs2[1] !== cs2[2] && !Mycelium._same_cluster(cs2[1], cs2[2])
    Λx = (S = randn(rng, dx, dx); S * S' + I)
    ins = [(z = DiracBelief(randn(rng, dz)), x = rp_gauss(rng, dx, Λx)) for _ in 1:N]
    @test !Mycelium._cluster_batchable(cs2, ins, :k, (factor = p,))
    for target in (:k, :x)
        ins_t = [NamedTuple{Tuple(setdiff((:z, :x), (target,)))}(i) for i in ins]
        mb, _ = factor_messages(fs2, target, nothing, ins_t, fill(nothing, N), p, NamedTuple())
        ms = [first(factor_message(fs2[i], target, nothing, ins_t[i], nothing, p, NamedTuple())) for i in 1:N]
        @test all(rp_close(mb[i], ms[i]; atol = 1e-12) for i in 1:N)
    end
    B = factor_beliefs(fs2, (z = belief_block([i.z for i in ins]), x = belief_block([i.x for i in ins])), p, NamedTuple())
    @test B isa Vector && length(B) == N
end

# I20: a plate sweep hands a cluster node's inputs as blocks; read from the blocks, the messages
# must equal the column path's exactly (the same arithmetic in the same order), whichever block
# comes out: a categorical block, a mixture block (one precision, no equal components), a
# Gaussian block (one assignment left), or the replicas stacked (several precisions, merged
# components, an assignment of weight zero, inputs whose shape differs between replicas).
@testset "RegimeFactor plate: block inputs give the column path's messages exactly (I20)" begin
    # the column path stacks mixtures with a precision per component into a vector block (I21)
    rp_same_kind(a, b) = typeof(a) == typeof(b) || (a isa ComponentMixtureBlock && b isa Mycelium.VectorBlock)
    rp_replicas_equal(a, b) = rp_same_kind(a, b) && nreplicas(a) == nreplicas(b) &&
        all(i -> rp_close(replica(a, i), replica(b, i); atol = 0.0), 1:nreplicas(a))
    kinds = Set{Symbol}()
    for seed in 1:6
        rng = Xoshiro(100 + seed)
        dz, dx, K, N = rand(rng, 1:3), rand(rng, 1:2), rand(rng, 2:3), 5
        f, p = rp_factor(rng, dz, dx, K)
        # one noise and one A_z for every outcome: the message to z has one precision, A_z' Q⁻¹ A_z
        p1 = merge(p, (L = repeat(p.L[:, :, 1:1], 1, 1, K), A = (z = repeat(p.A.z[:, :, 1:1], 1, 1, K), x = p.A.x)))
        p0 = merge(p1, (A = (z = repeat(p.A.z[:, :, 1:1], 1, 1, K), x = repeat(p.A.x[:, :, 1:1], 1, 1, K)),
                        b = repeat(p.b[:, 1:1], 1, K)))                            # equal outcomes: components merge
        fs = [RegimeFactor((z = dz, x = dx), K, dz; learn_noise = true) for _ in 1:N]
        Λx = (S = randn(rng, dx, dx); S * S' + I)
        Λz = (S = randn(rng, dz, dz); S * S' + I)
        lk = CategoricalBelief(Val(:log), (l = randn(rng, K); l .- log(sum(exp.(l)))), nothing)
        lz = CategoricalBelief(Val(:log), [-Inf; zeros(K - 1) .- log(K - 1)], nothing)  # outcome 1 impossible
        pts(d) = belief_block([DiracBelief(randn(rng, d)) for _ in 1:N])
        gss(d, Λ) = belief_block([rp_gauss(rng, d, Λ) for _ in 1:N])
        cases = [
            (:k, (z = pts(dz), x = gss(dx, Λx))),
            (:k, (z = pts(dz), x = pts(dx))),
            (:x, (z = pts(dz),)),
            (:x, (z = pts(dz), k = belief_block(fill(lk, N)))),
            (:z, (x = pts(dx), k = belief_block(fill(lk, N)))),
            (:z, (x = pts(dx), k = belief_block(fill(lz, N)))),
            (:z, (x = gss(dx, Λx),)),
            (:x, (z = gss(dz, Λz),)),
            (:z, (x = belief_block([rp_gauss(rng, dx, (S = randn(rng, dx, dx); S * S' + I)) for _ in 1:N]),)),  # precisions differ
            (:z, (x = pts(dx), k = belief_block([CategoricalBelief(Val(:log), (l = randn(rng, K); l .- log(sum(exp.(l)))), nothing)
                                                 for _ in 1:N])))]                                          # outcome messages differ
        for q in (p, p1, p0), (target, blocks) in cases
            mb, _ = factor_messages(fs, target, nothing, blocks, fill(nothing, N), q, NamedTuple())
            cols = [Mycelium._column(blocks, i) for i in 1:N]
            mc, _ = factor_messages(fs, target, nothing, cols, fill(nothing, N), q, NamedTuple())
            @test rp_replicas_equal(mb, belief_block(mc))
            push!(kinds, nameof(typeof(mb)))
            # the beliefs of a row, read from the blocks, equal those from the columns
            if target === :k
                B = factor_beliefs(fs, blocks, q, NamedTuple())
                cs = Lenticulum._as_clusters(fs)
                jt, _, pc, nA = Mycelium._batch_joint(cs, cols, (factor = q,), nothing)
                @test isequal((B.q, B.M, B.Σ, B.H, B.points), (Mycelium._belief_parts(jt, Dict{Any,Any}(), nA)..., pc))
            end
        end
        # with one precision and points in, the message to z is a mixture block, without per-replica beliefs
        mb, _ = factor_messages(fs, :z, nothing, (x = pts(dx), k = belief_block(fill(lk, N))), fill(nothing, N), p1, NamedTuple())
        @test mb isa MixtureBlock
    end
    @test kinds ⊇ Set([:CategoricalBlock, :MixtureBlock, :ComponentMixtureBlock, :GaussianBlock, :VectorBlock])
end

# I20: the data factors of a graph made a plate by `as_plate` each have their own (empty)
# parameter key; their messages and energies are grouped, and a key that is not empty sends the
# group back edge by edge.
@testset "RegimeFactor plate: untied data factors grouped, and edge by edge when a key is not empty (I20)" begin
    rng = Xoshiro(17)
    N = 25
    x = randn(rng, N)
    y = [rand(rng) < 0.5 ? x[i] + 0.2randn(rng) : -x[i] + 0.5randn(rng) for i in 1:N]
    a, c, σ, w = [0.8, -0.7], [0.1, 0.4], [0.3, 0.6], [0.5, 0.5]
    g = rp_mixreg_graph(x, y; plated = true)
    sched = plate_schedule(g)
    dplans = [p for p in sched.plans if p isa Mycelium._GroupPlan && eltype(p.factors) <: DataFactor]
    @test length(dplans) == 2 && all(p -> p.keys !== nothing && length(p.keys) == N, dplans)
    ps = rp_mixreg_ps(N, a, c, σ, w)
    s = MessageStore(g); sweep!(s, g, sched, ps, NamedTuple())
    drows = [r for r in Mycelium._energy_rows(s, g) if g.factors[first(r.fids)].factor isa DataFactor]
    @test length(drows) == 2 && all(r -> r.keys !== nothing, drows)
    F = first(scalar_free_energy(s, g, ps, NamedTuple()))
    @test F ≈ rp_mixreg_nlp(x, y, a, c, σ, w) rtol = 1e-12
    # a parameter subtree or a state under one data key: that group and that row go node by node
    k2 = replica_name(:dx, 2)
    psx = merge(ps, NamedTuple{(k2,)}(((u = [1.0],),)))
    stx = NamedTuple{(replica_name(:dy, 3),)}(((u = 1,),))
    @test !Mycelium._untied_empty(dplans[1].keys ∪ dplans[2].keys, Mycelium._indexed(psx), Mycelium._statebuffer(NamedTuple()))
    # (the per-edge path writes an empty state for each key it visits; the given entries stay)
    kept(st2, st) = all(k -> st2[k] == st[k], keys(st)) && all(k -> haskey(st, k) || Mycelium._empty_graph_state(st2[k]), keys(st2))
    for (q, st) in ((psx, NamedTuple()), (ps, stx))
        s2 = MessageStore(g)
        _, st2 = sweep!(s2, g, sched, q, st)
        @test kept(st2, st)
        @test all(e -> belief_distance(s2.to_variable[e].belief, s.to_variable[e].belief) == 0, 1:nedges(g))
        F2, st3 = scalar_free_energy(s2, g, q, st)
        @test F2 == F && kept(st3, st)
    end
end

# I21: a mixture block with a precision per component, what the regime factors of a plate send to a
# continuous channel once the outcomes' noises differ. Its operations against the per-belief rules.
@testset "ComponentMixtureBlock: columns, distance and products against the per-belief rules (I21)" begin
    rng = Xoshiro(41)
    for (d, K, n) in ((1, 2, 4), (2, 3, 5))
        spd() = (S = randn(rng, d, d); S * S' + I)
        Λs = [spd() for _ in 1:K]
        lw() = (l = randn(rng, K, n); l .- log.(sum(exp.(l); dims = 1)))
        a = ComponentMixtureBlock(randn(rng, d, K, n), Λs, lw(), fill(false, K))
        b = ComponentMixtureBlock(randn(rng, d, K, n), [Λ + 0.1I for Λ in Λs], lw(), [true; fill(false, K - 1)])
        @test nreplicas(a) == n
        r = replica(b, 2)
        @test r isa MixtureBelief && length(r.components) == K && r.components[1].improper && r.components[2].Λ === b.Λs[2]
        @test r.logw == b.logw[:, 2] && r.components[K].η == b.η[:, K, 2]
        @test Mycelium.block_distance(a, b) == maximum(i -> belief_distance(replica(a, i), replica(b, i)), 1:n)
        @test Mycelium.block_distance(a, a) == 0.0
        a2 = ComponentMixtureBlock(a.η, [Λ + 0.25I for Λ in Λs], a.logw, a.improper)   # only the precisions move
        @test Mycelium.block_distance(a, a2) == maximum(i -> belief_distance(replica(a, i), replica(a2, i)), 1:n) ≈ 0.25
        @test Mycelium.block_distance(a, ComponentMixtureBlock(randn(rng, d, K + 1, n), [Λs; [spd()]], randn(rng, K + 1, n), fill(false, K + 1))) == Inf
        p = belief_block([DiracBelief(randn(rng, d)) for _ in 1:n])
        for c in (Mycelium.combine_blocks(p, a), Mycelium.combine_blocks(a, p))
            @test c isa PointBlock && all(i -> replica(c, i).value == Mycelium.combine(replica(p, i), replica(a, i)).value, 1:n)
        end
        @test Mycelium._uninformative(a) == false
        # blocks of different sizes are refused, not truncated (Mistral's review of I21, LP-036)
        pn = belief_block([DiracBelief(randn(rng, d)) for _ in 1:n + 1])
        mb = MixtureBlock(randn(rng, d, K, n), Λs[1], lw(), false)
        for m in (a, mb)
            @test_throws DimensionMismatch Mycelium.combine_blocks(pn, m)
            @test_throws DimensionMismatch Mycelium.combine_blocks(m, pn)
        end
        @test_throws DimensionMismatch ComponentMixtureBlock(randn(rng, d, K, n), Λs[1:K-1], lw(), fill(false, K - 1))
        @test_throws DimensionMismatch ComponentMixtureBlock(randn(rng, d, K, n), Λs, randn(rng, K, n + 1), fill(false, K))
    end
    # one noise row for a two-dimensional target: each outcome's precision has rank one, so the
    # components are improper; per-outcome coefficients give each its own precision
    N, K = 4, 2
    fs = [RegimeFactor((z = 2, x = 1), K, 1; learn_noise = true) for _ in 1:N]
    p = (A = (z = randn(rng, 1, 2, K), x = randn(rng, 1, 1, K)), b = randn(rng, 1, K), logw = randn(rng, K),
         L = randn(rng, 1, 1, K))
    blocks = (x = belief_block([DiracBelief(randn(rng, 1)) for _ in 1:N]),)
    mb, _ = factor_messages(fs, :z, nothing, blocks, fill(nothing, N), p, NamedTuple())
    mc, _ = factor_messages(fs, :z, nothing, [Mycelium._column(blocks, i) for i in 1:N], fill(nothing, N), p, NamedTuple())
    @test mb isa ComponentMixtureBlock && all(mb.improper)
    @test all(i -> rp_close(replica(mb, i), mc[i]; atol = 0.0), 1:N)
end

# I21: the plate's energy rows are found once per graph and store layout. A second store of the
# same graph reuses them; a store whose block rows are numbered otherwise finds its own, and both
# give the closed form.
@testset "RegimeFactor plate: energy rows kept per store layout (I21)" begin
    rng = Xoshiro(23)
    N = 12
    x = randn(rng, N)
    y = [rand(rng) < 0.5 ? x[i] + 0.2randn(rng) : -x[i] + 0.5randn(rng) for i in 1:N]
    a, c, σ, w = [0.8, -0.7], [0.1, 0.4], [0.3, 0.6], [0.5, 0.5]
    g = rp_mixreg_graph(x, y; plated = true)
    ps = rp_mixreg_ps(N, a, c, σ, w)
    sched = plate_schedule(g)
    s1 = MessageStore(g); sweep!(s1, g, sched, ps, NamedTuple())
    s2 = MessageStore(g); sweep!(s2, g, sched, ps, NamedTuple())
    r1 = Mycelium._energy_rows(s1, g)
    @test Mycelium._energy_rows(s2, g) === r1                     # same layout: the kept rows
    rows = s1.to_factor.rows
    rev = reverse(rows)                                           # the same rows, numbered the other way
    s3 = MessageStore(Mycelium.EdgeMessages(nedges(g), rev), Mycelium.EdgeMessages(nedges(g), reverse(s1.to_variable.rows)), 0, Dict{Int,Any}())
    # swept edge by edge (a plate schedule's plans hold the row numbers of `MessageStore(g)`; on
    # such a store it runs without them since P22, tested below)
    sweep!(s3, g, SequentialSchedule(tasks(sched)), ps, NamedTuple())
    r3 = Mycelium._energy_rows(s3, g)
    @test r3 !== r1 && length(r3) == length(r1)
    @test all(r -> all(j -> s3.to_factor.rows[r.rows[j]] == [g.edges_of_factor[f][j] for f in r.fids], eachindex(r.rows)), r3)
    nlp = rp_mixreg_nlp(x, y, a, c, σ, w)
    for s in (s1, s3, s1)                                         # back and forth between layouts
        @test first(scalar_free_energy(s, g, ps, NamedTuple())) ≈ nlp rtol = 1e-12
    end
end

@testset "PlateSchedule on a store with another block numbering (P22)" begin
    # The plans of a plate schedule hold the block row numbers of `MessageStore(g)`. A store whose
    # rows are numbered otherwise (both directions, or only one) must still get the edge-by-edge
    # messages: before P22 the sweep wrote to the wrong rows and F was 2.68 against 14.54.
    rng = Xoshiro(23)
    N = 12
    x = randn(rng, N)
    y = [rand(rng) < 0.5 ? x[i] + 0.2randn(rng) : -x[i] + 0.5randn(rng) for i in 1:N]
    a, c, σ, w = [0.8, -0.7], [0.1, 0.4], [0.3, 0.6], [0.5, 0.5]
    g = rp_mixreg_graph(x, y; plated = true)
    ps = rp_mixreg_ps(N, a, c, σ, w)
    sched = plate_schedule(g)
    @test any(p -> p isa Mycelium._GroupPlan || p isa Mycelium._CavityPlan, sched.plans)   # the plans are used
    nlp = rp_mixreg_nlp(x, y, a, c, σ, w)
    ref = MessageStore(g); sweep!(ref, g, SequentialSchedule(tasks(sched)), ps, NamedTuple())
    rows = MessageStore(g).to_factor.rows
    layouts = ((reverse(rows), reverse(rows)),                    # both directions renumbered
               (reverse(rows), copy(rows)),                       # only the factor side
               (copy(rows), reverse(rows)),                       # only the variable side
               (copy(rows), copy(rows)))                          # the graph's numbering, other vectors
    for (tf, tv) in layouts
        s = MessageStore(Mycelium.EdgeMessages(nedges(g), tv), Mycelium.EdgeMessages(nedges(g), tf), 0, Dict{Int,Any}())
        @test Mycelium._graph_layout(s, g) == (tf == rows && tv == rows)   # the graph's numbering keeps the plans
        for _ in 1:2
            sweep!(s, g, sched, ps, NamedTuple())
        end
        @test first(scalar_free_energy(s, g, ps, NamedTuple())) ≈ nlp rtol = 1e-12
        same(a, b) = a === b === nothing || a !== nothing && b !== nothing && rp_close(a.belief, b.belief; atol = 1e-12)
        @test all(e -> same(s.to_variable[e], ref.to_variable[e]) && same(s.to_factor[e], ref.to_factor[e]), 1:nedges(g))
        tf == rows && tv == rows && @test any(b -> b !== nothing, s.to_variable.blocks)   # swept on blocks
    end
end
