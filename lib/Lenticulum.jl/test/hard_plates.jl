# P6 (Feature Backlog): HardConstraintFactor in plates, and its refusal on MeanField() nodes.
#
# Oracle: `factor_message` of each replica on its own. The batched method shares replica 1's
# decompositions when the replicas' inputs share their covariances and flat directions; every
# replica's message (mean, covariance, flat span) must be the per-replica one, for every kind of
# input (points, nothing, proper and improper Gaussians of one precision, extended Gaussians with
# hard and flat directions), with and without (singular) noise, from per-replica inputs and from
# message blocks. Inputs that do not share, and dual numbers, take the per-replica path. A replica
# whose constraints contradict each other throws as it does alone. The graph-level oracle (a plate
# of hard relations against the conditioned dense Gaussian) is `extended_plates.jl`, which goes
# through this method.

using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, ForwardDiff, Zygote

module HardPlates
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, LuxCore

psd(rng, n, r) = (B = randn(rng, n, r); Matrix(Symmetric(B * B')))

# k replicas' inputs on a channel of dimension d, sharing everything but the mean
function inputs(rng, kind, d, k)
    if kind === :point
        return Any[DiracBelief(randn(rng, d)) for _ in 1:k]
    elseif kind === :none
        return Any[nothing for _ in 1:k]
    elseif kind === :trivial
        return Any[TrivialBelief() for _ in 1:k]
    elseif kind === :gauss
        Λ = psd(rng, d, d) + 0.5I
        return Any[GaussianBelief(randn(rng, d), Λ) for _ in 1:k]
    elseif kind === :improper                      # flat along null(Λ), η in range(Λ)
        r = rand(rng, 0:(d - 1))
        Λ = psd(rng, d, r)
        return Any[GaussianBelief(Λ * randn(rng, d), Λ, true) for _ in 1:k]
    else                                           # :extended: hard, Gaussian and flat directions
        nf = rand(rng, 0:(d - 1))
        D = randn(rng, d, nf)
        Σ = psd(rng, d, rand(rng, 0:d))
        e1 = ExtendedGaussian(randn(rng, d), Σ, D)
        return Any[ExtendedGaussian(randn(rng, d), e1.Σ, e1.D) for _ in 1:k]
    end
end

span(D) = D * D'

# max deviation between two extended Gaussians (their normalised representatives)
function gap(a, b)
    a, b = extended(a), extended(b)
    size(a.D, 2) == size(b.D, 2) || return Inf
    s = max(1.0, norm(b.μ), norm(b.Σ))
    return max(norm(a.μ - b.μ), norm(a.Σ - b.Σ), norm(span(a.D) - span(b.D))) / s
end

function instance(rng)
    names = (:a, :b, :t)[1:rand(rng, 2:3)]
    dims = NamedTuple{names}(Tuple(rand(rng, 1:3) for _ in names))
    m = rand(rng, 1:sum(values(dims)))
    noise = rand(rng, (0, :psd, :singular))
    Q = noise === 0 ? 0 : psd(rng, m, noise === :psd ? m : rand(rng, 0:(m - 1)))
    f = HardConstraintFactor(dims, m; noise = Q)
    ps = LuxCore.initialparameters(rng, f)
    return f, ps
end
end # module

const HP = HardPlates

@testset "P6: HardConstraintFactor plates, batched against per replica" begin
    rng = Xoshiro(606)
    kinds = (:point, :none, :trivial, :gauss, :improper, :extended)
    nshared = Ref(0); ntested = Ref(0); nthrow = Ref(0)
    worst = 0.0
    for trial in 1:300
        f, ps = HP.instance(rng)
        names = keys(f.dims)
        target = rand(rng, names)
        k = rand(rng, 2:5)
        ins = Dict(n => HP.inputs(rng, rand(rng, kinds), f.dims[n], k) for n in names if n !== target)
        cols = [NamedTuple{Tuple(keys(ins))}(Tuple(ins[n][i] for n in keys(ins))) for i in 1:k]
        fs = fill(f, k)
        ref = try
            [first(Mycelium.factor_message(f, target, nothing, cols[i], nothing, ps, NamedTuple())) for i in 1:k]
        catch e
            e isa ArgumentError || rethrow()
            nothing
        end
        if ref === nothing          # a replica the per-replica path refuses: the batch refuses too
            nthrow[] += 1
            @test_throws ArgumentError Mycelium.factor_messages(fs, target, nothing, cols, fill(nothing, k), ps, NamedTuple())
            continue
        end
        ntested[] += 1
        shared = Lenticulum._hard_shared_messages(fs, target, ps, k,
            n -> [haskey(c, n) ? getfield(c, n) : nothing for c in cols], Lenticulum._hard_channel_parts) !== nothing
        nshared[] += shared
        out, _ = Mycelium.factor_messages(fs, target, nothing, cols, fill(nothing, k), ps, NamedTuple())
        blocks = NamedTuple{Tuple(keys(ins))}(Tuple(Mycelium.belief_block(ins[n]) for n in keys(ins)))
        outb, _ = Mycelium.factor_messages(fs, target, nothing, blocks, fill(nothing, k), ps, NamedTuple())
        @test length(out) == k && Mycelium.nreplicas(outb) == k
        for i in 1:k
            g1, g2 = HP.gap(out[i], ref[i]), HP.gap(Mycelium.replica(outb, i), ref[i])
            worst = max(worst, g1, g2)
            @test g1 < 1e-8
            @test g2 < 1e-8
        end
    end
    @info "P6 batched vs per replica" ntested = ntested[] shared = nshared[] refused = nthrow[] worst
    @test nshared[] > 150           # the shared path is what is tested
end

@testset "P6: a contradiction in one replica" begin
    # t has dimension 1 and m = 2: the points of a and b must put c − A_a a − A_b b on the line A_t
    f = HardConstraintFactor((a = 1, b = 1, t = 1), 2)
    ps = (A = (a = [1.0; 0.0;;], b = [0.0; 1.0;;], t = [1.0; 1.0;;]), c = [0.0, 0.0])
    ok = (a = DiracBelief([1.0]), b = DiracBelief([1.0]))      # r = (−1, −1) = A_t · (−1)
    bad = (a = DiracBelief([1.0]), b = DiracBelief([2.0]))     # r = (−1, −2): no solution
    fs = fill(f, 3)
    @test_throws ArgumentError Mycelium.factor_message(f, :t, nothing, bad, nothing, ps, NamedTuple())
    @test_throws ArgumentError Mycelium.factor_messages(fs, :t, nothing, [ok, ok, bad], fill(nothing, 3), ps, NamedTuple())
    out, _ = Mycelium.factor_messages(fs, :t, nothing, [ok, ok, ok], fill(nothing, 3), ps, NamedTuple())
    @test all(b -> b.μ ≈ [-1.0] && size(b.D, 2) == 0 && iszero(b.Σ), out)
end

@testset "P6: what takes the per-replica path" begin
    rng = Xoshiro(7)
    f = HardConstraintFactor((u = 2, v = 2), 2)
    ps = LuxCore.initialparameters(rng, f)
    fs = fill(f, 3)
    via(cols) = Lenticulum._hard_shared_messages(fs, :v, ps, 3, n -> [getfield(c, n) for c in cols],
                                                 Lenticulum._hard_channel_parts)
    # different precisions per replica, and mixed kinds
    cols = [(u = GaussianBelief(randn(rng, 2), Matrix((1.0 + i) * I, 2, 2)),) for i in 1:3]
    @test via(cols) === nothing
    cols2 = [(u = DiracBelief(randn(rng, 2)),), (u = GaussianBelief(randn(rng, 2), Matrix(1.0I, 2, 2)),), (u = nothing,)]
    @test via(cols2) === nothing
    for c in (cols, cols2)
        out, _ = Mycelium.factor_messages(fs, :v, nothing, c, fill(nothing, 3), ps, NamedTuple())
        ref = [first(Mycelium.factor_message(f, :v, nothing, c[i], nothing, ps, NamedTuple())) for i in 1:3]
        @test all(i -> HP.gap(out[i], ref[i]) < 1e-12, 1:3)
    end
    # one replica: per replica
    @test Lenticulum._hard_shared_messages(fs[1:1], :v, ps, 1, n -> [DiracBelief(zeros(2))],
                                           Lenticulum._hard_channel_parts) === nothing
    # dual parameters: per replica, so the derivative is that of `factor_message`
    cols3 = [(u = DiracBelief(randn(rng, 2)),) for _ in 1:3]
    loss(cv) = begin
        q = (A = ps.A, c = cv)
        out, _ = Mycelium.factor_messages(fs, :v, nothing, cols3, fill(nothing, 3), q, NamedTuple())
        sum(b -> sum(b.μ), out)
    end
    loss1(cv) = begin
        q = (A = ps.A, c = cv)
        sum(i -> sum(first(Mycelium.factor_message(f, :v, nothing, cols3[i], nothing, q, NamedTuple())).μ), 1:3)
    end
    @test ForwardDiff.gradient(loss, ps.c) ≈ ForwardDiff.gradient(loss1, ps.c)
    @test loss(ps.c) ≈ loss1(ps.c)
    # under Zygote too: the batched path steps aside inside a reverse pass (`_within_gradient`)
    @test only(Zygote.gradient(loss, ps.c)) ≈ ForwardDiff.gradient(loss1, ps.c)
    cols4 = [(u = DiracBelief(randn(rng, 2)),) for _ in 1:3]
    lossA(Au) = begin
        q = (A = (u = Au, v = ps.A.v), c = ps.c)
        out, _ = Mycelium.factor_messages(fs, :v, nothing, cols4, fill(nothing, 3), q, NamedTuple())
        sum(b -> sum(abs2, b.μ), out)
    end
    lossA1(Au) = begin
        q = (A = (u = Au, v = ps.A.v), c = ps.c)
        sum(i -> sum(abs2, first(Mycelium.factor_message(f, :v, nothing, cols4[i], nothing, q, NamedTuple())).μ), 1:3)
    end
    @test only(Zygote.gradient(lossA, ps.A.u)) ≈ ForwardDiff.gradient(lossA1, ps.A.u)
end

@testset "P6: block inputs under Zygote, argument lengths (LP-044)" begin
    # LP-044 (1): the per-replica path from blocks went through `belief_block`, which Zygote cannot take
    rng = Xoshiro(41)
    f = HardConstraintFactor((a = 2, t = 2), 2)
    ps = LuxCore.initialparameters(rng, f)
    fs = fill(f, 3)
    B = randn(rng, 2, 2); Λ = Matrix(Symmetric(B * B' + 1e-3I))
    H0 = randn(rng, 2, 3)
    # by the parameter c, the block built outside the loss (Mycelium's GaussianBlock constructor is
    # itself not Zygote-safe; that is not this method's)
    blk = (a = Mycelium.GaussianBlock(H0, Λ, [false, false, false]),)
    lossb(cv) = begin
        out, _ = Mycelium.factor_messages(fs, :t, nothing, blk, fill(nothing, 3), (A = ps.A, c = cv), NamedTuple())
        sum(i -> sum(abs2, Mycelium.replica(out, i).μ), 1:3)
    end
    lossv(cv) = sum(i -> sum(abs2, first(Mycelium.factor_message(f, :t, nothing, (a = GaussianBelief(H0[:, i], Λ),), nothing,
                                                                  (A = ps.A, c = cv), NamedTuple())).μ), 1:3)
    @test lossb(ps.c) ≈ lossv(ps.c)
    @test only(Zygote.gradient(lossb, ps.c)) ≈ ForwardDiff.gradient(lossv, ps.c)
    # LP-044 (3): mismatched lengths are refused, on the shared path and on the per-replica one
    cols = [(a = DiracBelief(randn(rng, 2)),) for _ in 1:3]
    @test_throws DimensionMismatch Mycelium.factor_messages(fs, :t, nothing, cols, fill(nothing, 2), ps, NamedTuple())
    @test_throws DimensionMismatch Mycelium.factor_messages(fs[1:2], :t, nothing, cols, fill(nothing, 3), ps, NamedTuple())
end

@testset "P6: MeanField() nodes, LinearConstraintFactor's with a positive definite noise (LP-044)" begin
    rng = Xoshiro(44)
    for trial in 1:20
        dims = (u = rand(rng, 1:3), v = rand(rng, 1:3))
        m = rand(rng, 1:3)
        B = randn(rng, m, m); Q = Matrix(Symmetric(B * B' + 0.1I))
        h = HardConstraintFactor(dims, m; noise = Q)
        l = LinearConstraintFactor(dims, m; noise = Q)
        ps = LuxCore.initialparameters(rng, h)
        q = (v = GaussianBelief(randn(rng, dims.v), Matrix(1.0I, dims.v, dims.v)),)
        mh, _ = Mycelium.mean_field_message(h, :u, nothing, q, ps, NamedTuple())
        ml, _ = Mycelium.mean_field_message(l, :u, nothing, q, ps, NamedTuple())
        @test HP.gap(mh, ml) < 1e-12
        marg = (u = GaussianBelief(randn(rng, dims.u), Matrix(2.0I, dims.u, dims.u)), v = q.v)
        Fh, _ = Mycelium.mean_field_energy(h, marg, ps, NamedTuple())
        Fl, _ = Mycelium.mean_field_energy(l, marg, ps, NamedTuple())
        @test all(k -> Fh[k] ≈ Fl[k], (:fit, :complexity, :negentropy))
    end
end

@testset "P6: MeanField() nodes refuse an exact relation, with the reason" begin
    # a singular nonzero noise is exact along its null space: refused too
    fsing = HardConstraintFactor((u = 1, v = 1), 2; noise = [1.0 0.0; 0.0 0.0])
    psing = (A = (u = ones(2, 1), v = ones(2, 1)), c = zeros(2))
    @test_throws ArgumentError Mycelium.mean_field_message(fsing, :u, nothing, (v = DiracBelief([1.0]),), psing, NamedTuple())
    f = HardConstraintFactor((u = 1, v = 1), 1)
    ps = (A = (u = ones(1, 1), v = ones(1, 1)), c = [0.0])
    err = try
        Mycelium.mean_field_message(f, :u, nothing, (v = DiracBelief([1.0]),), ps, NamedTuple()); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("BeliefPropagation()", err.msg) && occursin("prior", err.msg)
    @test_throws ArgumentError Mycelium.mean_field_energy(f, (u = DiracBelief([1.0]), v = DiracBelief([-1.0])), ps, NamedTuple())
end
