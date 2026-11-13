# Flat directions inside cluster nodes (`cluster.md`, "Flat directions"): continuous directions
# that no factor and no incoming message determines are integrated out in the diffuse limit.
# Oracle: the same graph or message with a vague Gaussian prior of standard deviation s = 10⁴ on
# the flat variables, corrected by the factor that prior contributes (derived per case below);
# the remaining error is O(1/s²).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using MyceliumLearning
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote
using LenticulumCore: MixtureBelief, CategoricalBelief, DiracBelief, GaussianBelief

const FL_S = 1e4
fl_lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

# u and w enter only through u − w: z = u − w + ε observed, and a switch on u − w. The direction
# u + w is flat. With the vague priors N(0, s²) on u and w,
#   Z_s = ∫ N(u)N(w) L(u − w) = ∫ N(d; 0, 2s²) L(d) dd ≈ (4πs²)^{-1/2} ∫ L(d) dd,
# and the diffuse Z, on the range coordinate ρ = (u − w)/√2, is ∫ L(√2 ρ) dρ = 2^{-1/2} ∫ L(d) dd,
# so F = F_s − ½ log(2πs²).
function fl_graph(; vague = false)
    b = GraphBuilder()
    variable!(b, :z, 1); variable!(b, :u, 1); variable!(b, :w, 1); variable!(b, :k, 2)
    factor!(b, :data, DataFactor(:z, [0.8])); connect!(b, :data, :z, :z; direction = Emitting())
    factor!(b, :lc, LinearConstraintFactor((z = 1, u = 1, w = 1), 1; noise = 0.3))
    connect!(b, :lc, :z, :z); connect!(b, :lc, :u, :u); connect!(b, :lc, :w, :w)
    factor!(b, :sw, SwitchingConstraintFactor((u = 1, w = 1), 2, 1; noise = 0.2))
    connect!(b, :sw, :u, :u); connect!(b, :sw, :w, :w); connect!(b, :sw, :k, :k)
    fs = [:lc, :sw]
    if vague
        for v in (:u, :w)
            f = Symbol(:pr, v)
            factor!(b, f, GaussianPrior(v, 0.0, FL_S^2)); connect!(b, f, v, v); push!(fs, f)
        end
    end
    return cluster(build(b), :c, fs)
end
fl_ps(; b = [-1.0 1.5]) = (data = NamedTuple(), lc = (A = (z = ones(1, 1), u = -ones(1, 1), w = ones(1, 1)), c = [0.0]),
                           sw = (A = (u = ones(1, 1), w = -ones(1, 1)), b = b, logw = log.([0.3, 0.7])),
                           pru = NamedTuple(), prw = NamedTuple())

@testset "Flat directions: a cluster with an undetermined direction" begin
    g, gv = fl_graph(), fl_graph(vague = true)
    ps, psv = cluster_parameters(g, :c, fl_ps()), cluster_parameters(gv, :c, fl_ps())
    _, _, _, s = infer!(g, tree_schedule(g), ps, NamedTuple())
    _, _, _, sv = infer!(gv, tree_schedule(gv), psv, NamedTuple())
    F = first(scalar_free_energy(s, g, ps, NamedTuple()))
    Fv = first(scalar_free_energy(sv, gv, psv, NamedTuple()))
    @test F ≈ Fv - log(2π * FL_S^2) / 2 atol = 1e-6
    # closed form: d = u − w, z = d + ε and d = b_k + ε_k, so p(z) = Σ w_k N(z; b_k, 0.3 + 0.2)
    @test F ≈ -fl_lse([log(w) - (0.8 - bk)^2 / 1.0 - log(2π * 0.5) / 2 for (w, bk) in zip([0.3, 0.7], [-1.0, 1.5])]) + log(2) / 2 atol = 1e-12
    # the switch posterior is the vague one; u alone is flat (an improper marginal of zero precision)
    @test exp.(cluster_marginal(s, g, :c, :k, ps).logp) ≈ exp.(cluster_marginal(sv, gv, :c, :k, psv).logp) atol = 1e-8
    mu = cluster_marginal(s, g, :c, :u, ps)
    comps = mu isa MixtureBelief ? mu.components : [mu]
    @test all(c -> c isa GaussianBelief && !isproper(c) && norm(c.Λ) < 1e-10 && norm(c.η) < 1e-8, comps)
    # the gradient at the belief equals the vague one (the constant does not depend on θ)
    for ad in (AutoForwardDiff(), AutoZygote())
        G = free_energy_gradient(s, g, ps, NamedTuple(); ad).c
        Gv = free_energy_gradient(sv, gv, psv, NamedTuple(); ad).c
        @test isapprox(G.sw.b, Gv.sw.b; atol = 1e-6) && isapprox(G.sw.logw, Gv.sw.logw; atol = 1e-6)
        @test isapprox(G.lc.A.u, Gv.lc.A.u; atol = 1e-6)
    end
end

@testset "Flat directions: messages with flat inputs" begin
    # RegimeFactor with x ∈ R³ and m = 2: z = μ_k + W_k x + ε_k, x without a message, so one
    # direction of x (W_k's null space) is flat in each regime. Oracle: x with the vague message
    # N(0, s²I); the dropped factor (2πs²)^{-3/2} is common to the regimes and cancels in the
    # normalised message to k.
    W = [[1.0 0.3 -0.2; 0.0 1.0 0.5], [0.5 -1.0 0.0; 1.0 0.2 0.8]]
    Q = [Matrix(0.1I, 2, 2), [0.3 0.1; 0.1 0.2]]
    f = RegimeFactor((z = 2, x = 3), 2, 2; noise = Q)
    p = (A = (z = cat(Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2); dims = 3), x = cat(-W[1], -W[2]; dims = 3)),
         b = [0.0 1.0; 0.5 -1.0], logw = log.([0.4, 0.6]))
    z = DiracBelief([0.3, 0.2])
    mk, _ = factor_message(f, :k, nothing, (z = z,), nothing, p, NamedTuple())
    vague = GaussianBelief(zeros(3), Matrix(I / FL_S^2, 3, 3))
    mv, _ = factor_message(f, :k, nothing, (z = z, x = vague), nothing, p, NamedTuple())
    @test exp.(mk.logp) ≈ exp.(mv.logp) atol = 1e-7
    # the closed form: w_k ∫ N(z; μ_k + W_k x, Q_k) dx over the range of W_kᵀ is w_k / √det(W_k W_kᵀ)
    lk = [log(w) - logdet(Wk * Wk') / 2 for (w, Wk) in zip([0.4, 0.6], W)]
    @test exp.(mk.logp) ≈ exp.(lk .- fl_lse(lk)) atol = 1e-12
    # the message to z with x flat: in each regime z is unconstrained (W_k has full row rank)
    mz, _ = factor_message(f, :z, nothing, NamedTuple(), nothing, p, NamedTuple())
    @test mz isa LenticulumCore.TrivialBelief || all(c -> norm(c.Λ) < 1e-8, (mz isa MixtureBelief ? mz.components : [mz]))
    # a linear term along a flat direction diverges and is refused
    tilt = GaussianBelief([1.0, -2.0, 0.5], zeros(3, 3), true)
    @test_throws ArgumentError factor_message(f, :k, nothing, (z = z, x = tilt), nothing, p, NamedTuple())
    # regimes that leave different numbers of directions flat are refused: x ∈ R², one regime
    # reads x₁ − x₂ (one flat direction), the other both coordinates (none)
    f2 = RegimeFactor((z = 2, x = 2), 2, 2; noise = 0.1)
    p2 = (A = (z = cat(Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2); dims = 3), x = cat([-1.0 1.0; 0.0 0.0], -Matrix(1.0I, 2, 2); dims = 3)),
          b = zeros(2, 2), logw = zeros(2))
    @test_throws ArgumentError factor_message(f2, :k, nothing, (z = z,), nothing, p2, NamedTuple())
end

@testset "Flat directions: improper components with different precisions" begin
    # z = 1-d observed, x ∈ R², m = 1: each regime's message to x is the unnormalised exponential
    # w_k N(z + a_kᵀx − b_k; 0, q_k) of x, of rank one, a different precision per regime; Mycelium's
    # products read improper components as such exponentials, so their weights are the coefficients.
    f = RegimeFactor((z = 1, x = 2), 2, 1; noise = [fill(0.1, 1, 1), fill(0.3, 1, 1)])
    a = [[-1.0, 0.5], [0.3, -2.0]]; q = [0.1, 0.3]; bb = [0.0, 1.0]; z = 0.2
    p = (A = (z = ones(1, 1, 2), x = cat(a[1]', a[2]'; dims = 3)), b = reshape(bb, 1, 2), logw = zeros(2))
    m, _ = factor_message(f, :x, nothing, (z = DiracBelief([z]),), nothing, p, NamedTuple())
    @test m isa MixtureBelief && length(m.components) == 2 && all(c -> !isproper(c), m.components)
    for k in 1:2
        @test m.components[k].Λ ≈ a[k] * a[k]' / q[k] atol = 1e-12
        @test m.components[k].η ≈ a[k] * (bb[k] - z) / q[k] atol = 1e-12
    end
    lk = [-(bb[k] - z)^2 / (2q[k]) - log(2π * q[k]) / 2 for k in 1:2]
    @test exp.(m.logw) ≈ exp.(lk .- fl_lse(lk)) atol = 1e-12
end

# Flat directions in tied clusters over a plate (`cluster_batch.jl`): the batched messages,
# beliefs and free energies equal the per-replica ones, each replica's free energy is the closed
# form, and a linear term in the flat direction is refused by both paths (I19; from Mistral's
# probe p4, LP-034).
fl_close(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}; atol) = size(a) == size(b) && isapprox(a, b; atol, rtol = 0)
fl_close(a::Number, b::Number; atol) = isapprox(a, b; atol, rtol = 0)
fl_close(a::AbstractVector, b::AbstractVector; atol) = length(a) == length(b) && all(fl_close(x, y; atol) for (x, y) in zip(a, b))
fl_close(a::AbstractDict, b::AbstractDict; atol) = keys(a) == keys(b) && all(fl_close(a[k], b[k]; atol) for k in keys(a))
fl_close(a, b; atol) = typeof(a) == typeof(b) &&
    (isempty(fieldnames(typeof(a))) ? isequal(a, b) : all(f -> fl_close(getfield(a, f), getfield(b, f); atol), fieldnames(typeof(a))))

@testset "Flat directions: tied clusters over a plate, batched" begin
    rng = Xoshiro(5)
    N = 4
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:z), 1); variable!(b, r(:u), 1); variable!(b, r(:w), 1); variable!(b, r(:k), 2)
        factor!(b, r(:dz), DataFactor(:z, [randn(rng)])); connect!(b, r(:dz), :z, r(:z); direction = Emitting())
        factor!(b, r(:lc), LinearConstraintFactor((z = 1, u = 1, w = 1), 1; noise = 0.3); params = :lc)
        connect!(b, r(:lc), :z, r(:z)); connect!(b, r(:lc), :u, r(:u)); connect!(b, r(:lc), :w, r(:w))
        factor!(b, r(:sw), SwitchingConstraintFactor((u = 1, w = 1), 2, 1; noise = 0.2); params = :sw)
        connect!(b, r(:sw), :u, r(:u)); connect!(b, r(:sw), :w, r(:w)); connect!(b, r(:sw), :k, r(:k))
    end
    gc = cluster(as_plate(build(b), :pts, N), :pts, :c, [:lc, :sw]; params = :mix)
    p = (lc = (A = (z = ones(1, 1), u = -ones(1, 1), w = ones(1, 1)), c = [0.0]),
         sw = (A = (u = ones(1, 1), w = -ones(1, 1)), b = [-1.0 1.5], logw = log.([0.3, 0.7])))
    pl = only(plates(gc))
    row = findfirst(i -> gc.factors[pl.factors[i, 1]].factor isa ClusterFactor, axes(pl.factors, 1))
    cs = [gc.factors[pl.factors[row, i]].factor for i in 1:N]
    @test cs[1].boundary == [:z]
    zpts = [randn(rng) for _ in 1:N]
    ins = [(z = DiracBelief([z]),) for z in zpts]
    @test Mycelium._cluster_batchable(cs, ins, :z, p)
    mb, _ = factor_messages(cs, :z, nothing, ins, fill(nothing, N), p, NamedTuple())
    ms = [first(factor_message(cs[i], :z, nothing, ins[i], nothing, p, NamedTuple())) for i in 1:N]
    @test all(fl_close(mb[i], ms[i]; atol = 1e-9) for i in 1:N)
    msgs = (z = belief_block([DiracBelief([z]) for z in zpts]),)
    B = factor_beliefs(cs, msgs, p, NamedTuple())
    bs = [factor_belief(cs[i], Mycelium._column(msgs, i), p, NamedTuple()) for i in 1:N]
    @test B isa FrozenClusterBeliefs
    nA = length(bs[1].q)
    @test all(fl_close(B.q[:, r], bs[r].q; atol = 1e-10) && fl_close(B.M[:, ((r - 1) * nA + 1):(r * nA)], bs[r].M; atol = 1e-9) for r in 1:N)
    @test all(fl_close(B.Σ[κ], bs[1].Σ[κ]; atol = 1e-9) for κ in eachindex(B.Σ))
    Eb, _ = belief_free_energies(cs, B, p, NamedTuple())
    Es = [first(belief_free_energy(cs[i], bs[i], p, NamedTuple())) for i in 1:N]
    total(e) = e[:fit] + e[:complexity] + e[:negentropy]
    @test all(fl_close(collect(values(Eb[r].parts)), collect(values(Es[r].parts)); atol = 1e-9) for r in 1:N)
    # closed form: with d = u − w, z = d + ε (variance 0.3) and d = b_k + ε_k (variance 0.2), so
    # z ~ Σ_k w_k N(b_k, 0.5); the flat direction u + w drops out, and integrating on the range
    # coordinate (u − w)/√2 adds log 2 / 2 (derived at the top of this file)
    for (i, z) in enumerate(zpts)
        exact = -fl_lse([log(w) - (z - bk)^2 / (2 * 0.5) - log(2π * 0.5) / 2 for (w, bk) in zip([0.3, 0.7], [-1.0, 1.5])]) + log(2) / 2
        @test total(Es[i]) ≈ exact atol = 1e-10
    end
    # a linear term in the flat direction (an improper tilt on u, its mean differing per replica):
    # still batchable, and both paths refuse it
    tilts = [GaussianBelief([0.1 * i], zeros(1, 1), true) for i in 1:N]
    ins2 = [(z = DiracBelief([zpts[i]]), u = tilts[i]) for i in 1:N]
    eb = try; factor_messages(cs, :z, nothing, ins2, fill(nothing, N), p, NamedTuple()); nothing; catch e; e; end
    er = try; factor_message(cs[3], :z, nothing, ins2[3], nothing, p, NamedTuple()); nothing; catch e; e; end
    @test eb isa ArgumentError && er isa ArgumentError && occursin("flat direction", eb.msg)
end
