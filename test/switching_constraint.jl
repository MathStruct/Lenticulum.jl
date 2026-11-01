using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, CategoricalBelief, MixtureBelief
isdefined(@__MODULE__, :check_factor_conformance) || include(joinpath(@__DIR__, "factor_conformance.jl"))

# Independent oracles: plain Gaussian conditioning, outcome by outcome.
lognormal(x, μ, Σ) = -(dot(x - μ, Σ \ (x - μ)) + logdet(2π * Σ)) / 2
logsumexp_(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
softmax_(v) = exp.(v .- logsumexp_(v))

@testset "SwitchingConstraintFactor: the interface" begin
    f = SwitchingConstraintFactor((a = 2, b = 3), 4, 2; noise = 0.5)
    ps, st = LuxCore.setup(Xoshiro(1), f)
    @test size(ps.A.a) == (2, 2) && size(ps.A.b) == (2, 3) && size(ps.b) == (2, 4) && length(ps.logw) == 4
    @test LuxCore.parameterlength(f) == 2 * 5 + 2 * 4 + 4
    @test map(LenticulumCore.channelname, LenticulumCore.channels(f)) == (:a, :b, :k)
    pols = supported_polarities(f)
    @test length(pols) == 3
    @test Set(only(unobserved_channels(p)) for p in pols) == Set((:a, :b, :k))
    @test supports_polarity(f, LenticulumCore.Polarity(; a = Unobserved(), b = Observed(), k = Latent()))
    @test !supports_polarity(f, LenticulumCore.Polarity(; a = Unobserved(), b = Unobserved(), k = Observed()))
    @test !supports_polarity(f, LenticulumCore.Polarity(; a = Unobserved(), b = Observed()))
    @test_throws ArgumentError SwitchingConstraintFactor((k = 1,), 2, 1)
    @test residual(f, (a = [1.0, 0.0], b = zeros(3), k = 3), ps) ≈ ps.A.a * [1.0, 0.0] - ps.b[:, 3]
end

@testset "SwitchingConstraintFactor with one outcome is a LinearConstraintFactor" begin
    rng = Xoshiro(2)
    dims = (x = 2, y = 1, z = 2)
    Q = [0.4 0.1; 0.1 0.3]
    sf = SwitchingConstraintFactor(dims, 1, 2; noise = Q)
    lf = LinearConstraintFactor(dims, 2; noise = Q)
    A = (x = randn(rng, 2, 2), y = randn(rng, 2, 1), z = randn(rng, 2, 2))
    c = randn(rng, 2)
    sps, lps = (A = A, b = reshape(c, 2, 1), logw = [0.3]), (A = A, c = c)
    beliefs = (x = Gaussian([0.5, -1.0], [1.0 0.2; 0.2 0.5]), y = DiracBelief([0.7]), z = Gaussian([1.0, 0.0], 0.3 * Matrix(1.0I, 2, 2)))
    for t in (:x, :y, :z)
        inputs = NamedTuple{Tuple(n for n in keys(dims) if n !== t)}(Tuple(beliefs[n] for n in keys(dims) if n !== t))
        ms, _ = Mycelium.factor_message(sf, t, nothing, inputs, nothing, sps, NamedTuple())
        ml, _ = Mycelium.factor_message(lf, t, nothing, inputs, nothing, lps, NamedTuple())
        @test ms.η ≈ ml.η && ms.Λ ≈ ml.Λ
    end
    Fs, _ = Mycelium.local_free_energy(sf, beliefs, sps, NamedTuple())
    Fl, _ = Mycelium.local_free_energy(lf, beliefs, lps, NamedTuple())
    @test Fs[:fit] ≈ Fl[:fit] && Fs[:complexity] ≈ Fl[:complexity] && Fs[:negentropy] ≈ Fl[:negentropy]
end

@testset "SwitchingConstraintFactor messages against Gaussian conditioning" begin
    rng = Xoshiro(3)
    K, Q = 3, [0.3 0.05; 0.05 0.2]
    f = SwitchingConstraintFactor((z = 2, x = 2), K, 2; noise = Q)
    ps = (A = (z = randn(rng, 2, 2), x = randn(rng, 2, 2)), b = randn(rng, 2, K), logw = randn(rng, K))
    w = softmax_(ps.logw)
    μ0, Σ0 = [0.3, -0.2], [0.8 0.3; 0.3 0.6]
    π0 = Gaussian(μ0, Σ0)
    x = [0.4, 1.1]
    ρ = [0.2, 0.5, 0.3]
    # oracle: z ~ N(μ0, Σ0), y_k = b_k − A_x x = A_z z + ε
    y = [ps.b[:, k] - ps.A.x * x for k in 1:K]
    S = ps.A.z * Σ0 * ps.A.z' + Q
    G = Σ0 * ps.A.z' / S
    post_μ = [μ0 + G * (y[k] - ps.A.z * μ0) for k in 1:K]
    post_Σ = Σ0 - G * ps.A.z * Σ0
    lev = [log(ρ[k] * w[k]) + lognormal(y[k], ps.A.z * μ0, S) for k in 1:K]

    # to the outcome, with z marginalised under π0 as a message
    mk, _ = Mycelium.factor_message(f, :k, nothing, (z = π0, x = DiracBelief(x)), nothing, ps, NamedTuple())
    @test probabilities(mk) ≈ softmax_([log(w[k]) + lognormal(y[k], ps.A.z * μ0, S) for k in 1:K])
    # to z: pooled with the prior it is the exact posterior mixture
    mz, _ = Mycelium.factor_message(f, :z, nothing, (x = DiracBelief(x), k = CategoricalBelief(ρ)), nothing, ps, NamedTuple())
    @test mz isa MixtureBelief
    post = Mycelium.combine(π0, mz)
    @test mixture_weights(post) ≈ softmax_(lev)
    for k in 1:K
        @test belief_mean(post.components[k]) ≈ post_μ[k]
        @test belief_cov(post.components[k]) ≈ post_Σ
    end
    # an observed outcome gives one Gaussian
    m1, _ = Mycelium.factor_message(f, :z, nothing, (x = DiracBelief(x), k = DiracBelief(2)), nothing, ps, NamedTuple())
    @test m1 isa GaussianBelief
    @test belief_mean(Mycelium.combine(π0, m1)) ≈ post_μ[2]

    # an improper message: one equation, a 2-D target
    g = SwitchingConstraintFactor((z = 2, x = 1), K, 1; noise = 0.2)
    gps = (A = (z = randn(rng, 1, 2), x = randn(rng, 1, 1)), b = randn(rng, 1, K), logw = randn(rng, K))
    wg = softmax_(gps.logw)
    mz, _ = Mycelium.factor_message(g, :z, nothing, (x = DiracBelief([0.5]),), nothing, gps, NamedTuple())
    @test !isproper(first(mz.components))
    post = Mycelium.combine(π0, mz)
    yg = [gps.b[:, k] - gps.A.x * [0.5] for k in 1:K]
    Sg = gps.A.z * Σ0 * gps.A.z' + 0.2 * I
    Gg = Σ0 * gps.A.z' / Sg
    @test mixture_weights(post) ≈ softmax_([log(wg[k]) + lognormal(yg[k], gps.A.z * μ0, Sg) for k in 1:K])
    for k in 1:K
        @test belief_mean(post.components[k]) ≈ μ0 + Gg * (yg[k] - gps.A.z * μ0)
    end

    # the local free energy: F_c = E_b[log π0(z) + log ρ(k)] − log Z, with b ∝ f π0 ρ
    F, _ = Mycelium.local_free_energy(f, (z = π0, x = DiracBelief(x), k = CategoricalBelief(ρ)), ps, NamedTuple())
    q = softmax_(lev)
    Elogπ = sum(q[k] * (-(dot(post_μ[k] - μ0, Σ0 \ (post_μ[k] - μ0)) + tr(Σ0 \ post_Σ) + logdet(2π * Σ0)) / 2) for k in 1:K)
    @test scalarise(scalarisation(f), F) ≈ Elogπ + sum(q .* log.(ρ)) - logsumexp_(lev)
    q2, _, _, _ = switch_statistics(f, (z = π0, x = DiracBelief(x), k = CategoricalBelief(ρ)), ps)
    @test q2 ≈ q
end

@testset "SwitchingConstraintFactor in a graph: the Bethe free energy is −log p" begin
    # k ~ categorical prior; z observed through z = b_k + ε. The marginal of z is a Gaussian mixture.
    K = 3
    f = SwitchingConstraintFactor((z = 2,), K, 2; noise = [0.3 0.1; 0.1 0.2])
    ps_sw = (A = (z = Matrix(1.0I, 2, 2),), b = [-1.0 1.0 0.0; 0.0 0.5 2.0], logw = log.([0.2, 0.3, 0.5]))
    prior = [0.5, 0.25, 0.25]
    z = [0.4, 0.9]
    b = GraphBuilder()
    variable!(b, :z, 2); variable!(b, :k, K)
    factor!(b, :data, DataFactor(:z, z))
    factor!(b, :sw, f)
    factor!(b, :pk, PriorFactor(:k, CategoricalBelief(prior), k -> -log(prior[k])))
    connect!(b, :data, :z, :z; direction = Emitting())
    connect!(b, :sw, :z, :z; direction = Absorbing())
    connect!(b, :sw, :k, :k); connect!(b, :pk, :k, :k)
    g = build(b)
    ps = merge(LuxCore.initialparameters(Xoshiro(0), g), (sw = ps_sw,))
    marg, report, st, store = infer!(g, tree_schedule(g), ps, LuxCore.initialstates(Xoshiro(0), g))
    w = softmax_(ps_sw.logw)
    lj = [log(prior[k] * w[k]) + lognormal(z, ps_sw.b[:, k], f.Q) for k in 1:K]
    @test probabilities(marg.k) ≈ softmax_(lj)
    total, _ = LenticulumCore.scalar_free_energy(store, g, ps, st)
    @test total ≈ -logsumexp_(lj)
end

@testset "SwitchingConstraintFactor conformance (L-SPEC-001 §1–§5)" begin
    nt = NamedTuple()
    f = SwitchingConstraintFactor((x = 1, y = 1), 2, 1; noise = 0.2)
    ps = (A = (x = fill(-2.0, 1, 1), y = fill(1.0, 1, 1)), b = [0.0 1.5], logw = [0.0, 0.4])
    px = Gaussian([1.0], fill(1.0, 1, 1))
    ρ = CategoricalBelief([0.4, 0.6])
    for p in supported_polarities(f)
        t = only(unobserved_channels(p))
        cavity = t === :k ? ρ : px
        inputs = t === :x ? (y = DiracBelief([4.0]), k = ρ) : t === :y ? (x = DiracBelief([1.0]), k = ρ) :
                 (x = DiracBelief([1.0]), y = DiracBelief([4.0]))
        msgs = t === :k ? (x = DiracBelief([1.0]), y = DiracBelief([4.0]), k = ρ) : merge(inputs, NamedTuple{(t,)}((px,)))
        game_y = Base.structdiff(msgs, NamedTuple{(t,)})
        check_factor_conformance(f, p; π = cavity, inputs, ps, st = nt,
                                 pointwise = ((x = [1.0], y = [4.0], k = 2), nothing, nothing),
                                 game_prior = cavity, game_y, msgs)
    end
end

@testset "Mean-field messages of the linear and switching constraints" begin
    rng = Xoshiro(5)
    Q = [0.3 0.05; 0.05 0.2]
    # LinearConstraintFactor: the BP message with every input at its mean
    lf = LinearConstraintFactor((x = 2, y = 2, z = 2), 2; noise = Q)
    lps = (A = (x = randn(rng, 2, 2), y = randn(rng, 2, 2), z = randn(rng, 2, 2)), c = randn(rng, 2))
    qy = Gaussian([0.3, -0.4], [0.5 0.1; 0.1 0.4])
    qz = MixtureBelief([Gaussian([1.0, 0.0], 0.2 * Matrix(1.0I, 2, 2)), Gaussian([-1.0, 2.0], 0.3 * Matrix(1.0I, 2, 2))], [0.3, 0.7])
    mz = 0.3 * [1.0, 0.0] + 0.7 * [-1.0, 2.0]
    m, _ = Mycelium.mean_field_message(lf, :x, nothing, (y = qy, z = qz), lps, NamedTuple())
    d = lps.c - lps.A.y * [0.3, -0.4] - lps.A.z * mz
    @test m.η ≈ lps.A.x' * (Q \ d) && m.Λ ≈ lps.A.x' * (Q \ lps.A.x)
    # SwitchingConstraintFactor: E_q over the outcome and the other channels' means
    K = 3
    sf = SwitchingConstraintFactor((x = 2, y = 2), K, 2; noise = Q)
    sps = (A = (x = randn(rng, 2, 2), y = randn(rng, 2, 2)), b = randn(rng, 2, K), logw = randn(rng, K))
    qk = CategoricalBelief([0.2, 0.5, 0.3])
    mx, _ = Mycelium.mean_field_message(sf, :x, nothing, (y = qy, k = qk), sps, NamedTuple())
    bbar = sps.b * [0.2, 0.5, 0.3]
    @test mx.η ≈ sps.A.x' * (Q \ (bbar - sps.A.y * [0.3, -0.4])) && mx.Λ ≈ sps.A.x' * (Q \ sps.A.x)
    mk, _ = Mycelium.mean_field_message(sf, :k, nothing, (x = DiracBelief([0.1, 0.2]), y = qy), sps, NamedTuple())
    sbar = sps.A.x * [0.1, 0.2] + sps.A.y * [0.3, -0.4]
    @test probabilities(mk) ≈ softmax_([sps.logw[k] - dot(sps.b[:, k] - sbar, Q \ (sps.b[:, k] - sbar)) / 2 for k in 1:K])
    # E_q[log f] as a function of x_t, by Monte Carlo over q, has the message's slope and curvature
    xs = [randn(rng, 2) for _ in 1:3]
    logf(x, y, k) = sps.logw[k] - dot(sps.A.x * x + sps.A.y * y - sps.b[:, k], Q \ (sps.A.x * x + sps.A.y * y - sps.b[:, k])) / 2
    pk = [0.2, 0.5, 0.3]; Ly = cholesky(Symmetric([0.5 0.1; 0.1 0.4])).L
    ys = [[0.3, -0.4] + Ly * randn(rng, 2) for _ in 1:20_000]
    Elogf(x) = sum(pk[k] * sum(logf(x, y, k) for y in ys) / length(ys) for k in 1:K)
    lm(x) = dot(mx.η, x) - dot(x, mx.Λ * x) / 2
    @test (Elogf(xs[1]) - Elogf(xs[2])) ≈ (lm(xs[1]) - lm(xs[2])) rtol = 1e-2
    @test (Elogf(xs[1]) - Elogf(xs[3])) ≈ (lm(xs[1]) - lm(xs[3])) rtol = 1e-2
end

@testset "A mixture input to a linear factor is collapsed to its moments (src/gaussian.jl)" begin
    f = LinearConstraintFactor((a = 2, b = 2), 2; noise = 0.3)
    ps = LuxCore.initialparameters(Xoshiro(1), f)
    mix = MixtureBelief([Gaussian([1.0, 0.0], 0.2 * Matrix(1.0I, 2, 2)), Gaussian([-1.0, 2.0], [0.5 0.1; 0.1 0.4])], [0.3, 0.7])
    m1, _ = Mycelium.factor_message(f, :b, nothing, (a = mix,), nothing, ps, NamedTuple())
    m2, _ = Mycelium.factor_message(f, :b, nothing, (a = Lenticulum.moment_match(mix),), nothing, ps, NamedTuple())
    @test m1.η ≈ m2.η && m1.Λ ≈ m2.Λ
end

@testset "A hybrid graph (MeanField sum node, switching blocks on BP) is structured mean field" begin
    # z = c + Σ_l k_l δ_l + ε with four binary switches in two blocks; each block is one
    # SwitchingConstraintFactor with 4 outcomes, feeding the sum node z + u′₁ + u′₂ = c.
    rng = Xoshiro(7)
    d, L = 3, 4
    c = randn(rng, d); Δ = 2.0 .* randn(rng, d, L); p = [0.3, 0.6, 0.5, 0.4]; Σ = 0.3 * Matrix(1.0I, d, d)
    blocks = [[1, 3], [2, 4]]
    K = reduce(hcat, [digits(i; base = 2, pad = 2) for i in 0:3])                 # 2 × 4 configurations
    logprior(bl) = vec(log.(p[bl])' * K .+ log1p.(-p[bl])' * (1 .- K))
    # the oracle: block coordinate ascent on the structured mean-field bound, from the prior means
    function structured(x)
        m = copy(p); P = [zeros(4), zeros(4)]
        for _ in 1:1000
            old = copy(m)
            for (b, bl) in enumerate(blocks)
                o = setdiff(1:L, bl); U = Δ[:, bl] * K
                P[b] = softmax_(logprior(bl) .+ vec(U' * (Σ \ (x - c - Δ[:, o] * m[o]))) .- vec(sum(U .* (Σ \ U); dims = 1)) ./ 2)
                m[bl] = K * P[b]
            end
            maximum(abs.(m - old)) < 1e-14 && break
        end
        return P
    end
    function graph(x, rule; ε = 1e-8)
        b = GraphBuilder()
        variable!(b, :z, d); variable!(b, :u1, d); variable!(b, :u2, d); variable!(b, :k1, 4); variable!(b, :k2, 4)
        factor!(b, :data, DataFactor(:z, x))
        factor!(b, :sum, LinearConstraintFactor((z = d, u1 = d, u2 = d), d; noise = Σ); rule)
        connect!(b, :data, :z, :z; direction = Emitting()); connect!(b, :sum, :z, :z; direction = Absorbing())
        for i in 1:2
            factor!(b, Symbol(:sw, i), SwitchingConstraintFactor((u = d,), 4, d; noise = ε))
            connect!(b, Symbol(:sw, i), :u, Symbol(:u, i)); connect!(b, :sum, Symbol(:u, i), Symbol(:u, i))
            connect!(b, Symbol(:sw, i), :k, Symbol(:k, i))
        end
        g = build(b)
        I3 = Matrix(1.0I, d, d)
        ps = merge(LuxCore.initialparameters(Xoshiro(0), g),
                   (sum = (A = (z = I3, u1 = I3, u2 = I3), c = c),
                    sw1 = (A = (u = I3,), b = -Δ[:, blocks[1]] * K, logw = logprior(blocks[1])),
                    sw2 = (A = (u = I3,), b = -Δ[:, blocks[2]] * K, logw = logprior(blocks[2]))))
        return g, ps
    end
    edge(g, f, ch) = findfirst(e -> e.factor == g.factor_index[f] && e.channel === ch, g.edges)
    function hybrid(x)
        g, ps = graph(x, MeanField())
        T = MessageTask
        init = [T(edge(g, :data, :z), :to_variable), T(edge(g, :sw1, :u), :to_variable), T(edge(g, :sw2, :u), :to_variable)]
        sweep = [T(edge(g, :sum, :u1), :to_variable), T(edge(g, :sum, :u2), :to_variable),
                 T(edge(g, :sw1, :u), :to_factor), T(edge(g, :sw1, :k), :to_variable),
                 T(edge(g, :sw2, :u), :to_factor), T(edge(g, :sw2, :k), :to_variable)]
        s = MessageStore(g); st = NamedTuple()
        sweep!(s, g, SequentialSchedule(init), ps, st)
        report, _ = propagate!(s, g, SequentialSchedule(sweep), ps, st; maxsweeps = 500, tol = 1e-6)
        return [probabilities(marginal(s, g, g.variable_index[k])) for k in (:k1, :k2)], report
    end
    function bp(x)
        g, ps = graph(x, BeliefPropagation())
        marg, _, _, _ = infer!(g, tree_schedule(g), ps, NamedTuple())
        return [probabilities(marg.k1), probabilities(marg.k2)]
    end
    X = [c + Δ * (rand(rng, L) .< p) + sqrt(0.3) * randn(rng, d) for _ in 1:8]
    for x in X
        P, report = hybrid(x)
        @test report.converged
        @test maximum(maximum(abs.(P[b] - structured(x)[b])) for b in 1:2) < 1e-6
    end
    # belief propagation at the sum node is a different approximation (prior moments of the other block)
    @test maximum(maximum(maximum(abs.(bp(x)[b] - first(hybrid(x))[b])) for b in 1:2) for x in X) > 1e-3
    # and the free energy refuses the hybrid rather than misreport it
    g, ps = graph(X[1], MeanField())
    _, _, _, s = infer!(g, SequentialSchedule(all_tasks(g)), ps, NamedTuple(); maxsweeps = 3)
    @test_throws ArgumentError LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple())
end

@testset "Gibbs sampling over switches, the sample buffer, and the Monte Carlo EM gradient" begin
    # z = c + u₁ + u₂ + ε, each u_l a binary switch with offsets {0, δ_l} and width η; the
    # switches overlap, so their posterior is correlated. Exact: p(x, k) = w₁(k₁) w₂(k₂) N(x; c + Σ k_l δ_l, Σ + 2ηI).
    d, η = 2, 1e-3
    Σ = [0.3 0.05; 0.05 0.25]; x = [1.1, 0.7]
    I2 = Matrix(1.0I, d, d)
    function graph()
        b = GraphBuilder()
        variable!(b, :z, d); variable!(b, :u1, d); variable!(b, :u2, d); variable!(b, :k1, 2); variable!(b, :k2, 2)
        factor!(b, :data, DataFactor(:z, x))
        factor!(b, :sum, LinearConstraintFactor((z = d, u1 = d, u2 = d), d; noise = Σ))
        connect!(b, :data, :z, :z; direction = Emitting()); connect!(b, :sum, :z, :z; direction = Absorbing())
        for l in 1:2
            factor!(b, Symbol(:sw, l), SwitchingConstraintFactor((u = d,), 2, d; noise = η))
            connect!(b, Symbol(:sw, l), :u, Symbol(:u, l)); connect!(b, :sum, Symbol(:u, l), Symbol(:u, l))
            connect!(b, Symbol(:sw, l), :k, Symbol(:k, l))
        end
        return build(b)
    end
    g = graph()
    params(c, δ1, δ2, lw1, lw2) = merge(LuxCore.initialparameters(Xoshiro(0), g),
        (sum = (A = (z = I2, u1 = -I2, u2 = -I2), c = c),
         sw1 = (A = (u = I2,), b = [zeros(d) δ1], logw = lw1), sw2 = (A = (u = I2,), b = [zeros(d) δ2], logw = lw2)))
    c0, δ1, δ2, lw1, lw2 = [0.1, -0.2], [2.0, 0.0], [1.6, 1.2], log.([0.6, 0.4]), log.([0.5, 0.5])
    ps = params(c0, δ1, δ2, lw1, lw2)
    configs = [(a, b) for a in 1:2 for b in 1:2]
    function logjoint(c, δ1, δ2, lw1, lw2, (a, b))
        m = c + (a - 1) * δ1 + (b - 1) * δ2
        lw1[a] - logsumexp_(lw1) + lw2[b] - logsumexp_(lw2) + lognormal(x, m, Σ + 2η * I2)
    end
    lj = [logjoint(c0, δ1, δ2, lw1, lw2, k) for k in configs]
    P = softmax_(lj)

    # the sampler's long-run frequencies are the exact posterior
    sampler = GibbsSampler(g, (:k1, :k2))
    rng = Xoshiro(11)
    state, _ = gibbs_init(sampler, ps, NamedTuple(); rng)
    counts = zeros(4)
    for _ in 1:4000
        gibbs_sweep!(state, sampler, ps, NamedTuple(); rng)
        pt = gibbs_points(state, sampler)
        counts[findfirst(==((pt.k1, pt.k2)), configs)] += 1
    end
    @test maximum(abs.(counts ./ 4000 - P)) < 0.03

    # a buffer holding every configuration at its exact posterior weight
    buf = SampleBuffer(WindowFilter(4))
    for (k, p) in zip(configs, P)
        record!(buf, sampler, (k1 = k[1], k2 = k[2]), ps, NamedTuple(); weight = p)
    end
    @test [e.total for e in buf.entries] ≈ -lj                  # each total is −log p(x, pins)
    @test free_energy_mean(buf) ≈ -sum(P .* lj)
    # residual moments of the sum node: law of total covariance against Gaussian conditioning
    rmean, rcov = residual_moments(buf, :sum)
    rs, Cs = map(configs) do (a, b)
        sprior = (a - 1) * δ1 + (b - 1) * δ2                    # u₁ + u₂ ~ N(sprior, 2ηI) a priori
        Λ = inv(2η * I2) + inv(Σ)
        S = inv(Λ); ms = S * ((2η * I2) \ sprior + Σ \ (x - c0))
        x - c0 - ms, S                                          # r = z − u₁ − u₂ − c
    end |> v -> (first.(v), last.(v))
    r̄ = sum(P .* rs)
    @test rmean ≈ r̄ atol = 1e-8
    @test rcov ≈ sum(P[i] * (Cs[i] + (rs[i] - r̄) * (rs[i] - r̄)') for i in 1:4) atol = 1e-8
    # the gradient: with exact posterior weights it is the gradient of the exact −log p(x)
    grad = sample_gradient(buf, sampler, ps, NamedTuple())
    nlp(c, δ1, δ2, lw1, lw2) = -logsumexp_([logjoint(c, δ1, δ2, lw1, lw2, k) for k in configs])
    h = 1e-6
    fd(f, v) = [(f(v + h * e) - f(v - h * e)) / 2h for e in eachcol(Matrix(1.0I, length(v), length(v)))]
    @test grad.sum.c ≈ fd(c -> nlp(c, δ1, δ2, lw1, lw2), c0) rtol = 1e-4
    @test grad.sw1.b[:, 2] ≈ fd(v -> nlp(c0, v, δ2, lw1, lw2), δ1) rtol = 1e-4   # b's second column is δ₁
    @test grad.sw2.logw ≈ fd(v -> nlp(c0, δ1, δ2, lw1, v), lw2) rtol = 1e-4 atol = 1e-8
    # the default (fixed beliefs, one inference per sample) and differentiating through inference agree
    gi = sample_gradient(buf, sampler, ps, NamedTuple(); through_inference = true)
    @test LenticulumCore._flatten(grad) ≈ LenticulumCore._flatten(gi) atol = 1e-6
end
