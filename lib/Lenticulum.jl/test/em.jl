# Closed-form M-steps from cluster beliefs (`lib/Mycelium.jl/src/em.jl`, `src/m_steps.jl`).
# Oracles: hand-written EM for a mixture of linear regressions (the same iterates), the free
# energy's non-increase, and stationarity at the fixed point (the free-energy gradient, by
# ForwardDiff, vanishes there), which checks the M-step formulas independently.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using MyceliumLearning
using ADTypes, DifferentiationInterface, ForwardDiff

em_lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

# --- a mixture of regressions with one noise level per line ---------------------------------
function em_lines_data(N)
    rng = Xoshiro(11)
    x = 6 .* rand(rng, N) .- 3
    k = [rand(rng) < 0.5 ? 1 : 2 for _ in 1:N]
    return x, [1.0, -1.0][k] .* x .+ [0.0, 0.5][k] .+ [0.2, 0.5][k] .* randn(rng, N)
end
function em_lines_graph(x, y; freeze = ())
    b = GraphBuilder()
    for i in eachindex(x)
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 1); variable!(b, r(:y), 1); variable!(b, r(:k), 2)
        factor!(b, r(:dx), DataFactor(:x, [x[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
        factor!(b, r(:dy), DataFactor(:y, [y[i]])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
        factor!(b, r(:reg), RegimeFactor((y = 1, x = 1), 2, 1; learn_noise = true, output = :y); params = :reg, freeze)
        connect!(b, r(:reg), :y, r(:y)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    return build(b)
end
em_lines_ps(N, a, c, logw, logσ) = merge(NamedTuple(s => NamedTuple() for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i))),
    (reg = (A = (x = reshape(-a, 1, 1, 2),), b = reshape(c, 1, 2), logw = logw, L = reshape(logσ, 1, 1, 2)),))
function em_lines_hand(x, y, a, c, σ, w, steps)
    a, c, σ, w = copy(a), copy(c), copy(σ), copy(w)
    X = [x ones(length(x))]
    for _ in 1:steps
        lr = [log(w[k]) - (y[i] - a[k] * x[i] - c[k])^2 / (2σ[k]^2) - log(σ[k]) for i in eachindex(x), k in 1:2]
        R = exp.(lr .- [em_lse(lr[i, :]) for i in eachindex(x)])
        for k in 1:2
            Wt = Diagonal(R[:, k])
            a[k], c[k] = (X' * Wt * X) \ (X' * Wt * y)
            σ[k] = sqrt(sum(R[:, k] .* (y .- a[k] .* x .- c[k]) .^ 2) / sum(R[:, k]))
            w[k] = sum(R[:, k]) / length(x)
        end
    end
    return a, c, σ, w
end
em_sweep(g) = SequentialSchedule(all_tasks(g))

@testset "EM: a mixture of regressions follows hand-written EM step for step" begin
    N = 120
    x, y = em_lines_data(N)
    g = em_lines_graph(x, y)
    a0, c0, σ0, w0 = [0.5, -0.5], [0.0, 0.0], [1.0, 1.0], [0.5, 0.5]
    for steps in (1, 3, 8)
        # `iters` E-steps, each followed by an M-step (`steps` EM steps), and the free energy at the end
        ps, Fs, _ = fit_em(g, em_lines_ps(N, a0, c0, log.(w0), log.(σ0)), NamedTuple(); schedule = em_sweep(g), iters = steps, tol = -Inf)
        @test length(Fs) == steps + 1
        a, c, σ, w = em_lines_hand(x, y, a0, c0, σ0, w0, steps)
        p = ps.reg
        @test -vec(p.A.x) ≈ a rtol = 1e-10
        @test vec(p.b) ≈ c rtol = 1e-10 atol = 1e-12
        @test exp.(vec(p.L)) ≈ σ rtol = 1e-10
        @test exp.(p.logw) ≈ w rtol = 1e-10
        @test all(diff(Fs) .<= 1e-9)                       # EM does not increase −log p
    end
    # the statistics: the outcome masses add up to the number of points; a custom M-step
    ps0 = em_lines_ps(N, a0, c0, log.(w0), log.(σ0))
    s = MessageStore(g); sweep!(s, g, em_sweep(g), ps0, NamedTuple()); sweep!(s, g, em_sweep(g), ps0, NamedTuple())
    st = moment_statistics(s, g, ps0, NamedTuple())
    @test collect(keys(st)) == [(:reg, nothing)]
    stats = st[(:reg, nothing)][2]
    @test sum(stats.n) ≈ N && stats.channels == [:x, :y]
    @test sum(stats.m[2, :]) ≈ sum(y)                         # y is observed: its first moment is the data's
    pa, _, _ = fit_em(g, ps0, NamedTuple(); schedule = em_sweep(g), iters = 2, tol = -Inf)
    pb, _, _ = fit_em(g, ps0, NamedTuple(); schedule = em_sweep(g), iters = 2, tol = -Inf, mstep = em_step)
    @test pa.reg.b == pb.reg.b && pa.reg.L == pb.reg.L
    # frozen weights stay
    gf = em_lines_graph(x, y; freeze = (:logw,))
    ps, _, _ = fit_em(gf, em_lines_ps(N, a0, c0, log.([0.3, 0.7]), log.(σ0)), NamedTuple(); schedule = em_sweep(gf), iters = 3, tol = -Inf)
    @test ps.reg.logw ≈ log.([0.3, 0.7])
end

@testset "EM: a mixture of factor analysers in a cluster reaches a stationary point" begin
    # z = W_k x + μ_k + ε_k, x ~ N(0, I₂), z ∈ R³ observed, three regimes, a cluster per point (tied)
    rng = Xoshiro(4)
    Wt = [randn(rng, 3, 2) for _ in 1:3]; μt = [[0.0, 3.0, 0.0], [3.0, -2.0, 1.0], [-3.0, 0.0, 2.0]]
    N = 90
    Z = reduce(hcat, [(k = mod1(i, 3); Wt[k] * randn(rng, 2) .+ μt[k] .+ 0.2 .* randn(rng, 3)) for i in 1:N])
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:z), 3); variable!(b, r(:x), 2); variable!(b, r(:k), 3)
        factor!(b, r(:d), DataFactor(:z, Z[:, i])); connect!(b, r(:d), :z, r(:z); direction = Emitting())
        factor!(b, r(:pr), GaussianPrior(:x, zeros(2), Matrix(1.0I, 2, 2)); params = :pr); connect!(b, r(:pr), :x, r(:x))
        factor!(b, r(:reg), RegimeFactor((z = 3, x = 2), 3, 3; learn_noise = true, output = :z, noise = 0.5); params = :reg)
        connect!(b, r(:reg), :z, r(:z)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    g = build(b)
    gc = cluster(as_plate(g, :pts, N), :pts, :c, [:pr, :reg]; params = :mix)
    f = RegimeFactor((z = 3, x = 2), 3, 3; learn_noise = true, output = :z, noise = 0.5)
    p0 = merge(LuxCore.initialparameters(Xoshiro(2), f), (b = reduce(hcat, [Z[:, i] for i in 1:3]),))
    ps0 = merge(NamedTuple(replica_name(:d, i) => NamedTuple() for i in 1:N), (mix = (pr = NamedTuple(), reg = p0),))
    ps, Fs, s = fit_em(gc, ps0, NamedTuple(); schedule = plate_schedule(gc), iters = 400, tol = 1e-10)
    @test all(diff(Fs) .<= 1e-8)
    @test Fs[end] < Fs[1] - 100
    G = free_energy_gradient(s, gc, ps, NamedTuple(); ad = AutoForwardDiff()).mix.reg
    @test maximum(abs, G.b) < 1e-4 && maximum(abs, G.A.x) < 1e-4 && maximum(abs, G.L) < 1e-4
    @test abs(sum(G.logw)) < 1e-6                           # the weights' gradient sums to zero at a stationary point
end

@testset "EM: switches over a frozen sum, and the refusal" begin
    # E3's additive wiring: z = c + u₁ + u₂ + ε, the sum's coefficients frozen; EM in c, b, logw
    rng = Xoshiro(9)
    N = 60
    Z = [0.5 .+ (rand(rng) < 0.5 ? 2.0 : 0.0) .+ (rand(rng) < 0.3 ? -1.5 : 0.0) .+ 0.3 * randn(rng) for _ in 1:N]
    function graph(freeze; swnoise = 0.05)
        b = GraphBuilder()
        for i in 1:N
            r(s) = replica_name(s, i)
            variable!(b, r(:z), 1); variable!(b, r(:u1), 1); variable!(b, r(:u2), 1); variable!(b, r(:k1), 2); variable!(b, r(:k2), 2)
            factor!(b, r(:d), DataFactor(:z, [Z[i]])); connect!(b, r(:d), :z, r(:z); direction = Emitting())
            factor!(b, r(:sum), LinearConstraintFactor((z = 1, u1 = 1, u2 = 1), 1; noise = 0.09); params = :sum, freeze)
            connect!(b, r(:sum), :z, r(:z)); connect!(b, r(:sum), :u1, r(:u1)); connect!(b, r(:sum), :u2, r(:u2))
            for l in 1:2
                factor!(b, r(Symbol(:sw, l)), SwitchingConstraintFactor((u = 1,), 2, 1; noise = swnoise); params = Symbol(:sw, l), freeze = (:A,))
                connect!(b, r(Symbol(:sw, l)), :u, r(Symbol(:u, l))); connect!(b, r(Symbol(:sw, l)), :k, r(Symbol(:k, l)))
            end
        end
        return cluster(as_plate(build(b), :pts, N), :pts, :c, [:sum, :sw1, :sw2]; params = :mix)
    end
    sw(b) = (A = (u = ones(1, 1),), b = b, logw = zeros(2))
    inner = (sum = (A = (z = ones(1, 1), u1 = -ones(1, 1), u2 = -ones(1, 1)), c = [0.0]), sw1 = sw([0.0 1.0]), sw2 = sw([0.0 -1.0]))
    ps0 = merge(NamedTuple(replica_name(:d, i) => NamedTuple() for i in 1:N), (mix = inner,))
    gc = graph((:A,))
    ps, Fs, s = fit_em(gc, ps0, NamedTuple(); schedule = plate_schedule(gc), iters = 500, tol = 1e-11, accelerate = true)
    @test all(diff(Fs) .<= 1e-8) && Fs[end] < Fs[1]
    @test ps.mix.sum.A == inner.sum.A && ps.mix.sw1.A == inner.sw1.A      # frozen paths unchanged
    # the statistics from the plate's stacked beliefs equal those node by node
    sb = moment_statistics(s, gc, ps, NamedTuple())
    sn = moment_statistics(s, gc, ps, NamedTuple(); batched = false)
    @test keys(sb) == keys(sn) && length(sb) == 3
    @test all(k -> isapprox(sb[k][2].n, sn[k][2].n; rtol = 1e-12) && isapprox(sb[k][2].m, sn[k][2].m; rtol = 1e-12) &&
                   isapprox(sb[k][2].S, sn[k][2].S; rtol = 1e-12), collect(keys(sb)))
    G = free_energy_gradient(s, gc, ps, NamedTuple(); ad = AutoForwardDiff()).mix
    @test maximum(abs, G.sum.c) < 1e-5 && maximum(abs, G.sw1.b) < 1e-5 && maximum(abs, G.sw2.b) < 1e-5
    @test abs(sum(G.sw1.logw)) < 1e-6
    # nearly deterministic switches (noise 1e-3) make plain EM slow (contraction 0.99 per step);
    # SQUAREM reaches the same fixed point in far fewer E-steps
    gs = graph((:A,); swnoise = 1e-3)
    _, Fp, _ = fit_em(gs, ps0, NamedTuple(); schedule = plate_schedule(gs), iters = 3000, tol = 1e-9)
    _, Fa, _ = fit_em(gs, ps0, NamedTuple(); schedule = plate_schedule(gs), iters = 3000, tol = 1e-9, accelerate = true)
    @test length(Fp) > 1000 && length(Fa) < 200
    @test Fa[end] ≈ Fp[end] atol = 1e-6
    # with every coefficient of the sum learned the M-step would be A = 0: refused (LP-022)
    gl = graph(())
    s0 = MessageStore(gl); sweep!(s0, gl, plate_schedule(gl), ps0, NamedTuple()); sweep!(s0, gl, plate_schedule(gl), ps0, NamedTuple())
    @test_throws ArgumentError em_step(s0, gl, ps0, NamedTuple())
end

@testset "EM: tied factors with different fixed structure are not pooled" begin
    # review of 2026-11-05 (Mistral): tied constraint factors with different `output_coeff` passed
    # the tie check and pooled statistics under the first factor's fixed coefficient
    b = GraphBuilder()
    for (i, a) in enumerate((1.0, 2.0))
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 1); variable!(b, r(:y), 1)
        factor!(b, r(:d), DataFactor(:y, [0.5 * i])); connect!(b, r(:d), :y, r(:y); direction = Emitting())
        factor!(b, r(:pr), GaussianPrior(:x, 0.0, 1.0); params = :pr); connect!(b, r(:pr), :x, r(:x))
        factor!(b, r(:rel), LinearConstraintFactor((y = 1, x = 1), 1; noise = 0.1, output = :y, output_coeff = fill(a, 1, 1)); params = :rel)
        connect!(b, r(:rel), :y, r(:y)); connect!(b, r(:rel), :x, r(:x))
    end
    gc = cluster(build(b), [replica_name(:c, i) => [replica_name(:pr, i), replica_name(:rel, i)] for i in 1:2]; params = :mix)
    ps = (d = NamedTuple(), var"d[1]" = NamedTuple(), var"d[2]" = NamedTuple(),
          mix = (pr = NamedTuple(), rel = (A = (x = -ones(1, 1),), c = [0.0])))
    s = MessageStore(gc); for _ in 1:2; sweep!(s, gc, SequentialSchedule(all_tasks(gc)), ps, NamedTuple()); end
    @test_throws r"differ in fixed structure" em_step(s, gc, ps, NamedTuple())
end

@testset "EM: Gaussian factors (regression in one step, factor analysis, a learned prior covariance)" begin
    rng = Xoshiro(31)
    # (a) y = A x + b + ε, both observed: one M-step is least squares and the residual covariance
    N = 80
    X = randn(rng, 2, N); At = [1.0 -0.5; 0.3 2.0; -1.0 0.0]; bt = [0.5, -1.0, 2.0]
    Lq = [0.4 0.0 0.0; 0.1 0.3 0.0; -0.2 0.05 0.5]
    Y = At * X .+ bt .+ Lq * randn(rng, 3, N)
    function regression_graph(f)
        b = GraphBuilder()
        for i in 1:N
            r(s) = replica_name(s, i)
            variable!(b, r(:x), 2); variable!(b, r(:y), 3)
            factor!(b, r(:dx), DataFactor(:x, X[:, i])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
            factor!(b, r(:dy), DataFactor(:y, Y[:, i])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
            factor!(b, r(:gf), f; params = :gf); connect!(b, r(:gf), :x, r(:x)); connect!(b, r(:gf), :y, r(:y))
        end
        return cluster(as_plate(build(b), :pts, N), :pts, :c, [:gf]; params = :mix)
    end
    f = GaussianFactor(2 => 3; noise = 1.0, learn_noise = true)
    gc = regression_graph(f)
    data = NamedTuple(s => NamedTuple() for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i)))
    ps = merge(data, (mix = (gf = LuxCore.initialparameters(Xoshiro(1), f),),))
    s = MessageStore(gc); for _ in 1:2; sweep!(s, gc, plate_schedule(gc), ps, NamedTuple()); end
    p1 = em_step(s, gc, ps, NamedTuple()).mix.gf
    D = [X; ones(1, N)]; B = (Y * D') / (D * D')                      # least squares, [A b]
    @test isapprox(p1.A, B[:, 1:2]; rtol = 1e-10) && isapprox(p1.b, B[:, 3]; rtol = 1e-10)
    R = Y .- B * D
    @test noise_covariance(f, p1) ≈ R * R' ./ N rtol = 1e-8
    # (b) factor analysis: x ~ N(0, 1) latent, y ∈ R³ observed, diagonal noise; the clustered free
    # energy equals the unclustered one, EM does not raise it and stops at a stationary point
    Y2 = [0.8, -0.4, 1.5] * randn(rng, 1, N) .+ [0.2, 0.0, -0.5] .+ [0.3, 0.5, 0.2] .* randn(rng, 3, N)
    fa = GaussianFactor(1 => 3; noise = [1.0, 1.0, 1.0], learn_noise = :diagonal)
    function fa_graph(cl)
        b = GraphBuilder()
        for i in 1:N
            r(s) = replica_name(s, i)
            variable!(b, r(:x), 1); variable!(b, r(:y), 3)
            factor!(b, r(:dy), DataFactor(:y, Y2[:, i])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
            factor!(b, r(:pr), GaussianPrior(:x, 0.0, 1.0); params = :pr); connect!(b, r(:pr), :x, r(:x))
            factor!(b, r(:gf), fa; params = :gf); connect!(b, r(:gf), :x, r(:x)); connect!(b, r(:gf), :y, r(:y))
        end
        g = build(b)
        return cl ? cluster(as_plate(g, :pts, N), :pts, :c, [:pr, :gf]; params = :mix) : g
    end
    p0 = merge(LuxCore.initialparameters(Xoshiro(3), fa), (A = reshape([0.5, 0.5, 0.5], 3, 1),))
    dy = NamedTuple(replica_name(:dy, i) => NamedTuple() for i in 1:N)
    gu = fa_graph(false)
    psu = merge(dy, (pr = NamedTuple(), gf = p0))
    _, _, _, su = infer!(gu, SequentialSchedule(all_tasks(gu)), psu, NamedTuple(); maxsweeps = 50, tol = 1e-12)
    gcf = fa_graph(true)
    psc = merge(dy, (mix = (pr = NamedTuple(), gf = p0),))
    sc = MessageStore(gcf); for _ in 1:2; sweep!(sc, gcf, plate_schedule(gcf), psc, NamedTuple()); end
    @test first(scalar_free_energy(sc, gcf, psc, NamedTuple())) ≈ first(scalar_free_energy(su, gu, psu, NamedTuple())) rtol = 1e-10
    pf, Fs, sf = fit_em(gcf, psc, NamedTuple(); schedule = plate_schedule(gcf), iters = 500, tol = 1e-11, accelerate = true)
    @test all(diff(Fs) .<= 1e-8) && Fs[end] < Fs[1]
    G = free_energy_gradient(sf, gcf, pf, NamedTuple(); ad = AutoForwardDiff()).mix.gf
    @test maximum(abs, G.A) < 1e-5 && maximum(abs, G.b) < 1e-5 && maximum(abs, G.noise.log_precision) < 1e-5
    # (c) a Gaussian prior with learned noise on observed points: the covariance about its mean
    fp = GaussianPrior(:x, [0.5, -0.5], Matrix(1.0I, 2, 2); learn_noise = true)
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 2)
        factor!(b, r(:dx), DataFactor(:x, X[:, i])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
        factor!(b, r(:pr), fp; params = :pr); connect!(b, r(:pr), :x, r(:x))
    end
    gp = cluster(as_plate(build(b), :pts, N), :pts, :c, [:pr]; params = :mix)
    psp = merge(NamedTuple(replica_name(:dx, i) => NamedTuple() for i in 1:N), (mix = (pr = LuxCore.initialparameters(Xoshiro(1), fp),),))
    sp = MessageStore(gp); for _ in 1:2; sweep!(sp, gp, plate_schedule(gp), psp, NamedTuple()); end
    Xc = X .- [0.5, -0.5]
    @test noise_covariance(fp, em_step(sp, gp, psp, NamedTuple()).mix.pr) ≈ Xc * Xc' ./ N rtol = 1e-8
    # a noise prior has no place in a cluster's free energy
    fw = GaussianFactor(2 => 3; noise = 1.0, learn_noise = true, noise_prior = WishartPrecisionPrior(5.0, Matrix(1.0I, 3, 3)))
    @test_throws ArgumentError cluster_form(fw, LuxCore.initialparameters(Xoshiro(1), fw))
end
