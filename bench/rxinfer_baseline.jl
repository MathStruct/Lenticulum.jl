# An external baseline: Lenticulum against RxInfer and the dense solve on the same models.
# Recorded in vault/Gaussian Graphs/Benchmark against RxInfer.md.
#
# Run:  julia -t auto --project=bench bench/rxinfer_baseline.jl
# (once: julia --project=bench -e 'using Pkg; Pkg.instantiate()')
#
# 1. A linear-Gaussian state-space model (a Kalman smoother): a constant-velocity state in ℝ²,
#    position-only measurements, T steps. Both libraries run belief propagation on a tree, which
#    is exact, so both free energies must equal −log p(y) of the dense joint solve, and both
#    posteriors its conditional marginals. Times are warm (after one run), best of `REPS`.
# 2. A mixture of two linear regressions with one noise level per line. The libraries estimate
#    different things: Lenticulum's `fit_em` the maximum-likelihood parameters (EM on a graph of
#    `RegimeFactor`s), RxInfer a mean-field variational posterior under priors (VMP). Both are
#    scored by one function, −log p(y | x, θ̂) at their point estimates (RxInfer's posterior means,
#    σ = 1/√E[τ]), with the labels matched; hand-written closed-form EM is the reference.

using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, LuxCore, Random, Printf
import RxInfer
using RxInfer: @model, @constraints, @initialization, MvNormalMeanCovariance, NormalMixture, Categorical,
    Dirichlet, GammaShapeRate, NormalMeanPrecision, infer, GraphPPL   # @model expands to GraphPPL calls

const REPS = 5
best(f) = (f(); minimum(@elapsed(f()) for _ in 1:REPS))

# --- the model, as plain data ---------------------------------------------------------------

function ssm_data(rng, T)
    A = [1.0 1.0; 0.0 1.0]
    H = [1.0 0.0]
    Q = [0.05 0.0; 0.0 0.02]
    R = fill(0.5, 1, 1)
    μ0, P0 = [0.0, 1.0], Matrix(1.0I, 2, 2)
    x = μ0 + cholesky(P0).L * randn(rng, 2)
    ys = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        x = A * x + cholesky(Q).L * randn(rng, 2)
        ys[t] = H * x + sqrt(R[1]) * randn(rng, 1)
    end
    return (; A, H, Q, R, μ0, P0, ys, T)
end

# --- the references ---------------------------------------------------------------------------
# Marginals: the dense joint solve in information form. The prior of x = (x0, …, x_T) has the
# block-tridiagonal precision of the Markov chain; adding CᵀR⁻¹C gives the posterior precision,
# whose Cholesky factor gives the posterior means and covariances.
#
# −log p(y): the Kalman filter's prediction-error decomposition, Σₜ ½(eₜ²/Sₜ + log 2πSₜ), in 256-bit
# BigFloat. Two dense formulas were tried first and lost accuracy at T = 1000: the covariance form
# (Σ of the prior grows like T³ here, then CΣCᵀ + R) by 2.8e-7, the information form (a difference
# of terms of order 1e3–1e4) by 1.2e-5, while the two libraries agreed with each other to 2e-11.

function ssm_nlogp(m)
    setprecision(BigFloat, 256) do
        A, H, Q, R = big.(m.A), big.(m.H), big.(m.Q), big(m.R[1])
        μ, P = big.(m.μ0), big.(m.P0)
        F = big(0)
        for t in 1:m.T
            μ, P = A * μ, A * P * A' + Q                      # predict
            S = (H * P * H')[1] + R
            e = big(m.ys[t][1]) - (H * μ)[1]
            F += (e^2 / S + log(2 * big(π) * S)) / 2
            K = P * H' / S                                    # update
            μ, P = μ + K[:, 1] * e, P - K * H * P
        end
        Float64(F)
    end
end

function ssm_dense(m)
    T, d = m.T, 2
    N = d * (T + 1)
    blk(t) = (d * t + 1):(d * (t + 1))
    Qi, P0i, Ri = inv(m.Q), inv(m.P0), 1 / m.R[1]
    Λ = zeros(N, N); h = zeros(N)
    Λ[blk(0), blk(0)] += P0i
    h[blk(0)] += P0i * m.μ0
    for t in 1:T
        Λ[blk(t - 1), blk(t - 1)] += m.A' * Qi * m.A
        Λ[blk(t - 1), blk(t)] -= m.A' * Qi
        Λ[blk(t), blk(t - 1)] -= Qi * m.A
        Λ[blk(t), blk(t)] += Qi
    end
    for t in 1:T
        Λ[blk(t), blk(t)] += Ri * (m.H' * m.H)
        h[blk(t)] += Ri * (m.H' * m.ys[t])
    end
    C = cholesky(Symmetric(Λ))
    μp = C \ h
    Σp = inv(C)
    return (μ = [μp[blk(t)] for t in 0:T], Σ = [Σp[blk(t), blk(t)] for t in 0:T], nlogp = ssm_nlogp(m))
end

# --- Lenticulum ---------------------------------------------------------------------------

function ssm_lenticulum_graph(m)
    gb = GraphBuilder()
    for t in 0:m.T; variable!(gb, Symbol(:x, t), 2); end
    factor!(gb, :prior, GaussianPrior(:x, m.μ0, m.P0)); connect!(gb, :prior, :x, :x0)
    for t in 1:m.T
        f, h, d, y = Symbol(:f, t), Symbol(:h, t), Symbol(:d, t), Symbol(:y, t)
        factor!(gb, f, GaussianFactor(2 => 2; noise = m.Q, channels = (:a, :b)))
        connect!(gb, f, :a, Symbol(:x, t - 1)); connect!(gb, f, :b, Symbol(:x, t))
        variable!(gb, y, 1)
        factor!(gb, h, GaussianFactor(2 => 1; noise = m.R, channels = (:a, :b)))
        connect!(gb, h, :a, Symbol(:x, t)); connect!(gb, h, :b, y)
        factor!(gb, d, DataFactor(:v, m.ys[t])); connect!(gb, d, :v, y; direction = Emitting())
    end
    g = validate(build(gb))
    ps0 = LuxCore.initialparameters(Xoshiro(0), g)
    kv = Dict{Symbol,Any}(pairs(ps0))
    for t in 1:m.T
        kv[Symbol(:f, t)] = (A = m.A, b = zeros(2))
        kv[Symbol(:h, t)] = (A = m.H, b = zeros(1))
    end
    ps = NamedTuple{keys(ps0)}(Tuple(kv[k] for k in keys(ps0)))
    return g, ps, LuxCore.initialstates(Xoshiro(0), g)
end

function ssm_lenticulum(m)
    g, ps, st = ssm_lenticulum_graph(m)
    marg, rep, _, s = infer!(g, tree_schedule(g), ps, st)
    rep.converged || error("infer! did not converge on the chain")
    F = first(LenticulumCore.scalar_free_energy(s, g, ps, st))
    return (marg, F)
end

# --- RxInfer ------------------------------------------------------------------------------

@model function rx_ssm(y, A, H, Q, R, μ0, P0)
    x0 ~ MvNormalMeanCovariance(μ0, P0)
    x_prev = x0
    for t in eachindex(y)
        x[t] ~ MvNormalMeanCovariance(A * x_prev, Q)
        y[t] ~ MvNormalMeanCovariance(H * x[t], R)
        x_prev = x[t]
    end
end

# Beyond T = 1000 RxInfer's default run overflows the stack (its recursive message updates); its
# documented remedy is the `limit_stack_depth` option, used there only.
rx_options(T) = T > 1000 ? (limit_stack_depth = 100,) : (;)
ssm_rxinfer(m; free_energy = true) =
    infer(model = rx_ssm(A = m.A, H = m.H, Q = m.Q, R = m.R, μ0 = m.μ0, P0 = m.P0),
          data = (y = m.ys,), free_energy = free_energy, options = rx_options(m.T))

# --- the comparison -------------------------------------------------------------------------

function run_ssm()
    println("1. Linear-Gaussian state-space model (constant velocity, position measurements)\n")
    @printf("%6s | %9s %9s %9s | %9s %9s | %8s %8s %8s %8s | %8s %8s\n", "T", "|F_L-F*|", "|F_R-F*|", "|F_L-F_R|",
            "err μ L", "err μ R", "L build", "L infer", "L F", "L all", "R", "R noFE")
    for T in (10, 100, 1000, 2000)
        m = ssm_data(Xoshiro(T), T)
        ref = ssm_dense(m)
        margL, FL = ssm_lenticulum(m)
        r = ssm_rxinfer(m)
        FR = last(r.free_energy)
        xs = r.posteriors[:x]
        errL = maximum(maximum(abs, belief_mean(margL[Symbol(:x, t)]) - ref.μ[t + 1]) for t in 0:T)
        rx(t) = t == 0 ? r.posteriors[:x0] : xs[t]          # RxInfer's x0 and x[1..T]
        errR = maximum(maximum(abs, RxInfer.mean(rx(t)) - ref.μ[t + 1]) for t in 0:T)
        errLΣ = maximum(maximum(abs, belief_cov(margL[Symbol(:x, t)]) - ref.Σ[t + 1]) for t in 0:T)
        errRΣ = maximum(maximum(abs, RxInfer.cov(rx(t)) - ref.Σ[t + 1]) for t in 0:T)
        g, ps, st = ssm_lenticulum_graph(m)
        sched = tree_schedule(g)
        _, _, _, s = infer!(g, sched, ps, st)
        tb = best(() -> ssm_lenticulum_graph(m))
        ti = best(() -> infer!(g, tree_schedule(g), ps, st))
        tF = best(() -> LenticulumCore.scalar_free_energy(s, g, ps, st))
        tL = best(() -> ssm_lenticulum(m))
        tR = best(() -> ssm_rxinfer(m))
        tR0 = best(() -> ssm_rxinfer(m; free_energy = false))
        @printf("%6d | %9.1e %9.1e %9.1e | %9.1e %9.1e | %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f\n", T,
                abs(FL - ref.nlogp), abs(FR - ref.nlogp), abs(FL - FR), errL, errR,
                1000tb, 1000ti, 1000tF, 1000tL, 1000tR, 1000tR0)
        @printf("%6s   −log p(y) = %.6f; largest covariance error: L %.1e, R %.1e\n", "", ref.nlogp, errLΣ, errRΣ)
    end
    println("""
    F_L, F_R: the free energies of Lenticulum and RxInfer; F* = −log p(y) by the Kalman filter in
    BigFloat. err μ: the largest error of a posterior mean against the dense solve.
    Times in ms, warm, best of $REPS. L build: the graph and parameters; L infer: tree_schedule and
    infer!; L F: scalar_free_energy; L all: the three. R: RxInfer's infer (model creation included)
    with free_energy = true; R noFE: without it. For T > 1000 RxInfer runs with
    limit_stack_depth = 100 (without it, a stack overflow at T = 2000).""")
end

# BENCH_PART = ssm, mix or all (the default)
const PART = get(ENV, "BENCH_PART", "all")
PART in ("all", "ssm") && run_ssm()

# --- 2. a mixture of regressions ------------------------------------------------------------

function mix_data(N; seed = 11)
    rng = Xoshiro(seed)
    x = 6 .* rand(rng, N) .- 3
    k = [rand(rng) < 0.5 ? 1 : 2 for _ in 1:N]
    y = [1.0, -1.0][k] .* x .+ [0.0, 0.5][k] .+ [0.2, 0.5][k] .* randn(rng, N)
    return x, y
end

lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
# −log p(y | x) of the mixture at (a, c, σ, w): the common score
mix_nll(x, y, a, c, σ, w) = -sum(lse([log(w[k]) - (y[i] - a[k] * x[i] - c[k])^2 / (2σ[k]^2) - log(σ[k]) - log(2π) / 2
                                       for k in 1:2]) for i in eachindex(x))

const START = (a = [0.5, -0.5], c = [0.0, 0.0], σ = [1.0, 1.0], w = [0.5, 0.5])

# closed-form EM, until −log p changes by less than tol
function mix_em_hand(x, y; tol = 1e-10, maxiter = 10_000)
    a, c, σ, w = copy(START.a), copy(START.c), copy(START.σ), copy(START.w)
    X = [x ones(length(x))]
    F = mix_nll(x, y, a, c, σ, w); it = 0
    for outer it in 1:maxiter
        lr = [log(w[k]) - (y[i] - a[k] * x[i] - c[k])^2 / (2σ[k]^2) - log(σ[k]) for i in eachindex(x), k in 1:2]
        R = exp.(lr .- [lse(lr[i, :]) for i in eachindex(x)])
        for k in 1:2
            W = Diagonal(R[:, k])
            a[k], c[k] = (X' * W * X) \ (X' * W * y)
            σ[k] = sqrt(sum(R[:, k] .* (y .- a[k] .* x .- c[k]) .^ 2) / sum(R[:, k]))
            w[k] = sum(R[:, k]) / length(x)
        end
        Fn = mix_nll(x, y, a, c, σ, w)
        abs(F - Fn) < tol && (F = Fn; break)
        F = Fn
    end
    return (; a, c, σ, w, iters = it)
end

# Lenticulum: one RegimeFactor per point, tied parameters, fit_em (as test/em.jl); with `plated`
# the replicas are recorded as a plate (`as_plate`), and messages, free energies and EM statistics
# of the regime factors are batched over it (Feature Backlog I6)
function mix_graph(x, y; plated = false)
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
mix_ps(N) = merge(NamedTuple(s => NamedTuple() for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i))),
    (reg = (A = (x = reshape(-START.a, 1, 1, 2),), b = reshape(START.c, 1, 2), logw = log.(START.w),
            L = reshape(log.(START.σ), 1, 1, 2)),))
function mix_lenticulum(x, y, g; tol = 1e-10)
    schedule = isempty(plates(g)) ? SequentialSchedule(all_tasks(g)) : plate_schedule(g)
    ps, Fs, _ = fit_em(g, mix_ps(length(x)), NamedTuple(); schedule, iters = 10_000, tol)
    p = ps.reg
    w = exp.(p.logw); w ./= sum(w)
    return (a = -vec(p.A.x), c = vec(p.b), σ = exp.(vec(p.L)), w, iters = length(Fs) - 1, F = last(Fs))
end

# RxInfer: priors, mean-field VMP
@model function rx_mixreg(y, X)
    w ~ Dirichlet([1.0, 1.0])
    θ1 ~ MvNormalMeanCovariance(zeros(2), 100.0 * Matrix(1.0I, 2, 2))
    θ2 ~ MvNormalMeanCovariance(zeros(2), 100.0 * Matrix(1.0I, 2, 2))
    τ1 ~ GammaShapeRate(1.0, 1.0)
    τ2 ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        z[i] ~ Categorical(w)
        m1[i] := dot(X[i], θ1)
        m2[i] := dot(X[i], θ2)
        y[i] ~ NormalMixture(switch = z[i], m = (m1[i], m2[i]), p = (τ1, τ2))
    end
end
const RX_CONS = @constraints begin
    q(z, w, θ1, θ2, τ1, τ2, m1, m2) = q(z)q(w)q(θ1)q(θ2)q(τ1)q(τ2)q(m1)q(m2)
end
# the same start as EM: the lines' means at START, unit noise, equal weights
const RX_INIT = @initialization begin
    q(θ1) = MvNormalMeanCovariance([0.5, 0.0], Matrix(1.0I, 2, 2))
    q(θ2) = MvNormalMeanCovariance([-0.5, 0.0], Matrix(1.0I, 2, 2))
    q(τ1) = GammaShapeRate(1.0, 1.0); q(τ2) = GammaShapeRate(1.0, 1.0)
    q(w) = Dirichlet([1.0, 1.0])
    q(z) = Categorical([0.5, 0.5])
    q(m1) = NormalMeanPrecision(0.0, 1.0); q(m2) = NormalMeanPrecision(0.0, 1.0)
end
function mix_rxinfer(x, y, iterations)
    r = infer(model = rx_mixreg(X = [[xi, 1.0] for xi in x]), data = (y = y,), constraints = RX_CONS,
              initialization = RX_INIT, iterations = iterations, free_energy = true,
              options = rx_options(length(x)),
              returnvars = NamedTuple{(:θ1, :θ2, :τ1, :τ2, :w)}(ntuple(_ -> RxInfer.KeepLast(), 5)))
    θ = [RxInfer.mean(r.posteriors[:θ1]), RxInfer.mean(r.posteriors[:θ2])]
    τ = [RxInfer.mean(r.posteriors[:τ1]), RxInfer.mean(r.posteriors[:τ2])]
    return (a = first.(θ), c = last.(θ), σ = 1 ./ sqrt.(τ), w = RxInfer.mean(r.posteriors[:w]), F = r.free_energy)
end
# the iterations after which VMP's free energy changes by less than tol per iteration
function rx_converged(F; tol = 1e-8)
    j = findfirst(i -> abs(F[i] - F[i - 1]) < tol, 2:length(F))     # an index into 2:length(F)
    return j === nothing ? length(F) : j + 1
end

# the labels of `est` in the order of `ref` (two lines: keep or swap, by the slopes)
match(est, ref) = abs(est.a[1] - ref.a[1]) + abs(est.a[2] - ref.a[2]) <= abs(est.a[2] - ref.a[1]) + abs(est.a[1] - ref.a[2]) ?
    est : merge(est, (a = reverse(est.a), c = reverse(est.c), σ = reverse(est.σ), w = reverse(est.w)))

function run_mix()
    println("\n2. A mixture of two linear regressions, one noise level per line (a = (1, −1), c = (0, 0.5), σ = (0.2, 0.5))\n")
    @printf("%6s | %-10s | %8s %8s | %8s %8s | %8s %8s | %6s %6s | %14s | %6s %10s\n", "N", "method",
            "a1", "a2", "c1", "c2", "σ1", "σ2", "w1", "w2", "−log p(y|x,θ̂)", "iters", "time (ms)")
    for N in (300, 3000)
        x, y = mix_data(N)
        ref = mix_em_hand(x, y)
        g = mix_graph(x, y)
        L = mix_lenticulum(x, y, g)
        gp = mix_graph(x, y; plated = true)
        Lp = mix_lenticulum(x, y, gp)
        # VMP to convergence: find the iteration count once, then time runs of that length
        R0 = mix_rxinfer(x, y, 1000)
        nR = rx_converged(R0.F)
        R = mix_rxinfer(x, y, nR)
        # the graph's free energy at the fit is −log p(y | x) (a forest, exact): checked against the score
        abs(L.F - mix_nll(x, y, L.a, L.c, L.σ, L.w)) < 1e-8 * (1 + abs(L.F)) ||
            error("fit_em's free energy $(L.F) is not −log p(y|x) at its estimate")
        tE = best(() -> mix_em_hand(x, y))
        tLb = best(() -> mix_graph(x, y))
        tL = best(() -> mix_lenticulum(x, y, g))
        tLp = best(() -> mix_lenticulum(x, y, gp))
        tR = best(() -> mix_rxinfer(x, y, nR))
        for (name, e, t, it) in (("EM (hand)", ref, tE, ref.iters), ("Lenticulum", match(L, ref), tL, L.iters),
                              ("L (plate)", match(Lp, ref), tLp, Lp.iters), ("RxInfer", match(R, ref), tR, nR))
            @printf("%6d | %-10s | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %6.3f %6.3f | %14.6f | %6d %10.1f\n", N, name,
                    e.a..., e.c..., e.σ..., e.w..., mix_nll(x, y, e.a, e.c, e.σ, e.w), it, 1000t)
        end
        @printf("%6s   Lenticulum's graph build %.1f ms (not in its time above); RxInfer's time includes creating its model\n", "", 1000tLb)
        @printf("%6s   RxInfer's free energy at convergence %.4f (a bound under its priors, not −log p(y|x,θ̂))\n", "", last(R.F))
    end
    println("""
    Start for all: a = (0.5, −0.5), c = 0, σ = 1, w = (0.5, 0.5). EM (hand) and Lenticulum stop when −log p
    changes by less than 1e-10; RxInfer when its free energy changes by less than 1e-8 per iteration
    (iteration count found in a run of 1000, then timed). Lenticulum's time excludes building the graph,
which is reported below each block; RxInfer's includes creating its model.""")
end

PART in ("all", "mix") && run_mix()
