# Random linear-Gaussian trees against the dense joint solve (review of 2026-11-09, B1).
#
# Every variable but the root has a parent through a `GaussianFactor` with a random rectangular
# coefficient; some variables carry an observation through a second factor and a `DataFactor`.
# Children of lower dimension than their parent, and observations of lower dimension than the
# state, send backward messages whose precision is rank-deficient: a Kalman smoother with partial
# observations is the chain case. The oracle is the joint Gaussian over all variables, assembled
# densely: every marginal's mean and covariance, the free energy (−log p of the data) and its
# gradient.

module GaussianOracle
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, LuxCore
using MyceliumLearning

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# the tree as plain data: dims, parents, coefficients, noises, observations
function random_tree(rng, nvars; maxdim = 3, pobs = 0.6, chain = false)
    dims = rand(rng, 1:maxdim, nvars)
    parent = [i == 1 ? 0 : (chain ? i - 1 : rand(rng, 1:(i - 1))) for i in 1:nvars]
    A = [i == 1 ? zeros(0, 0) : 0.7 .* randn(rng, dims[i], dims[parent[i]]) for i in 1:nvars]
    b = [i == 1 ? zeros(0) : randn(rng, dims[i]) for i in 1:nvars]
    Q = [i == 1 ? zeros(0, 0) : spd(rng, dims[i]) for i in 1:nvars]
    obs = [i for i in 1:nvars if rand(rng) < pobs]
    isempty(obs) && push!(obs, nvars)
    dobs = Dict(i => rand(rng, 1:dims[i]) for i in obs)          # often fewer rows than columns
    H = Dict(i => randn(rng, dobs[i], dims[i]) for i in obs)
    c = Dict(i => randn(rng, dobs[i]) for i in obs)
    R = Dict(i => spd(rng, dobs[i]) for i in obs)
    y = Dict(i => randn(rng, dobs[i]) for i in obs)
    μ0, P0 = randn(rng, dims[1]), spd(rng, dims[1])
    return (; dims, parent, A, b, Q, obs, H, c, R, y, μ0, P0)
end

function graph(t)
    gb = GraphBuilder()
    n = length(t.dims)
    for i in 1:n; variable!(gb, Symbol(:x, i), t.dims[i]); end
    factor!(gb, :prior, GaussianPrior(:x, t.μ0, t.P0)); connect!(gb, :prior, :x, :x1)
    for i in 2:n
        f = Symbol(:f, i)
        factor!(gb, f, GaussianFactor(t.dims[t.parent[i]] => t.dims[i]; noise = t.Q[i], channels = (:a, :b)))
        connect!(gb, f, :a, Symbol(:x, t.parent[i])); connect!(gb, f, :b, Symbol(:x, i))
    end
    for i in t.obs
        h, d, yv = Symbol(:h, i), Symbol(:d, i), Symbol(:y, i)
        variable!(gb, yv, length(t.y[i]))
        factor!(gb, h, GaussianFactor(t.dims[i] => length(t.y[i]); noise = t.R[i], channels = (:a, :b)))
        connect!(gb, h, :a, Symbol(:x, i)); connect!(gb, h, :b, yv)
        factor!(gb, d, DataFactor(:v, t.y[i])); connect!(gb, d, :v, yv; direction = Emitting())
    end
    g = validate(build(gb))
    ps0 = LuxCore.initialparameters(Xoshiro(0), g)
    kv = Dict{Symbol,Any}(pairs(ps0))
    for i in 2:n; kv[Symbol(:f, i)] = (A = t.A[i], b = t.b[i]); end
    for i in t.obs; kv[Symbol(:h, i)] = (A = t.H[i], b = t.c[i]); end
    ps = NamedTuple{keys(ps0)}(Tuple(kv[k] for k in keys(ps0)))
    return g, ps, LuxCore.initialstates(Xoshiro(0), g)
end

# The joint over x = (x1, …, xn) given the data: precision Λ, information η, and −log p(y).
function dense(t)
    n = length(t.dims)
    off = cumsum([0; t.dims]); N = off[end]
    rg(i) = (off[i] + 1):off[i + 1]
    Λ = zeros(N, N); η = zeros(N); cst = 0.0                      # −log of the unnormalised joint at x: ½xᵀΛx − ηᵀx + cst
    Pi = inv(t.P0)
    Λ[rg(1), rg(1)] += Pi; η[rg(1)] += Pi * t.μ0
    cst += (t.μ0' * Pi * t.μ0 + logdet(2π .* t.P0)) / 2
    for i in 2:n
        G = zeros(t.dims[i], N); G[:, rg(t.parent[i])] = -t.A[i]; G[:, rg(i)] = I(t.dims[i])
        Qi = inv(t.Q[i])
        Λ += G' * Qi * G; η += G' * (Qi * t.b[i])
        cst += (t.b[i]' * Qi * t.b[i] + logdet(2π .* t.Q[i])) / 2
    end
    for i in t.obs
        G = zeros(length(t.y[i]), N); G[:, rg(i)] = t.H[i]
        Ri = inv(t.R[i]); d = t.y[i] - t.c[i]
        Λ += G' * Ri * G; η += G' * (Ri * d)
        cst += (d' * Ri * d + logdet(2π .* t.R[i])) / 2
    end
    Λ = Symmetric((Λ + Λ') / 2)
    μ = Λ \ η; Σ = inv(Λ)
    nlogp = cst - (η' * μ) / 2 + (logdet(Λ) - N * log(2π)) / 2
    return μ, Matrix(Σ), nlogp, rg
end

end # module

@testset "random linear-Gaussian trees against the dense joint solve" begin
    GO = GaussianOracle
    for seed in 1:40
        rng = Xoshiro(seed)
        t = GO.random_tree(rng, rand(rng, 2:6); chain = isodd(seed))
        g, ps, st = GO.graph(t)
        μ, Σ, nlogp, rg = GO.dense(t)
        scale = 1 + maximum(abs, Σ)
        for sched in (tree_schedule(g), tree_schedule(g; root = Symbol(:x, length(t.dims))))
            m, rep, _, s = infer!(g, sched, ps, st)
            @test rep.converged
            for i in eachindex(t.dims)
                b = m[Symbol(:x, i)]
                @test belief_mean(b) ≈ μ[rg(i)] atol = 1e-8 * scale
                @test belief_cov(b) ≈ Σ[rg(i), rg(i)] atol = 1e-8 * scale
            end
            @test first(LenticulumCore.scalar_free_energy(s, g, ps, st)) ≈ nlogp atol = 1e-8 * (1 + abs(nlogp))
        end
    end
end

@testset "a smoother with position-only measurements" begin
    # x_t = A x_{t-1} + w, y_t = H x_t + v with H of size 1 × 2, for the observation matrices that
    # threw or gave wrong marginals before (review of 2026-11-09)
    GO = GaussianOracle
    for H in ([1.0 0.0], [0.1 0.3], [1.0 1.0], [0.3 0.7])
        T = 4
        t = (dims = fill(2, T + 1), parent = [0; 1:T], A = [i == 1 ? zeros(0, 0) : [1.0 1.0; 0.0 1.0] for i in 1:(T + 1)],
             b = [i == 1 ? zeros(0) : zeros(2) for i in 1:(T + 1)], Q = [i == 1 ? zeros(0, 0) : Matrix(0.1I, 2, 2) for i in 1:(T + 1)],
             obs = collect(2:(T + 1)), H = Dict(i => H for i in 2:(T + 1)), c = Dict(i => [0.0] for i in 2:(T + 1)),
             R = Dict(i => fill(0.5, 1, 1) for i in 2:(T + 1)), y = Dict(i => [0.3 * (i - 1)] for i in 2:(T + 1)),
             μ0 = zeros(2), P0 = Matrix(1.0I, 2, 2))
        g, ps, st = GO.graph(t)
        μ, Σ, nlogp, rg = GO.dense(t)
        m, _, _, s = infer!(g, tree_schedule(g), ps, st)
        for i in 1:(T + 1)
            @test belief_mean(m[Symbol(:x, i)]) ≈ μ[rg(i)] atol = 1e-9
            @test belief_cov(m[Symbol(:x, i)]) ≈ Σ[rg(i), rg(i)] atol = 1e-9
        end
        @test first(LenticulumCore.scalar_free_energy(s, g, ps, st)) ≈ nlogp atol = 1e-9
    end
    # a square observation matrix of rank one
    f = GaussianFactor(2 => 2; noise = 0.5)
    msg, _ = Mycelium.factor_message(f, :x, nothing, (y = DiracBelief([1.0, 2.0]),), nothing, (A = [1.0 2.0; 2.0 4.0], b = zeros(2)), NamedTuple())
    @test msg.improper && !LenticulumCore.isproper(msg)
    full, _ = Mycelium.factor_message(f, :x, nothing, (y = DiracBelief([1.0, 2.0]),), nothing, (A = [1.0 0.5; 0.2 1.0], b = zeros(2)), NamedTuple())
    @test LenticulumCore.isproper(full)
end

@testset "the envelope gradient on a random tree is the gradient of −log p" begin
    GO = GaussianOracle
    t = GO.random_tree(Xoshiro(11), 4; chain = true)
    g, ps, st = GO.graph(t)
    _, _, _, s = infer!(g, tree_schedule(g), ps, st)
    grad = LenticulumCore.free_energy_gradient(s, g, ps, st)
    h = 1e-6
    for i in 2:length(t.dims), idx in eachindex(t.A[i])
        up = deepcopy(t.A); dn = deepcopy(t.A); up[i][idx] += h; dn[i][idx] -= h
        fd = (GO.dense(merge(t, (A = up,)))[3] - GO.dense(merge(t, (A = dn,)))[3]) / 2h
        @test grad[Symbol(:f, i)].A[idx] ≈ fd atol = 1e-6
    end
end

@testset "composition of Gaussian factors with full noise" begin
    # B Q Bᵀ is symmetric only up to rounding; the collapsed factor must accept it
    f1 = GaussianFactor(3 => 3; noise = [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0], channels = (:x, :m))
    f2 = GaussianFactor(3 => 2; noise = [0.4 0.1; 0.1 0.3], channels = (:m, :z))
    c = LenticulumCore.compose(f1, f2)
    for seed in 1:50
        ps = LuxCore.initialparameters(Xoshiro(seed), c)
        gcol, pcol = Lenticulum._collapsed_gaussian(c, ps)
        @test pcol.A ≈ ps.second.A * ps.first.A
        @test gcol.Q ≈ ps.second.A * f1.Q * ps.second.A' + f2.Q
    end
    ps = LuxCore.initialparameters(Xoshiro(1), c)
    x = [0.3, -0.2, 0.5]
    msg, _ = Mycelium.factor_message(c, :z, nothing, (x = DiracBelief(x),), nothing, ps, NamedTuple())
    @test belief_mean(msg) ≈ ps.second.A * (ps.first.A * x + ps.first.b) + ps.second.b
    @test belief_cov(msg) ≈ ps.second.A * f1.Q * ps.second.A' + f2.Q
end

@testset "parameter_step and free_energy_gradient take the same driver" begin
    for coupling in (DiagonalCoupling(), ExactCoupling(), PathwiseCoupling())
        gb = GraphBuilder(); variable!(gb, :x, 1); variable!(gb, :y, 1)
        factor!(gb, :prior, GaussianPrior(:x, [0.0], fill(1.0, 1, 1))); connect!(gb, :prior, :x, :x)
        factor!(gb, :f, GaussianFactor(1 => 1; noise = 0.1))
        connect!(gb, :f, :x, :x; coupling); connect!(gb, :f, :y, :y; coupling)
        factor!(gb, :d, DataFactor(:v, [1.0])); connect!(gb, :d, :v, :y; direction = Emitting())
        g = build(gb)
        ps = LuxCore.initialparameters(Xoshiro(2), g); st = LuxCore.initialstates(Xoshiro(2), g)
        _, _, _, s = infer!(g, tree_schedule(g), ps, st)
        grad = LenticulumCore.free_energy_gradient(s, g, ps, st)
        η = 0.05
        psn, _, _ = parameter_step(s, g, ps, st; rule = GradientDescent(η))
        @test psn.f.A ≈ ps.f.A .- η .* grad.f.A
        @test psn.f.b ≈ ps.f.b .- η .* grad.f.b
    end
end

@testset "dimension mismatches are refused by validate" begin
    gb = GraphBuilder(); variable!(gb, :x, 2); variable!(gb, :y, 3)
    factor!(gb, :f, GaussianFactor(2 => 2)); connect!(gb, :f, :x, :x); connect!(gb, :f, :y, :y)
    @test_throws DimensionMismatch validate(build(gb))
    gb = GraphBuilder(); variable!(gb, :y, 2)
    factor!(gb, :d, DataFactor(:v, [1.0])); connect!(gb, :d, :v, :y; direction = Emitting())
    @test_throws DimensionMismatch validate(build(gb))
end
