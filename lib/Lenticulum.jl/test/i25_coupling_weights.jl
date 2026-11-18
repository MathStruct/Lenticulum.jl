# I25: coupling weights from the factors, against the dense joint precision.

# A loopy graph of Gaussian priors and two-channel linear constraints with random coefficients,
# at most one constraint per pair of variables, and its dense precision.
function i25_graph(rng; n = 5, dims = fill(2, n), pairs = [(1,2), (2,3), (3,4), (4,5), (5,1), (1,3)], scale = ones(n))
    b = GraphBuilder(); pd = Dict{Symbol,Any}()
    off = cumsum([0; dims]); N = off[end]; rg(i) = off[i]+1:off[i+1]
    J = zeros(N, N)
    for i in 1:n
        v, f = Symbol(:x, i), Symbol(:p, i)
        variable!(b, v, dims[i])
        L = randn(rng, dims[i], dims[i]); Σ = L * L' + I
        factor!(b, f, GaussianPrior(:x, randn(rng, dims[i]), Σ))
        connect!(b, f, :x, v); pd[f] = NamedTuple()
        J[rg(i), rg(i)] += inv(Σ)
    end
    for (k, (i, j)) in enumerate(pairs)
        m = 2; f = Symbol(:c, k)
        L = randn(rng, m, m); Q = L * L' + 0.5I
        factor!(b, f, LinearConstraintFactor((a = dims[i], b = dims[j]), m; noise = Q))
        connect!(b, f, :a, Symbol(:x, i)); connect!(b, f, :b, Symbol(:x, j))
        A = (a = randn(rng, m, dims[i]) / scale[i], b = randn(rng, m, dims[j]) / scale[j])
        pd[f] = (A = A, c = zeros(m))
        C = zeros(m, N); C[:, rg(i)] = A.a; C[:, rg(j)] = A.b
        J += C' * inv(Q) * C
    end
    g = validate(build(b)); ps = NamedTuple(pd)
    return g, ps, J, rg, pairs
end

isqrt_psd(D) = (E = eigen(Symmetric(D)); E.vectors * Diagonal(1 ./ sqrt.(E.values)) * E.vectors')

@testset "I25 coupling weights against the dense precision" begin
    for seed in 1:5
        g, ps, J, rg, prs = i25_graph(Xoshiro(seed); dims = [1, 2, 3, 2, 1])
        w = coupling_weights(g, ps)
        wr = coupling_weights(g, ps; normalize = false)
        @test Set(keys(w)) == Set(f.name for f in g.factors)
        for (k, (i, j)) in enumerate(prs)
            # one constraint per pair: the dense off-diagonal block is this factor's, and the dense
            # diagonal blocks are the D of the definition
            Jij = J[rg(i), rg(j)]
            @test wr[Symbol(:c, k)] ≈ opnorm(Jij) rtol = 1e-10
            @test w[Symbol(:c, k)] ≈ opnorm(isqrt_psd(J[rg(i), rg(i)]) * Jij * isqrt_psd(J[rg(j), rg(j)])) rtol = 1e-10
            @test 0 ≤ w[Symbol(:c, k)] ≤ 1 + 1e-12
        end
        @test all(iszero(w[Symbol(:p, i)]) for i in 1:5)   # one variable: no coupling
        # rescaling a variable (x₂ ↦ 10 x₂: its coefficients ÷ 10, its prior's mean × 10 and
        # covariance × 100) leaves the normalised weights unchanged and divides the raw ones
        g1, ps1, = i25_graph(Xoshiro(seed); n = 3, dims = [2, 2, 2], pairs = [(1,2), (2,3), (3,1)])
        b = GraphBuilder()
        for i in 1:3
            variable!(b, Symbol(:x, i), 2)
            p = g1.factors[g1.factor_index[Symbol(:p, i)]].factor
            factor!(b, Symbol(:p, i), i == 2 ? GaussianPrior(:x, 10p.μ, 100p.Σ) : p)
            connect!(b, Symbol(:p, i), :x, Symbol(:x, i))
        end
        for (k, (i, j)) in enumerate([(1,2), (2,3), (3,1)])
            f = Symbol(:c, k)
            factor!(b, f, g1.factors[g1.factor_index[f]].factor)
            connect!(b, f, :a, Symbol(:x, i)); connect!(b, f, :b, Symbol(:x, j))
        end
        gs = validate(build(b))
        # x₂ is channel b of c1 and channel a of c2
        pss = merge(ps1, (c1 = (A = (a = ps1.c1.A.a, b = ps1.c1.A.b / 10), c = ps1.c1.c),
                          c2 = (A = (a = ps1.c2.A.a / 10, b = ps1.c2.A.b), c = ps1.c2.c)))
        w1 = coupling_weights(g1, ps1); ws = coupling_weights(gs, pss)
        for k in keys(w1)
            @test ws[k] ≈ w1[k] rtol = 1e-9
        end
        @test coupling_weights(gs, pss; normalize = false)[:c1] ≈ coupling_weights(g1, ps1; normalize = false)[:c1] / 10 rtol = 1e-9
    end
end

@testset "I25 coupling precision of each factor type" begin
    rng = Xoshiro(25)
    # GaussianFactor: y = A x + b + ε, precision [-A I]ᵀ Q⁻¹ [-A I]
    f = GaussianFactor(3 => 2; noise = [0.5 0.1; 0.1 0.3])
    p = LuxCore.initialparameters(rng, f)
    dims, J = coupling_precision(f, p)
    C = [-p.A I]
    @test dims == (x = 3, y = 2)
    @test J ≈ C' * inv([0.5 0.1; 0.1 0.3]) * C
    # LinearConstraintFactor with a declared output: the fixed coefficient is included
    f = LinearConstraintFactor((u = 2, y = 2), 2; noise = 0.2, output = :y)
    p = LuxCore.initialparameters(rng, f)
    dims, J = coupling_precision(f, p)
    @test J ≈ [p.A.u I]' * [p.A.u I] / 0.2
    # SwitchingConstraintFactor: the switch shifts the constant only
    f = SwitchingConstraintFactor((u = 1, w = 2), 3, 1; noise = 0.2)
    p = LuxCore.initialparameters(rng, f)
    dims, J = coupling_precision(f, p)
    @test dims == (u = 1, w = 2)
    @test J ≈ [p.A.u p.A.w]' * [p.A.u p.A.w] / 0.2
    # RegimeFactor: the regime-weighted precision
    f = RegimeFactor((z = 1, x = 2), 2, 1; noise = [fill(0.1, 1, 1), fill(0.3, 1, 1)])
    p = merge(LuxCore.initialparameters(rng, f), (logw = [0.0, log(3.0)],))
    dims, J = coupling_precision(f, p)
    Ck(k) = [p.A.z[:, :, k] p.A.x[:, :, k]]
    @test J ≈ 0.25 * Ck(1)' * Ck(1) / 0.1 + 0.75 * Ck(2)' * Ck(2) / 0.3
    @test coupling_precision(HardConstraintFactor((a = 1, b = 1), 1), NamedTuple()) === :hard
    @test coupling_precision(NonlinearFactor(x -> sin.(x), 1 => 1; noise = 0.1), NamedTuple()) === nothing
end

@testset "I25 hard, unknown and degenerate factors" begin
    b = GraphBuilder()
    for v in (:a, :b, :c, :d); variable!(b, v, 1); end
    factor!(b, :pa, GaussianPrior(:x, [0.0], fill(1.0, 1, 1))); connect!(b, :pa, :x, :a)
    factor!(b, :ab, LinearConstraintFactor((a = 1, b = 1), 1; noise = 0.5)); connect!(b, :ab, :a, :a); connect!(b, :ab, :b, :b)
    factor!(b, :bc, HardConstraintFactor((a = 1, b = 1), 1)); connect!(b, :bc, :a, :b); connect!(b, :bc, :b, :c)
    factor!(b, :cd, NonlinearFactor(x -> sin.(x), 1 => 1; noise = 0.1)); connect!(b, :cd, :x, :c); connect!(b, :cd, :y, :d)
    factor!(b, :da, LinearConstraintFactor((a = 1, b = 1), 1; noise = 2.0)); connect!(b, :da, :a, :d); connect!(b, :da, :b, :a)
    g = validate(build(b))
    one1 = ones(1, 1)
    ps = (pa = NamedTuple(), ab = (A = (a = one1, b = -one1), c = [0.0]), bc = (A = (a = one1, b = -one1), c = [0.0]),
          cd = NamedTuple(), da = (A = (a = one1, b = -one1), c = [0.0]))
    w = coupling_weights(g, ps)
    @test w[:bc] == 1.0 && w[:cd] == 0.0 && w[:pa] == 0.0
    # D_a = 1 + 2 + 0.5, D_b = 2 (the hard factor adds nothing), D_d = 0.5
    @test w[:ab] ≈ 2 / sqrt(3.5 * 2)
    @test w[:da] ≈ 0.5 / sqrt(0.5 * 3.5)
    @test coupling_weights(g, ps; default = 0.3, hard = 7)[:cd] == 0.3
    @test coupling_weights(g, ps; default = 0.3, hard = 7)[:bc] == 7.0
    wr = coupling_weights(g, ps; normalize = false)
    @test wr[:ab] ≈ 2 && wr[:da] ≈ 0.5 && wr[:bc] ≈ 3
    @test_throws ArgumentError coupling_weights(g, ps; default = Inf)
    @test_throws ArgumentError coupling_weights(g, ps; hard = NaN)
    # a zero coefficient: no coupling, and D with a zero direction is handled
    ps0 = merge(ps, (ab = (A = (a = zeros(1, 1), b = -one1), c = [0.0]),))
    @test coupling_weights(g, ps0)[:ab] == 0.0
    # the weights drive the schedule
    @test spanning_tree_schedule(g; weights = w) isa SequentialSchedule
end

@testset "I25 the coupling tree is the precision tree on anisotropic grids" begin
    n = 6
    for mode in (:horizontal, :vertical, :random)
        rng = Xoshiro(2511); pv = 100.0
        name(i, j) = Symbol(:x, i, :_, j); idx(i, j) = (j-1)*n + i
        b = GraphBuilder(); pd = Dict{Symbol,Any}(); qs = Dict{Symbol,Float64}()
        m = randn(rng, n*n); J = Matrix(1.0I, n*n, n*n) / pv
        for j in 1:n, i in 1:n
            v, f = name(i, j), Symbol(:p, i, :_, j)
            variable!(b, v, 1)
            factor!(b, f, GaussianPrior(:x, [m[idx(i, j)]], fill(pv, 1, 1)))
            connect!(b, f, :x, v); pd[f] = NamedTuple(); qs[f] = 1.0
        end
        for j in 1:n, i in 1:n, (di, dj) in ((1, 0), (0, 1))
            (i+di <= n && j+dj <= n) || continue
            q = mode === :random ? 10.0^(-2 + 3rand(rng)) : mode === :horizontal ? (di == 1 ? 0.01 : 1.0) : (dj == 1 ? 0.01 : 1.0)
            f = Symbol(:s, i, :_, j, :_, di)
            factor!(b, f, LinearConstraintFactor((a = 1, b = 1), 1; noise = fill(q, 1, 1)))
            connect!(b, f, :a, name(i, j)); connect!(b, f, :b, name(i+di, j+dj))
            pd[f] = (A = (a = ones(1, 1), b = -ones(1, 1)), c = [0.0]); qs[f] = q
            a, c = idx(i, j), idx(i+di, j+dj)
            J[a, a] += 1/q; J[c, c] += 1/q; J[a, c] -= 1/q; J[c, a] -= 1/q
        end
        g = validate(build(b)); ps = NamedTuple(pd)
        st = NamedTuple(k => NamedTuple() for k in keys(ps))
        raw = coupling_weights(g, ps; normalize = false)
        for (k, q) in qs
            startswith(String(k), "s") && @test raw[k] ≈ 1/q
        end
        precision = Dict(k => startswith(String(k), "s") ? 1/q : 0.0 for (k, q) in qs)
        @test Mycelium._spanning_forests(g; weights = raw) == Mycelium._spanning_forests(g; weights = precision)
        # normalised, every strong edge of a uniform anisotropic grid is still in the tree (the weak
        # ones are chosen by their variables' precision, not by index)
        if mode !== :random
            tree = only(Mycelium._spanning_forests(g; weights = coupling_weights(g, ps)))
            strong = [qs[g.factors[e.factor].name] == 0.01 for e in g.edges]
            @test all(tree[strong])
        end
        marg, rep, _, _ = infer!(g, spanning_tree_schedule(g; weights = coupling_weights(g, ps)), ps, st;
                                 maxsweeps = 5000, tol = 1e-11)
        @test rep.converged
        exact = J \ (m / pv)
        @test maximum(abs(only(belief_mean(marg[g.variables[i].name])) - exact[i]) for i in 1:nvariables(g)) < 1e-7
    end
end
