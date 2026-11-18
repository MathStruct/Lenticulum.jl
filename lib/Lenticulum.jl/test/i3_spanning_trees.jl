# Weighted tree orders and alternating forests against independent dense Gaussian solves.
@testset "I3 weighted Gaussian trees: one sweep is exact" begin
    GO = GaussianOracle
    for seed in 1:6, dim in (1, 2)
        rng = Xoshiro(seed)
        t = GO.random_tree(rng, 6; maxdim = dim, pobs = 0.5)
        while !all(==(dim), t.dims)
            t = GO.random_tree(rng, 6; maxdim = dim, pobs = 0.5)
        end
        g, ps, st = GO.graph(t)
        μ, Σ, _, rg = GO.dense(t)
        w = Dict(f.name => randn(rng) for f in g.factors)
        for root in (nothing, :x3)
            store = MessageStore(g)
            sweep!(store, g, spanning_tree_schedule(g; weights = w, root), ps, st)
            for i in eachindex(t.dims)
                b = marginal(store, g, g.variable_index[Symbol(:x, i)])
                @test belief_mean(b) ≈ μ[rg(i)] atol = 1e-8
                @test belief_cov(b) ≈ Σ[rg(i), rg(i)] atol = 1e-8
            end
        end
    end
end

@testset "I3 loopy Gaussian means against dense solve" begin
    for seed in 1:5, disconnected in (false, true)
        rng = Xoshiro(seed)
        n = disconnected ? 8 : 5
        b = GraphBuilder(); pd = Dict{Symbol,Any}()
        J = zeros(n, n); h = zeros(n)
        for i in 1:n
            v, f = Symbol(:x, i), Symbol(:p, i)
            variable!(b, v, 1)
            precision = 0.5 + rand(rng); m = randn(rng)
            factor!(b, f, GaussianPrior(:x, [m], fill(inv(precision), 1, 1)))
            connect!(b, f, :x, v)
            pd[f] = NamedTuple(); J[i, i] += precision; h[i] += precision * m
        end
        pairs = disconnected ? [(1,2), (2,3), (3,4), (4,1), (5,6), (6,7), (7,8), (8,5)] :
                               [(1,2), (2,3), (3,4), (4,5), (5,1), (1,3)]
        for (k, (i, j)) in enumerate(pairs)
            f = Symbol(:c, k); q = 0.3 + rand(rng)
            factor!(b, f, LinearConstraintFactor((a = 1, b = 1), 1; noise = fill(q, 1, 1)))
            connect!(b, f, :a, Symbol(:x, i)); connect!(b, f, :b, Symbol(:x, j))
            pd[f] = (A = (a = ones(1,1), b = -ones(1,1)), c = [0.0])
            J[i,i] += 1/q; J[j,j] += 1/q; J[i,j] -= 1/q; J[j,i] -= 1/q
        end
        @test all(J[i,i] > sum(abs, J[i,:]) - J[i,i] for i in 1:n)
        g = validate(build(b)); ps = NamedTuple(pd)
        st = NamedTuple(k => NamedTuple() for k in keys(ps))
        exact = J \ h
        weights = Dict(f.name => randn(rng) for f in g.factors)
        for sched in (spanning_tree_schedule(g), spanning_tree_schedule(g; weights),
                      spanning_tree_schedule(g; ntrees = 2), spanning_tree_schedule(g; ntrees = :cover),
                      spanning_tree_schedule(g; weights, ntrees = :cover),
                      AndersonSchedule(spanning_tree_schedule(g; weights, ntrees = 2); m = 3))
            marg, rep, _, _ = infer!(g, sched, ps, st; tol = 1e-11, maxsweeps = 20_000)
            @test rep.converged
            @test [only(belief_mean(marg[Symbol(:x,i)])) for i in 1:n] ≈ exact atol = 1e-8
        end
    end
end

@testset "I3 weighted tree on a loopy graph: one sweep is exact when the off-tree factors are void" begin
    # A ring with chords whose strong constraints form a random spanning path; every other
    # constraint has noise 1e12, so it carries (almost) nothing. With weights 1/q the weighted
    # tree is the path, and one sweep of its order must give the dense solve's marginals: this
    # checks the inward/outward order of a weighted tree that BFS would not choose.
    differs = 0
    for seed in 1:8
        rng = Xoshiro(seed)
        n = 7
        perm = randperm(rng, n)
        strong = Set(minmax(perm[k], perm[k + 1]) for k in 1:n-1)
        pairs = unique([[minmax(i, mod1(i + 1, n)) for i in 1:n]; [minmax(i, mod1(i + 3, n)) for i in 1:n];
                        collect(strong)])
        b = GraphBuilder(); pd = Dict{Symbol,Any}(); w = Dict{Symbol,Float64}()
        J = zeros(n, n); h = zeros(n)
        for i in 1:n
            v, f = Symbol(:x, i), Symbol(:p, i)
            variable!(b, v, 1)
            prec = 0.5 + rand(rng); m = randn(rng)
            factor!(b, f, GaussianPrior(:x, [m], fill(inv(prec), 1, 1)))
            connect!(b, f, :x, v)
            pd[f] = NamedTuple(); w[f] = 1.0; J[i, i] += prec; h[i] += prec * m
        end
        for (k, (i, j)) in enumerate(pairs)
            f = Symbol(:c, k); q = (i, j) in strong ? 0.3 + rand(rng) : 1e12
            factor!(b, f, LinearConstraintFactor((a = 1, b = 1), 1; noise = fill(q, 1, 1)))
            connect!(b, f, :a, Symbol(:x, i)); connect!(b, f, :b, Symbol(:x, j))
            pd[f] = (A = (a = ones(1, 1), b = -ones(1, 1)), c = [0.0]); w[f] = 1 / q
            J[i, i] += 1/q; J[j, j] += 1/q; J[i, j] -= 1/q; J[j, i] -= 1/q
        end
        g = validate(build(b)); ps = NamedTuple(pd)
        st = NamedTuple(k => NamedTuple() for k in keys(ps))
        Σ = inv(J); μ = J \ h
        forest = only(Mycelium._spanning_forests(g; weights = w))
        differs += forest != only(Mycelium._spanning_forests(g))
        # every strong constraint lies wholly in the tree; a void one hangs on it by one edge
        for (fi, f) in enumerate(g.factors)
            intree = count(ei -> forest[ei], g.edges_of_factor[fi])
            @test intree == (w[f.name] > 1e-6 ? length(g.edges_of_factor[fi]) : 1)
        end
        for root in (nothing, Symbol(:x, perm[end]))
            store = MessageStore(g)
            sweep!(store, g, spanning_tree_schedule(g; weights = w, root), ps, st)
            for i in 1:n
                bel = marginal(store, g, g.variable_index[Symbol(:x, i)])
                @test only(belief_mean(bel)) ≈ μ[i] atol = 1e-8
                @test only(belief_cov(bel)) ≈ Σ[i, i] atol = 1e-8
            end
        end
    end
    @test differs > 0   # the weighted tree is not the breadth-first one on some seeds
end
