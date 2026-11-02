using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

# Plates (`Extending Mycelium.md` §2, L-SPEC-006): a template replicated over n observations with
# tied parameters. Oracles: the loop of n separate graphs (marginals, free energy, gradients),
# closed-form -log p for linear-Gaussian models, and the sequential schedule for grouped sweeps.

pl_lognormal(x, μ, Σ) = -(dot(x - μ, Σ \ (x - μ)) + logdet(2π * Σ)) / 2
pl_edge(g, f, ch) = findfirst(e -> e.factor == g.factor_index[f] && e.channel === ch, g.edges)
pl_flat(x) = LenticulumCore._flatten(x)

# Factor analysis: z ~ N(0, I_k), x = W z + c + ε, ε ~ N(0, σ² I_d), x observed.
const PL_D, PL_K, PL_σ2 = 3, 2, 0.7
pl_lik() = LinearConstraintFactor((x = PL_D, z = PL_K), PL_D; noise = PL_σ2)
function pl_fa_template!(p, x)
    variable!(p, :z, PL_K); variable!(p, :x, PL_D)
    factor!(p, :prior, GaussianPrior(:z, zeros(PL_K), Matrix(1.0I, PL_K, PL_K)))
    factor!(p, :lik, pl_lik()); factor!(p, :data, DataFactor(:x, x))
    connect!(p, :prior, :z, :z); connect!(p, :lik, :z, :z)
    connect!(p, :lik, :x, :x; direction = Absorbing()); connect!(p, :data, :x, :x; direction = Emitting())
end
function pl_fa_graph(X)
    b = GraphBuilder()
    plate!((p, i) -> pl_fa_template!(p, X[:, i]), b, :obs, size(X, 2))
    return build(b)
end
pl_fa_lik(ps, W, c) = merge(ps.lik, (A = (x = Matrix(1.0I, PL_D, PL_D), z = -W), c = c))
pl_fa_nlp(X, W, c) = -sum(pl_lognormal(X[:, i], c, W * W' + PL_σ2 * I) for i in axes(X, 2))

@testset "plates: one template, many observations, tied parameters" begin
    rng = Xoshiro(7)
    N = 6
    X = randn(rng, PL_D, N); W = randn(rng, PL_D, PL_K); c = randn(rng, PL_D)

    @testset "the builder: names, the record, keys" begin
        g = pl_fa_graph(X)
        @test nvariables(g) == 2N && nfactors(g) == 3N && nedges(g) == 4N
        @test haskey(g.variable_index, Symbol("z[3]")) && replica_name(:z, 3) === Symbol("z[3]")
        pl = only(plates(g))
        @test pl.name === :obs && pl.n == N && size(pl.edges) == (4, N) && size(pl.factors) == (3, N)
        @test [g.variables[pl.variables[1, i]].name for i in 1:N] == [replica_name(:z, i) for i in 1:N]
        @test parameter_keys(g) == (:prior, :lik, :data)
        @test length(tied_factors(g, :lik)) == N
        @test parameter_key(factornode(g, Symbol("lik[2]"))) === :lik
        ps = LuxCore.initialparameters(Xoshiro(0), g)
        @test keys(ps) == (:prior, :lik, :data)
        @test occursin("forest", sprint(show, g))
        # untied: one entry per replica
        b = GraphBuilder()
        plate!(b, :obs, 2) do p, i
            variable!(p, :x, 1)
            factor!(p, :prior, GaussianPrior(:x, 0.0, 1.0); tie = false)
            connect!(p, :prior, :x, :x)
        end
        @test parameter_keys(build(b)) == (Symbol("prior[1]"), Symbol("prior[2]"))
    end

    @testset "the builder refuses what is not one template" begin
        b = GraphBuilder()
        @test_throws ArgumentError plate!(b, :obs, 2) do p, i
            variable!(p, :x, 1)
            i == 2 && variable!(p, :y, 1)
        end
        b = GraphBuilder()
        @test_throws ArgumentError plate!((p, i) -> nothing, b, :obs, 0)
        plate!((p, i) -> variable!(p, :x, 1), b, :obs, 1)
        @test_throws ArgumentError plate!((p, i) -> nothing, b, :obs, 1)
        # a replica may not reach into another replica
        b = GraphBuilder()
        @test_throws ArgumentError plate!(b, :obs, 2) do p, i
            variable!(p, :x, 1)
            factor!(p, :prior, GaussianPrior(:x, 0.0, 1.0))
            connect!(p, :prior, :x, i == 2 ? Symbol("x[1]") : :x)
        end
        # tied factors must have one type; a key may not be another node's name
        b = GraphBuilder()
        variable!(b, :x, 1)
        factor!(b, :a, GaussianPrior(:x, 0.0, 1.0)); factor!(b, :b, DataFactor(:x, [0.0]); params = :a)
        @test_throws ArgumentError build(b)
        b = GraphBuilder()
        variable!(b, :x, 1)
        factor!(b, :a, GaussianPrior(:x, 0.0, 1.0)); factor!(b, :b, GaussianPrior(:x, 0.0, 1.0); params = :a)
        @test build(b) isa FactorGraph
        factor!(b, :c, GaussianPrior(:x, 0.0, 1.0); params = :b)
        @test_throws ArgumentError build(b)
    end

    @testset "a plate equals the loop of its graphs, and the closed form" begin
        g = pl_fa_graph(X)
        ps0 = LuxCore.initialparameters(Xoshiro(0), g)
        ps = merge(ps0, (lik = pl_fa_lik(ps0, W, c),))
        st = LuxCore.initialstates(Xoshiro(0), g)
        sched = plate_schedule(g)
        @test sched isa PlateSchedule && all(length.(sched.groups) .== N) && length(sched.groups) == 6
        s = MessageStore(g)
        _, st = propagate!(s, g, sched, ps, st; maxsweeps = 3)
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, st))
        @test F ≈ pl_fa_nlp(X, W, c) atol = 1e-10
        Floop = 0.0
        Gloop = zero(pl_flat(ps.lik))
        for i in 1:N
            gi = pl_fa_graph(X[:, i:i])
            psi = merge(LuxCore.initialparameters(Xoshiro(0), gi), (lik = ps.lik,))
            _, _, sti, si = infer!(gi, tree_schedule(gi), psi, LuxCore.initialstates(Xoshiro(0), gi))
            Floop += first(LenticulumCore.scalar_free_energy(si, gi, psi, sti))
            Gloop .+= pl_flat(LenticulumCore.free_energy_gradient(si, gi, psi, sti; ad = AutoForwardDiff()).lik)
            mi, ml = marginal(si, gi, gi.variable_index[Symbol("z[1]")]), marginal(s, g, g.variable_index[replica_name(:z, i)])
            @test belief_mean(ml) ≈ belief_mean(mi) atol = 1e-12
            @test belief_cov(ml) ≈ belief_cov(mi) atol = 1e-12
        end
        @test F ≈ Floop atol = 1e-10
        # the gradient of the tied entry is the sum over replicas; against the closed form too
        Wc(θ) = (L = LenticulumCore._unflatten(θ, ps.lik); (-L.A.z, L.c))
        Gexact = ForwardDiff.gradient(θ -> pl_fa_nlp(X, Wc(θ)...), pl_flat(ps.lik))
        for ad in (AutoForwardDiff(), AutoZygote())
            G = LenticulumCore.free_energy_gradient(s, g, ps, st; ad)
            @test keys(G) == (:prior, :lik, :data)
            @test pl_flat(G.lik) ≈ Gloop atol = 1e-10
            # the A.x block is not identified by -log p of this parametrisation; compare W and c
            lG, lE = LenticulumCore._unflatten(pl_flat(G.lik), ps.lik), LenticulumCore._unflatten(Gexact, ps.lik)
            @test lG.A.z ≈ lE.A.z atol = 1e-8
            @test lG.c ≈ lE.c atol = 1e-8
        end
        Gfd = LenticulumCore.free_energy_gradient(s, g, ps, st)
        @test pl_flat(Gfd.lik) ≈ Gloop atol = 1e-6
        # a parameter step moves the shared entry by the summed gradient
        G = LenticulumCore.free_energy_gradient(s, g, ps, st; ad = AutoForwardDiff())
        ps2, _, _ = parameter_step(s, g, ps, st; ad = AutoForwardDiff(), rule = GradientDescent(0.01))
        @test pl_flat(ps2.lik) ≈ pl_flat(ps.lik) .- 0.01 .* pl_flat(G.lik) atol = 1e-12
    end

    @testset "a grouped sweep gives the sequential schedule's messages" begin
        g = pl_fa_graph(X)
        ps0 = LuxCore.initialparameters(Xoshiro(0), g)
        ps = merge(ps0, (lik = pl_fa_lik(ps0, W, c),))
        st = LuxCore.initialstates(Xoshiro(0), g)
        sched = plate_schedule(g)
        s1, s2 = MessageStore(g), MessageStore(g)
        for _ in 1:2
            sweep!(s1, g, sched, ps, st)
            sweep!(s2, g, SequentialSchedule(tasks(sched)), ps, st)
        end
        for i in 1:nedges(g), (a, b) in ((s1.to_variable[i], s2.to_variable[i]), (s1.to_factor[i], s2.to_factor[i]))
            @test (a === nothing) == (b === nothing)
            a === nothing || @test belief_distance(a.belief, b.belief) < 1e-12
        end
        # replicas whose data differ but whose inputs are not all points still agree (per-replica path)
        msgs = [(z = GaussianBelief(randn(rng, PL_K), Matrix((1.0 + i) * I, PL_K, PL_K)),) for i in 1:3]
        fs = [pl_lik() for _ in 1:3]
        bs, _ = Mycelium.factor_messages(fs, :x, nothing, msgs, fill(nothing, 3), ps.lik, NamedTuple())
        for i in 1:3
            b1, _ = factor_message(fs[i], :x, nothing, msgs[i], nothing, ps.lik, NamedTuple())
            @test belief_distance(bs[i], b1) < 1e-12
        end
    end

    @testset "what a grouped sweep and tying refuse" begin
        # a hand-built group of tied nodes on different rules runs member by member
        b = GraphBuilder()
        for (j, rule) in enumerate((BeliefPropagation(), MeanField()))
            u, v, y = Symbol(:u, j), Symbol(:v, j), Symbol(:y, j)
            variable!(b, u, 1); variable!(b, v, 1); variable!(b, y, 1)
            factor!(b, Symbol(:pu, j), GaussianPrior(u, 0.2, 1.0)); factor!(b, Symbol(:pv, j), GaussianPrior(v, -0.1, 2.0))
            factor!(b, Symbol(:lc, j), LinearConstraintFactor(NamedTuple{(:y, :u, :v)}((1, 1, 1)), 1; noise = 0.5);
                    rule, params = :lc)
            factor!(b, Symbol(:d, j), DataFactor(:y, [0.7 * j]))
            connect!(b, Symbol(:pu, j), u, u); connect!(b, Symbol(:pv, j), v, v)
            connect!(b, Symbol(:lc, j), :u, u); connect!(b, Symbol(:lc, j), :v, v)
            connect!(b, Symbol(:lc, j), :y, y; direction = Absorbing()); connect!(b, Symbol(:d, j), :y, y; direction = Emitting())
        end
        g = build(b)
        ps = merge(LuxCore.initialparameters(Xoshiro(0), g),
                   (lc = (A = (y = fill(1.0, 1, 1), u = fill(-1.0, 1, 1), v = fill(-1.0, 1, 1)), c = [0.1]),))
        T = MessageTask
        e(f, ch) = pl_edge(g, f, ch)
        init = [T(e(f, ch), :to_variable) for (f, ch) in ((:d1, :y), (:d2, :y), (:pu1, :u1), (:pu2, :u2), (:pv1, :v1), (:pv2, :v2))]
        cav = [T(e(f, ch), :to_factor) for (f, ch) in ((:lc1, :y), (:lc2, :y), (:lc1, :v), (:lc2, :v))]
        grp = PlateSchedule([init[1:2], init[3:4], init[5:6], cav[1:2], cav[3:4], [T(e(:lc1, :u), :to_variable), T(e(:lc2, :u), :to_variable)]])
        s1, s2 = MessageStore(g), MessageStore(g)
        sweep!(s1, g, grp, ps, NamedTuple()); sweep!(s2, g, SequentialSchedule(tasks(grp)), ps, NamedTuple())
        for i in 1:nedges(g)
            a, c2 = s1.to_variable[i], s2.to_variable[i]
            a === nothing || @test belief_distance(a.belief, c2.belief) < 1e-14
        end
        # tied factors must be stateless
        gf = pl_fa_graph(X[:, 1:2])
        psf = merge(LuxCore.initialparameters(Xoshiro(0), gf), (lik = pl_fa_lik(LuxCore.initialparameters(Xoshiro(0), gf), W, c),))
        sf = MessageStore(gf)
        sweep!(sf, gf, plate_schedule(gf), psf, NamedTuple())
        @test_throws ArgumentError factor_free_energies(sf, gf, psf, (prior = NamedTuple(), lik = (n = 1,), data = NamedTuple()))
        @test factor_free_energies(sf, gf, psf, LuxCore.initialstates(Xoshiro(0), gf)) isa Tuple
    end

    @testset "a shared latent variable couples the replicas" begin
        # μ ~ N(m0, v0) shared, x_i = μ + c + ε_i: a star, a tree when unrolled
        m0, v0, σ2, n = 0.4, 2.0, 0.5, 5
        xs = randn(rng, n)
        b = GraphBuilder()
        variable!(b, :mu, 1)
        factor!(b, :pm, GaussianPrior(:mu, m0, v0)); connect!(b, :pm, :mu, :mu)
        plate!(b, :obs, n) do p, i
            variable!(p, :x, 1)
            factor!(p, :lik, LinearConstraintFactor((x = 1, mu = 1), 1; noise = σ2))
            factor!(p, :data, DataFactor(:x, [xs[i]]))
            connect!(p, :lik, :x, :x; direction = Absorbing()); connect!(p, :lik, :mu, :mu)
            connect!(p, :data, :x, :x; direction = Emitting())
        end
        g = build(b)
        @test istree(g)
        @test_throws ArgumentError plate_schedule(g)
        ps0 = LuxCore.initialparameters(Xoshiro(0), g)
        cc = 0.3
        ps = merge(ps0, (lik = (A = (x = fill(1.0, 1, 1), mu = fill(-1.0, 1, 1)), c = [cc]),))
        _, _, st, s = infer!(g, tree_schedule(g), ps, LuxCore.initialstates(Xoshiro(0), g))
        P = 1 / v0 + n / σ2
        @test only(belief_mean(marginal(s, g, g.variable_index[:mu]))) ≈ (m0 / v0 + sum(xs .- cc) / σ2) / P atol = 1e-10
        nlp(c1) = -pl_lognormal(xs, fill(m0 + c1, n), σ2 * I + v0 * ones(n, n))
        @test first(LenticulumCore.scalar_free_energy(s, g, ps, st)) ≈ nlp(cc) atol = 1e-10
        G = LenticulumCore.free_energy_gradient(s, g, ps, st; ad = AutoForwardDiff())
        @test only(G.lik.c) ≈ ForwardDiff.derivative(nlp, cc) atol = 1e-10
        # a replica schedule lifted to all replicas; the shared target runs member by member
        T = MessageTask
        rt = [T(pl_edge(g, Symbol("data[1]"), :x), :to_variable), T(pl_edge(g, Symbol("lik[1]"), :x), :to_factor),
              T(pl_edge(g, Symbol("lik[1]"), :mu), :to_variable)]
        ls = plate_schedule(g, vcat([T(pl_edge(g, :pm, :mu), :to_variable)], rt))
        @test length(ls.groups) == 4 && length(ls.groups[1]) == 1 && all(length.(ls.groups[2:end]) .== n)
        @test_throws ArgumentError plate_schedule(g, [T(pl_edge(g, Symbol("lik[2]"), :mu), :to_variable)])
        s1, s2 = MessageStore(g), MessageStore(g)
        sweep!(s1, g, ls, ps, st); sweep!(s2, g, SequentialSchedule(tasks(ls)), ps, st)
        @test belief_mean(marginal(s1, g, g.variable_index[:mu])) == belief_mean(marginal(s2, g, g.variable_index[:mu]))
    end

    @testset "a plate of hybrid graphs: mean-field nodes inside" begin
        # u, v ~ priors, y_i = u_i + v_i + c + ε, the sum node on mean field, c shared
        μu, Vu, μv, Vv, σ2, n = 0.3, 1.5, -0.4, 0.8, 0.5, 4
        ys = randn(rng, n)
        function template!(p, y)
            variable!(p, :u, 1); variable!(p, :v, 1); variable!(p, :y, 1)
            factor!(p, :pu, GaussianPrior(:u, μu, Vu)); factor!(p, :pv, GaussianPrior(:v, μv, Vv))
            factor!(p, :lc, LinearConstraintFactor((y = 1, u = 1, v = 1), 1; noise = σ2); rule = MeanField())
            factor!(p, :data, DataFactor(:y, [y]))
            connect!(p, :pu, :u, :u); connect!(p, :pv, :v, :v)
            connect!(p, :lc, :u, :u); connect!(p, :lc, :v, :v); connect!(p, :lc, :y, :y; direction = Absorbing())
            connect!(p, :data, :y, :y; direction = Emitting())
        end
        function hybrid(ys)
            b = GraphBuilder()
            plate!((p, i) -> template!(p, ys[i]), b, :obs, length(ys))
            g = build(b)
            ps = merge(LuxCore.initialparameters(Xoshiro(0), g),
                       (lc = (A = (y = fill(1.0, 1, 1), u = fill(-1.0, 1, 1), v = fill(-1.0, 1, 1)), c = [0.1]),))
            T = MessageTask
            r(f, ch, k) = T(pl_edge(g, replica_name(f, 1), ch), k)
            init = plate_schedule(g, [r(:data, :y, :to_variable), r(:pu, :u, :to_variable), r(:pv, :v, :to_variable)])
            sweep = plate_schedule(g, [r(:lc, :u, :to_variable), r(:pu, :u, :to_factor),
                                       r(:lc, :v, :to_variable), r(:pv, :v, :to_factor)])
            s = MessageStore(g)
            sweep!(s, g, init, ps, NamedTuple())
            report, _ = propagate!(s, g, sweep, ps, NamedTuple(); maxsweeps = 20_000, tol = 1e-14)
            return g, ps, s, report
        end
        g, ps, s, report = hybrid(ys)
        @test report.converged
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple()))
        G = LenticulumCore.free_energy_gradient(s, g, ps, NamedTuple(); ad = AutoForwardDiff())
        parts = [hybrid(ys[i:i]) for i in 1:n]
        @test F ≈ sum(first(LenticulumCore.scalar_free_energy(q[3], q[1], q[2], NamedTuple())) for q in parts) atol = 1e-10
        @test pl_flat(G.lc) ≈ sum(pl_flat(LenticulumCore.free_energy_gradient(q[3], q[1], q[2], NamedTuple();
                                                                            ad = AutoForwardDiff()).lc) for q in parts) atol = 1e-10
        # and the derivative of the converged hybrid free energy, by central differences in c
        h = 1e-5
        Fat(c1) = begin
            gq, psq, _, _ = hybrid(ys)
            psq = merge(psq, (lc = merge(psq.lc, (c = [c1],)),))
            T = MessageTask
            r(f, ch, k) = T(pl_edge(gq, replica_name(f, 1), ch), k)
            sq = MessageStore(gq)
            sweep!(sq, gq, plate_schedule(gq, [r(:data, :y, :to_variable), r(:pu, :u, :to_variable), r(:pv, :v, :to_variable)]), psq, NamedTuple())
            propagate!(sq, gq, plate_schedule(gq, [r(:lc, :u, :to_variable), r(:pu, :u, :to_factor),
                                                  r(:lc, :v, :to_variable), r(:pv, :v, :to_factor)]), psq, NamedTuple();
                       maxsweeps = 20_000, tol = 1e-14)
            first(LenticulumCore.scalar_free_energy(sq, gq, psq, NamedTuple()))
        end
        @test only(G.lc.c) ≈ (Fat(0.1 + h) - Fat(0.1 - h)) / 2h atol = 1e-6
    end
end
