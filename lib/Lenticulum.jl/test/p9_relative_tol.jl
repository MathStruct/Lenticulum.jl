using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, ForwardDiff
using LenticulumCore: DiracBelief, CategoricalBelief, MixtureBelief, ExtendedGaussian, extended
using Mycelium: GaussianBlock, PointBlock, block_relative_residual, replica, nreplicas

# P9: a relative convergence criterion (`propagate!(...; rtol)`, `relative_residual`). Oracles: the
# dense joint solve of a loopy linear-Gaussian model, the same model in other units (the criterion
# must not see the units), and the replica-by-replica residual for the block fast paths.

const P9_R1, P9_R2, P9_VS = 2.0, 3.0, 10.0

# The voltage divider of runtests.jl (one cycle) in units c: every value scales by c, every
# variance by c², every precision by 1/c². Returns the graph and the exact means.
function p9_divider(c; pv = 1e4)
    Q = fill(1e-3 * c^2, 1, 1); wp = fill(pv * c^2, 1, 1); O = ones(1, 1)
    b = GraphBuilder()
    for v in (:v, :i1, :i2); variable!(b, v, 1); end
    factor!(b, :ohm1, LinearConstraintFactor((v = 1, i1 = 1), 1; noise = Q))
    factor!(b, :ohm2, LinearConstraintFactor((v = 1, i2 = 1), 1; noise = Q))
    factor!(b, :kcl,  LinearConstraintFactor((i1 = 1, i2 = 1), 1; noise = Q))
    for (f, a, z) in ((:ohm1, :v, :i1), (:ohm2, :v, :i2), (:kcl, :i1, :i2))
        connect!(b, f, a, a; direction = Bidirectional()); connect!(b, f, z, z; direction = Bidirectional())
    end
    for v in (:v, :i1, :i2)
        factor!(b, Symbol(:reg, v), GaussianPrior(v, [0.0], wp))
        connect!(b, Symbol(:reg, v), v, v; direction = Bidirectional())
    end
    g = validate(build(b))
    ps = merge((ohm1 = (A = (v = O, i1 = fill(P9_R1, 1, 1)), c = [P9_VS * c]),
                ohm2 = (A = (v = O, i2 = fill(-P9_R2, 1, 1)), c = [0.0]),
                kcl  = (A = (i1 = O, i2 = -O), c = [0.0])),
               NamedTuple(Symbol(:reg, v) => NamedTuple() for v in (:v, :i1, :i2)))
    st = NamedTuple(k => NamedTuple() for k in keys(ps))
    G = [1.0 P9_R1 0; 1 0 -P9_R2; 0 1 -1]
    Λ = G' * G / Q[1] + I / wp[1]
    return g, ps, st, Λ \ (G' * [P9_VS * c, 0, 0] / Q[1])
end
p9_means(marg) = [belief_mean(marg[v])[1] for v in (:v, :i1, :i2)]
p9_relerr(marg, m) = maximum(abs, p9_means(marg) .- m) / maximum(abs, m)

@testset "P9: a relative convergence criterion" begin
    @testset "relative_residual: groups, scales, limits" begin
        rng = Xoshiro(9)
        A = randn(rng, 3, 3); Λ = A * A' + I; η = randn(rng, 3)
        a = GaussianBelief(η, Λ); b = GaussianBelief(η .+ 1e-6 .* randn(rng, 3), Λ .+ 1e-6 .* Symmetric(randn(rng, 3, 3)))
        r = relative_residual(a, b, 0.0, 1e-8)
        # the definition: each group's largest change against its own largest entry
        @test r ≈ max(maximum(abs, a.η - b.η) / (1e-8 * max(maximum(abs, a.η), maximum(abs, b.η))),
                      maximum(abs, a.Λ - b.Λ) / (1e-8 * max(maximum(abs, a.Λ), maximum(abs, b.Λ))))
        # scale-free: messages of the same model in other units (η/c, Λ/c²) give the same residual
        for c in (1e-4, 1e4)
            sc(x) = GaussianBelief(x.η ./ c, x.Λ ./ c^2)
            @test relative_residual(sc(a), sc(b), 0.0, 1e-8) ≈ r rtol = 1e-10
        end
        # rtol = 0 is the absolute distance in units of atol
        @test relative_residual(a, b, 1e-8, 0.0) ≈ belief_distance(a, b) / 1e-8
        # no change is 0 even against a zero bound; a change against a zero bound is Inf
        @test relative_residual(a, a, 0.0, 1e-8) == 0
        z = GaussianBelief(zeros(3), zeros(3, 3), true)
        @test relative_residual(z, z, 0.0, 1e-8) == 0
        @test relative_residual(a, b, 0.0, 0.0) == Inf
        # a zero change that carries a tangent, against a zero bound (tol = 0, a zero group): 0,
        # not 0/0 (a dual's `<=` breaks ties by its partials; review of P9, probe 6c)
        Dz = ForwardDiff.Dual{Nothing}(0.0, 1.0)
        zd = GaussianBelief([Dz, Dz], zeros(2, 2), true)
        @test ForwardDiff.value(relative_residual(zd, GaussianBelief(zeros(2), zeros(2, 2), true), 0.0, 1e-8)) == 0
        # against an all-zero side the scale is the other side's: a message leaving zero counts in full
        @test relative_residual(z, GaussianBelief([2.0, 0, 0], zeros(3, 3), true), 0.0, 1e-8) ≈ 1e8
        @test relative_residual(GaussianBelief(zeros(2), zeros(2, 2), true), GaussianBelief(zeros(3), zeros(3, 3), true), 1e-8, 1e-8) == Inf
        # extended Gaussians: μ and Σ scale-free, the projector at scale 1
        ea, eb = extended(a), extended(b)
        re = relative_residual(ea, eb, 0.0, 1e-8)
        @test isfinite(re) && re > 0
        se(x, c) = ExtendedGaussian(x.μ .* c, x.Σ .* c^2, x.D)
        @test relative_residual(se(ea, 1e5), se(eb, 1e5), 0.0, 1e-8) ≈ re rtol = 1e-8
        @test relative_residual(ea, b, 0.0, 1e-8) ≈ relative_residual(ea, eb, 0.0, 1e-8)
        # points, mixtures, categoricals
        @test relative_residual(DiracBelief([3.0, 4.0]), DiracBelief([3.0, 4.0 + 5e-8]), 0.0, 1e-8) ≈ 1.0
        @test relative_residual(DiracBelief([3e6, 4e6]), DiracBelief([3e6, 4e6 + 5e-2]), 0.0, 1e-8) ≈ 1.0
        ma = MixtureBelief([a, a], [0.5, 0.5]); mb = MixtureBelief([a, b], [0.5, 0.5])
        @test relative_residual(ma, mb, 0.0, 1e-8) ≈ relative_residual(a, b, 0.0, 1e-8)
        mw = MixtureBelief([a, a], [0.5 + 1e-9, 0.5 - 1e-9])
        @test relative_residual(ma, mw, 0.0, 1e-8) ≈ 0.1 rtol = 1e-6          # total variation 1e-9 at scale 1
        ca, cb = CategoricalBelief([0.3, 0.7]), CategoricalBelief([0.3 + 2e-8, 0.7 - 2e-8])
        @test relative_residual(ca, cb, 1e-8, 1e-8) ≈ belief_distance(ca, cb) / 2e-8 rtol = 1e-6
    end

    @testset "blocks: the fast paths equal the replica-by-replica residual" begin
        rng = Xoshiro(91)
        for trial in 1:20
            d, n = rand(rng, 1:3), rand(rng, 1:5)
            atol, rtol = rand(rng, (0.0, 1e-10)), 10.0^rand(rng, -10:-6)
            sc = 10.0^rand(rng, -6:6)
            η = sc .* randn(rng, d, n); η2 = η .+ sc .* 1e-7 .* randn(rng, d, n)
            mk() = (B = randn(rng, d, d); sc^2 .* (B * B' + I))
            for shared in (true, false)
                Λa = shared ? mk() : [mk() for _ in 1:n]
                Λb = shared ? Λa .+ 1e-7 .* Λa : [L .+ 1e-7 .* randn(rng) .* L for L in Λa]
                A, B = GaussianBlock(η, Λa, fill(false, n)), GaussianBlock(η2, Λb, fill(false, n))
                ref = maximum(i -> relative_residual(replica(A, i), replica(B, i), atol, rtol), 1:n)
                @test block_relative_residual(A, B, atol, rtol) ≈ ref rtol = 1e-12
            end
            X = sc .* randn(rng, d, n); X2 = X .+ sc .* 1e-7 .* randn(rng, d, n)
            P, P2 = PointBlock(X), PointBlock(X2)
            ref = maximum(i -> relative_residual(replica(P, i), replica(P2, i), atol, rtol), 1:n)
            @test block_relative_residual(P, P2, atol, rtol) ≈ ref rtol = 1e-12
            @test block_relative_residual(P, P, atol, rtol) == 0
        end
    end

    @testset "loopy Gaussian BP: the criterion does not see the units" begin
        sweeps = Dict{Symbol,Vector{Int}}()
        for c in (1e-4, 1.0, 1e4)
            g, ps, st, m = p9_divider(c)
            for (name, sched) in ((:flooding, flooding_schedule(g)), (:spanning, spanning_tree_schedule(g)),
                                  (:anderson, AndersonSchedule(spanning_tree_schedule(g); m = 3)),
                                  (:residual, ResidualSchedule()))
                marg, rep, _, _ = infer!(g, sched, ps, st; maxsweeps = 3000, tol = 0.0, rtol = 1e-10)
                @test rep.converged
                @test rep.residual <= 1
                @test occursin("rtol", rep.reason)
                @test p9_relerr(marg, m) < 1e-8
                push!(get!(sweeps, name, Int[]), rep.sweeps)
            end
            # on marginals (means and covariances, each against its own size)
            marg, rep, _, _ = infer!(g, spanning_tree_schedule(g), ps, st; maxsweeps = 3000, tol = 0.0, rtol = 1e-10,
                                     criterion = :marginals)
            @test rep.converged && p9_relerr(marg, m) < 1e-8
            push!(get!(sweeps, :marginals, Int[]), rep.sweeps)
            # I22 (regression): flooding moves information one hop per sweep, so its second sweep
            # leaves every marginal at the prior while messages to factors still change; with one
            # quiet sweep `:marginals` stopped there (relative error 1). It now needs two in a row.
            for kw in ((tol = 1e-11 * c,), (tol = 0.0, rtol = 1e-10))
                marg, rep, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 3000, criterion = :marginals, kw...)
                @test rep.converged && rep.sweeps > 2 && p9_relerr(marg, m) < 1e-8
            end
        end
        # I24 (regression; LP-052 F1): a ResidualSchedule with maxtasks below the number of tasks
        # commits only part of a sweep, which can be messages to factors alone, so two quiet sweeps
        # of the marginals said nothing: `:marginals` stopped after 2 sweeps with every marginal
        # trivial. A partial residual sweep now also needs the largest pending message residual
        # within the bound, checked over every task when the run is about to stop (the last
        # sweep's own updates can raise residuals after its first: LP-054 F1 measured 2.6e-11
        # against tol = 1e-11 when only the sweep's first update was checked). Oracles: the dense
        # solve, and a scan of every pending residual after the report.
        g, ps, st, m = p9_divider(1.0)
        for mt in (1, 3), kw in ((tol = 1e-11,), (tol = 0.0, rtol = 1e-10))
            marg, rep, _, _ = infer!(g, ResidualSchedule(maxtasks = mt), ps, st; maxsweeps = 20000,
                                     criterion = :marginals, kw...)
            @test rep.converged && rep.sweeps > 2 && all(v -> marg[v] isa GaussianBelief, (:v, :i1, :i2)) &&
                  p9_relerr(marg, m) < 1e-8
            @test occursin("pending", rep.reason)
        end
        p9_pending(s, g, ps, st; damping = 0.0) = maximum(t -> step!(s, g, t, ps, st; damping, commit = false)[2], all_tasks(g))
        for tol in (1e-11, 5e-11, 1e-9), (mt, damping) in ((1, 0.0), (3, 0.3))
            s = MessageStore(g)
            rep, _ = propagate!(s, g, ResidualSchedule(maxtasks = mt), ps, st; maxsweeps = 30000, tol, damping,
                                criterion = :marginals)
            @test rep.converged && p9_pending(s, g, ps, st; damping) <= tol && rep.residual <= tol
            # warm start: a second run on the converged store stops soon (measured 4–15 sweeps against
            # 515–1106 cold: each sweep's updates can lift a residual above the bound for a while)
            rep2, _ = propagate!(s, g, ResidualSchedule(maxtasks = mt), ps, st; maxsweeps = 30000, tol, damping,
                                 criterion = :marginals)
            @test rep2.converged && rep2.sweeps < rep.sweeps ÷ 10 && p9_pending(s, g, ps, st; damping) <= tol
        end
        let s = MessageStore(g), kw = (tol = 0.0, rtol = 1e-10)
            rep, _ = propagate!(s, g, ResidualSchedule(maxtasks = 1), ps, st; maxsweeps = 30000, criterion = :marginals, kw...)
            pend = Base.ScopedValues.with(() -> p9_pending(s, g, ps, st), Mycelium._RELATIVE_TOL => (kw.tol, kw.rtol))
            @test rep.converged && pend <= 1
        end
        # a sweep that covers every task is unchanged: no pending-residual condition
        marg, rep, _, _ = infer!(g, ResidualSchedule(), ps, st; maxsweeps = 20000, tol = 1e-11, criterion = :marginals)
        @test rep.converged && !occursin("pending", rep.reason) && p9_relerr(marg, m) < 1e-8
        # the same model in other units needs the same sweeps (up to rounding, one or two)
        for (name, s) in sweeps
            @test maximum(s) - minimum(s) <= 2
        end
        # what the absolute criterion does with the same model (measured 2026-11-17): in small
        # units (precisions ~1e8) a tol of 1e-11 sits below the rounding of the messages and is
        # never met; in large units (precisions ~1e-8) tol = 1e-8 is met while the means are
        # still off by 3e-7 relative. The relative criterion avoids both.
        g, ps, st, m = p9_divider(1e-4)
        _, rep, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 1000, tol = 1e-11)
        @test !rep.converged
        g, ps, st, m = p9_divider(1e4)
        marg, rep, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 3000, tol = 1e-8)
        @test rep.converged && p9_relerr(marg, m) > 1e-8
        # tol is the floor of the relative bound: rtol = 0 keeps the absolute criterion unchanged
        g, ps, st, m = p9_divider(1.0)
        _, ra, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 3000, tol = 1e-8)
        _, rb, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 3000, tol = 1e-8, rtol = 0.0)
        @test ra.sweeps == rb.sweeps && ra.residual == rb.residual && ra.reason == "residual below tol"
        @test_throws ArgumentError infer!(g, flooding_schedule(g), ps, st; tol = 1e-8, rtol = -1.0)
        @test_throws ArgumentError infer!(g, flooding_schedule(g), ps, st; tol = -1.0, rtol = 1e-8)
        # tol = 0 without rtol stays legal, as before: it runs to maxsweeps
        _, r0, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps = 5, tol = 0.0)
        @test !r0.converged && r0.sweeps == 5
        # the relative tolerance is scoped to the call: a sweep outside reports absolute distances
        @test Mycelium._RELATIVE_TOL[] === nothing
    end

    @testset "plates: block residuals are relative too" begin
        # x_i ~ N(0, c²), y_i = x_i + noise of variance c²/2, y_i observed: a plate of n trees,
        # swept by the plate schedule (block messages, `_block_residual`)
        n = 4
        function plate_graph(c)
            b = GraphBuilder()
            plate!(b, :obs, n) do p, i
                variable!(p, :x, 1); variable!(p, :y, 1)
                factor!(p, :prior, GaussianPrior(:x, [0.0], fill(c^2, 1, 1)))
                factor!(p, :lik, LinearConstraintFactor((x = 1, y = 1), 1; noise = c^2 / 2))
                factor!(p, :data, DataFactor(:y, [c * i]))
                connect!(p, :prior, :x, :x); connect!(p, :lik, :x, :x)
                connect!(p, :lik, :y, :y; direction = Absorbing()); connect!(p, :data, :y, :y; direction = Emitting())
            end
            g = build(b)
            ps0 = LuxCore.initialparameters(Xoshiro(0), g)
            ps = merge(ps0, (lik = merge(ps0.lik, (A = (x = ones(1, 1), y = -ones(1, 1)), c = [0.0])),))
            return g, ps, LuxCore.initialstates(Xoshiro(0), g)
        end
        reps = map((1e-4, 1.0, 1e4)) do c
            g, ps, st = plate_graph(c)
            s = MessageStore(g)
            rep, _ = propagate!(s, g, plate_schedule(g), ps, st; maxsweeps = 20, tol = 0.0, rtol = 1e-12)
            @test rep.converged
            # the exact posterior mean of x_i: c·i·(1/(c²/2)) / (1/c² + 2/c²) = 2c·i/3
            for i in 1:n
                @test belief_mean(marginal(s, g, g.variable_index[replica_name(:x, i)]))[1] ≈ 2c * i / 3 rtol = 1e-10
            end
            rep
        end
        @test allequal(r.sweeps for r in reps)
    end

    # I23 (regression): an iterated `infer!` with dual parameters threw a `MethodError` in the
    # `ConvergenceReport` constructor (`residual::Float64` from a dual), with `rtol = 0` as well.
    # The report now holds the primal residual. Oracle: the exact means are linear in the source
    # `ohm1.c`, so their derivative is the dense solve with the source's direction, and Gaussian BP's
    # fixed point has the exact means.
    @testset "I23: iterated inference with dual parameters" begin
        g, ps, st, _ = p9_divider(1.0)
        G = [1.0 P9_R1 0; 1 0 -P9_R2; 0 1 -1]; q = 1e-3
        dm = (G' * G / q + I / 1e4) \ (G' * [1.0, 0, 0] / q)
        withsource(x) = merge(ps, (ohm1 = merge(ps.ohm1, (; c = [x])),))
        for (name, sched) in ((:flooding, flooding_schedule(g)), (:spanning, spanning_tree_schedule(g)),
                              (:residual, ResidualSchedule()), (:anderson, AndersonSchedule(spanning_tree_schedule(g)))),
            kw in ((tol = 1e-12,), (tol = 0.0, rtol = 1e-12), (tol = 1e-12, criterion = :marginals))
            reps = ConvergenceReport[]
            f(x) = begin
                marg, rep, _, _ = infer!(g, sched, withsource(x[1]), st; maxsweeps = 5000, kw...)
                push!(reps, rep)
                p9_means(marg)
            end
            J = ForwardDiff.jacobian(f, [P9_VS])
            rep = only(reps)
            @test rep.converged && rep.residual isa Float64 && isfinite(rep.residual)
            @test J[:, 1] ≈ dm rtol = 1e-6
            # about the same sweeps as without duals: dual matrices take the generic linear algebra,
            # whose rounding differs, so the residuals near the bound differ in the last digits
            # (measured: at most one sweep more or fewer)
            _, rep0, _, _ = infer!(g, sched, ps, st; maxsweeps = 5000, kw...)
            @test abs(rep.sweeps - rep0.sweeps) <= 1
        end
        # The tangents do not steer the iteration: the same tasks in the same order whatever the
        # partials (LP-053 F1: the residual schedule chose its task by comparing dual residuals,
        # and a dual breaks ties by its partials; damped, the run stopped at 111 or 110 sweeps).
        for (sched, kw) in ((ResidualSchedule(), (tol = 1e-12, damping = 0.3)), (ResidualSchedule(; maxtasks = 3), (tol = 1e-12,)),
                            (flooding_schedule(g), (tol = 1e-12,)))
            runs = map((1.0, -2.5, 0.0)) do t
                s = MessageStore(g); order = MessageTask[]
                rep, _ = propagate!(s, g, sched, withsource(ForwardDiff.Dual{Nothing}(P9_VS, t)), st;
                                    maxsweeps = 8000, record = order, kw...)
                (rep.converged, rep.sweeps, rep.residual, order)
            end
            @test all(r -> r[1], runs) && allequal(runs)
        end
        # not converged: the report of `maxsweeps` holds the primal residual too
        d = ForwardDiff.derivative(P9_VS) do x
            marg, rep, _, _ = infer!(g, flooding_schedule(g), withsource(x), st; maxsweeps = 3, tol = 0.0)
            @test !rep.converged && rep.sweeps == 3 && rep.residual isa Float64
            p9_means(marg)[1]
        end
        @test isfinite(d)
        # nested duals (second derivative): the means are linear in the source, so it is zero
        H = ForwardDiff.hessian(x -> p9_means(infer!(g, flooding_schedule(g), withsource(x[1]), st;
                                                      maxsweeps = 5000, tol = 1e-12)[1])[1], [P9_VS])
        @test abs(H[1, 1]) < 1e-8
    end
end
