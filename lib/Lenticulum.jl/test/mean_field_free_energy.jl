using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using MyceliumLearning
using LenticulumCore: DiracBelief, CategoricalBelief
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

# The merged free energy of Riegler et al. (2013), eq. (17): belief-propagation nodes keep their
# Bethe terms, mean-field nodes contribute E_{∏q}[-log f] with no entropy, and each variable's
# entropy is counted 1 - |N_BP(v)| times (`Extending Mycelium.md` §1). Oracles: naive mean field
# in closed form, Monte Carlo averages, exact enumeration, and central differences of the
# converged free energy for the gradient.

mf_lognormal(x, μ, Σ) = -(dot(x - μ, Σ \ (x - μ)) + logdet(2π * Σ)) / 2
mf_logsumexp(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
std_(v) = sqrt(sum(abs2, v .- sum(v) / length(v)) / (length(v) - 1))
mf_edge(g, f, ch) = findfirst(e -> e.factor == g.factor_index[f] && e.channel === ch, g.edges)

function mf_run(g, ps, init, sweep; tol = 1e-14, maxsweeps = 20_000)
    s = MessageStore(g)
    sweep!(s, g, SequentialSchedule(init), ps, NamedTuple())
    report, _ = propagate!(s, g, SequentialSchedule(sweep), ps, NamedTuple(); maxsweeps, tol)
    return s, report
end
mf_F(g, ps, init, sweep; tol = 1e-14) = (s = first(mf_run(g, ps, init, sweep; tol)); first(LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple())))
function mf_fd_gradient(g, ps, init, sweep; h = 1e-6, tol = 1e-14)
    θ = LenticulumCore._flatten(ps)
    F(t) = mf_F(g, LenticulumCore._unflatten(t, ps), init, sweep; tol)
    return [(F(θ .+ h .* (1:length(θ) .== i)) - F(θ .- h .* (1:length(θ) .== i))) / 2h for i in eachindex(θ)]
end
function mf_graph_gradient(g, ps, init, sweep; ad = nothing, tol = 1e-14)
    s, _ = mf_run(g, ps, init, sweep; tol)
    return LenticulumCore._flatten(LenticulumCore.free_energy_gradient(s, g, ps, NamedTuple(); ad))
end

@testset "the merged free energy of belief propagation and mean field (Riegler et al. 2013)" begin
    @testset "counting numbers count belief-propagation neighbours only" begin
        b = GraphBuilder()
        variable!(b, :u, 1); variable!(b, :v, 1)
        factor!(b, :pu, GaussianPrior(:u, 0.0, 1.0))
        factor!(b, :lc, LinearConstraintFactor((u = 1, v = 1), 1; noise = 1.0); rule = MeanField())
        factor!(b, :lb, LinearConstraintFactor((u = 1, v = 1), 1; noise = 1.0))
        connect!(b, :pu, :u, :u); connect!(b, :lc, :u, :u); connect!(b, :lc, :v, :v)
        connect!(b, :lb, :u, :u); connect!(b, :lb, :v, :v)
        g = build(b)
        @test bp_degree(g, g.variable_index[:u]) == 2 && variable_degree(g, g.variable_index[:u]) == 3
        @test counting_numbers(g) == (u = -1, v = 0)
    end

    # u ~ N(μu, Vu), v ~ N(μv, Vv), y = u + v + c + ε, ε ~ N(0, σ²), y observed; the sum node on
    # mean field. Its fixed point is naive mean field over (u, v), whose free energy is closed form.
    μu, Vu, μv, Vv, σ2, y = 0.3, 1.5, -0.4, 0.8, 0.5, 1.2
    function sum_graph(rule)
        b = GraphBuilder()
        variable!(b, :u, 1); variable!(b, :v, 1); variable!(b, :y, 1)
        factor!(b, :pu, GaussianPrior(:u, μu, Vu)); factor!(b, :pv, GaussianPrior(:v, μv, Vv))
        factor!(b, :lc, LinearConstraintFactor((y = 1, u = 1, v = 1), 1; noise = σ2); rule)
        factor!(b, :data, DataFactor(:y, [y]))
        connect!(b, :pu, :u, :u); connect!(b, :pv, :v, :v)
        connect!(b, :lc, :u, :u); connect!(b, :lc, :v, :v); connect!(b, :lc, :y, :y; direction = Absorbing())
        connect!(b, :data, :y, :y; direction = Emitting())
        g = build(b)
        ps = merge(LuxCore.initialparameters(Xoshiro(0), g),
                   (lc = (A = (y = fill(1.0, 1, 1), u = fill(-1.0, 1, 1), v = fill(-1.0, 1, 1)), c = [0.1]),))
        T = MessageTask
        init = [T(mf_edge(g, :data, :y), :to_variable), T(mf_edge(g, :pu, :u), :to_variable), T(mf_edge(g, :pv, :v), :to_variable)]
        sweep = [T(mf_edge(g, :lc, :u), :to_variable), T(mf_edge(g, :pu, :u), :to_factor),
                 T(mf_edge(g, :lc, :v), :to_variable), T(mf_edge(g, :pv, :v), :to_factor)]
        return g, ps, init, sweep
    end

    @testset "naive mean field: the closed form, the bound, and descent" begin
        g, ps, init, sweep = sum_graph(MeanField())
        c = only(ps.lc.c)
        # the mean-field fixed point by hand: each q is the prior times exp E_{q_other}[log f]
        su, sv = 1 / (1 / Vu + 1 / σ2), 1 / (1 / Vv + 1 / σ2)
        mu, mv = μu, μv
        for _ in 1:10_000
            mu = su * (μu / Vu + (y - c - mv) / σ2)
            mv = sv * (μv / Vv + (y - c - mu) / σ2)
        end
        prior_term(m, s, μ, V) = ((m - μ)^2 + s) / (2V) + log(2π * V) / 2 - log(2π * ℯ * s) / 2
        F_mf = prior_term(mu, su, μu, Vu) + prior_term(mv, sv, μv, Vv) +
               ((y - c - mu - mv)^2 + su + sv) / (2σ2) + log(2π * σ2) / 2
        s, report = mf_run(g, ps, init, sweep)
        @test report.converged
        @test only(belief_mean(marginal(s, g, g.variable_index[:u]))) ≈ mu atol = 1e-10
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple()))
        @test F ≈ F_mf atol = 1e-10
        # an upper bound on -log p(y), strictly, since mean field drops the posterior correlation
        nlp = -mf_lognormal([y], [μu + μv + c], fill(Vu + Vv + σ2, 1, 1))
        @test F > nlp + 1e-3
        # with belief propagation at the sum node it is Bethe on a tree: exact
        gb, psb, _, _ = sum_graph(BeliefPropagation())
        _, _, stb, sb = infer!(gb, tree_schedule(gb), psb, NamedTuple())
        @test first(LenticulumCore.scalar_free_energy(sb, gb, psb, stb)) ≈ nlp atol = 1e-10
        # coordinate updates never increase it (each pair of tasks is one exact update of a q)
        s = MessageStore(g)
        sweep!(s, g, SequentialSchedule(init), ps, NamedTuple())
        sweep!(s, g, SequentialSchedule(sweep), ps, NamedTuple())
        Fs = Float64[]
        for _ in 1:6, pair in (sweep[1:2], sweep[3:4])
            sweep!(s, g, SequentialSchedule(pair), ps, NamedTuple())
            push!(Fs, first(LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple())))
        end
        @test all(diff(Fs) .<= 1e-12)
        @test Fs[1] - Fs[end] > 1e-6
        # the schedule built from the rules reaches the same fixed point
        s_h = MessageStore(g)
        rh, _ = propagate!(s_h, g, hybrid_schedule(g), ps, NamedTuple(); maxsweeps = 20_000, tol = 1e-14)
        @test rh.converged
        @test first(LenticulumCore.scalar_free_energy(s_h, g, ps, NamedTuple())) ≈ F_mf atol = 1e-10
        # the graded parts: the mean-field node carries no entropy
        Fg, _ = factor_free_energies(s, g, ps, NamedTuple())
        @test Fg[:lc][:negentropy] == 0
    end

    @testset "the gradient at a fixed point is the derivative of the converged free energy" begin
        g, ps, init, sweep = sum_graph(MeanField())
        reference = mf_fd_gradient(g, ps, init, sweep)
        for ad in (nothing, AutoForwardDiff(), AutoZygote())
            @test mf_graph_gradient(g, ps, init, sweep; ad) ≈ reference atol = 1e-6
        end
    end

    @testset "average energies against Monte Carlo" begin
        rng = Xoshiro(11)
        Q = [0.4 0.1; 0.1 0.3]
        f = LinearConstraintFactor((a = 2, b = 1, z = 2), 2; noise = Q)
        ps = (A = (a = randn(rng, 2, 2), b = randn(rng, 2, 1), z = randn(rng, 2, 2)), c = randn(rng, 2))
        Sa = [0.5 0.2; 0.2 0.7]
        marg = (a = Gaussian([0.3, -0.6], Sa), b = Gaussian([1.1], fill(0.4, 1, 1)), z = DiracBelief([0.2, 0.9]))
        F, _ = Mycelium.mean_field_energy(f, marg, ps, NamedTuple())
        @test F[:negentropy] == 0
        La, Lb = cholesky(Sa).L, sqrt(0.4)
        n = 200_000
        e = map(1:n) do _
            a = [0.3, -0.6] + La * randn(rng, 2); bb = [1.1 + Lb * randn(rng)]
            -mf_lognormal(ps.A.a * a + ps.A.b * bb + ps.A.z * [0.2, 0.9] - ps.c, zeros(2), Q)
        end
        @test abs(F[:fit] + F[:complexity] - sum(e) / n) < 5 * std_(e) / sqrt(n)

        K = 3
        sf = SwitchingConstraintFactor((a = 2,), K, 2; noise = Q)
        sps = (A = (a = randn(rng, 2, 2),), b = randn(rng, 2, K), logw = [0.2, -0.5, 0.1])
        qk = [0.5, 0.2, 0.3]
        smarg = (a = Gaussian([0.3, -0.6], Sa), k = CategoricalBelief(qk))
        Fs, _ = Mycelium.mean_field_energy(sf, smarg, sps, NamedTuple())
        @test Fs[:negentropy] == 0
        lw = sps.logw .- mf_logsumexp(sps.logw)
        e = map(1:n) do _
            a = [0.3, -0.6] + La * randn(rng, 2); k = findfirst(cumsum(qk) .>= rand(rng))
            -lw[k] - mf_lognormal(sps.A.a * a - sps.b[:, k], zeros(2), Q)
        end
        @test abs(Fs[:fit] + Fs[:complexity] - sum(e) / n) < 5 * std_(e) / sqrt(n)
        # a marginal that is not yet a distribution is refused
        @test_throws ArgumentError Mycelium.mean_field_energy(f, merge(marg, (a = LenticulumCore.TrivialBelief(),)), ps, NamedTuple())
        @test_throws ArgumentError Mycelium.mean_field_energy(sf, (a = smarg.a, k = LenticulumCore.TrivialBelief()), sps, NamedTuple())
    end

    @testset "a mean-field sum of switches: a bound on -log p, and its gradient" begin
        # z = c + u₁ + u₂ + ε, each u_l a binary switch with offsets {0, δ_l} and width η; the
        # sum node on mean field, the switches on belief propagation. Each switch's part is a
        # tree, so the free energy is KL(q ‖ p) - log p(x) for q = q(u₁, k₁) q(u₂, k₂).
        d, η = 2, 1e-3
        Σ = [0.3 0.05; 0.05 0.25]; x = [1.1, 0.7]
        I2 = Matrix(1.0I, d, d)
        b = GraphBuilder()
        variable!(b, :z, d); variable!(b, :u1, d); variable!(b, :u2, d); variable!(b, :k1, 2); variable!(b, :k2, 2)
        factor!(b, :data, DataFactor(:z, x))
        factor!(b, :sum, LinearConstraintFactor((z = d, u1 = d, u2 = d), d; noise = Σ); rule = MeanField())
        connect!(b, :data, :z, :z; direction = Emitting()); connect!(b, :sum, :z, :z; direction = Absorbing())
        for l in 1:2
            factor!(b, Symbol(:sw, l), SwitchingConstraintFactor((u = d,), 2, d; noise = η))
            connect!(b, Symbol(:sw, l), :u, Symbol(:u, l)); connect!(b, :sum, Symbol(:u, l), Symbol(:u, l))
            connect!(b, Symbol(:sw, l), :k, Symbol(:k, l))
        end
        g = build(b)
        c0, δ1, δ2, lw1, lw2 = [0.1, -0.2], [2.0, 0.0], [1.6, 1.2], log.([0.6, 0.4]), log.([0.5, 0.5])
        ps = merge(LuxCore.initialparameters(Xoshiro(0), g),
            (sum = (A = (z = I2, u1 = -I2, u2 = -I2), c = c0),
             sw1 = (A = (u = I2,), b = [zeros(d) δ1], logw = lw1), sw2 = (A = (u = I2,), b = [zeros(d) δ2], logw = lw2)))
        T = MessageTask
        init = [T(mf_edge(g, :data, :z), :to_variable), T(mf_edge(g, :sw1, :u), :to_variable), T(mf_edge(g, :sw2, :u), :to_variable)]
        sweep = [T(mf_edge(g, :sum, :u1), :to_variable), T(mf_edge(g, :sw1, :u), :to_factor),
                 T(mf_edge(g, :sw1, :k), :to_variable), T(mf_edge(g, :sw1, :u), :to_variable),
                 T(mf_edge(g, :sum, :u2), :to_variable), T(mf_edge(g, :sw2, :u), :to_factor),
                 T(mf_edge(g, :sw2, :k), :to_variable), T(mf_edge(g, :sw2, :u), :to_variable)]
        # narrow mixture components leave a rounding floor in the message residuals (switching.md)
        tol = 1e-10
        s, report = mf_run(g, ps, init, sweep; tol)
        @test report.converged
        F = first(LenticulumCore.scalar_free_energy(s, g, ps, NamedTuple()))
        lj = [lw1[a] - mf_logsumexp(lw1) + lw2[bb] - mf_logsumexp(lw2) +
              mf_lognormal(x, c0 + (a - 1) * δ1 + (bb - 1) * δ2, Σ + 2η * I2) for a in 1:2 for bb in 1:2]
        nlp = -mf_logsumexp(lj)
        @test F > nlp
        # the gap is the KL from the factorised posterior over (k₁, k₂) to the exact one, up to
        # the width η: compare with the factorised posterior the graph converged to
        q1 = probabilities(marginal(s, g, g.variable_index[:k1])); q2 = probabilities(marginal(s, g, g.variable_index[:k2]))
        P = exp.(lj .- mf_logsumexp(lj))
        kl = sum(q1[a] * q2[bb] * log(q1[a] * q2[bb] / P[2(a - 1) + bb]) for a in 1:2 for bb in 1:2)
        @test F - nlp ≈ kl atol = 1e-2
        reference = mf_fd_gradient(g, ps, init, sweep; tol)
        for ad in (nothing, AutoForwardDiff(), AutoZygote())
            @test mf_graph_gradient(g, ps, init, sweep; ad, tol) ≈ reference atol = 1e-5
        end
        # paths that differentiate through inference, or record residual moments, still refuse
        @test_throws ArgumentError LenticulumCore.free_energy_gradient(s, g, ps, NamedTuple(); adjoint = :implicit)
    end
end

