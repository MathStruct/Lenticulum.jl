using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using LenticulumCore: CategoricalBelief, DiracBelief, ComposedFactor
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

# LP-020: the graph parameter gradient used to hold the incoming messages fixed, not the factor
# beliefs, so it was not the gradient of the free energy, and alternating inference with it did not
# converge to maximum likelihood. It now holds the beliefs fixed (`factor_belief` and
# `belief_free_energy`): the envelope gradient, exact on a tree at the belief-propagation fixed point.

# The converged Bethe free energy as a function of the parameters, inference re-run at each point:
# on a tree its derivative is the derivative of −log p, the reference for the graph gradient.
function converged_free_energy(g, θ, ps, st)
    p = LenticulumCore._unflatten(θ, ps)
    _, _, st2, s = infer!(g, tree_schedule(g), p, st)
    return first(LenticulumCore.scalar_free_energy(s, g, p, st2))
end
function converged_gradient(g, ps, st; h = 1e-6)
    θ = LenticulumCore._flatten(ps)
    return [(converged_free_energy(g, θ .+ h .* (1:length(θ) .== i), ps, st) -
             converged_free_energy(g, θ .- h .* (1:length(θ) .== i), ps, st)) / 2h for i in eachindex(θ)]
end
function graph_gradient(g, ps, st; ad = nothing)
    _, _, st2, s = infer!(g, tree_schedule(g), ps, st)
    return LenticulumCore._flatten(LenticulumCore.free_energy_gradient(s, g, ps, st2; ad))
end
# At the belief it was evaluated at, the free energy of a fixed belief is the local free energy.
function check_frozen_identity(g, ps, st)
    _, _, st2, s = infer!(g, tree_schedule(g), ps, st)
    n = 0
    for f in g.factors
        msgs, p = messages_into(s, g, f.id), ps[f.name]
        b = factor_belief(f.factor, msgs, p, st2[f.name])
        b === nothing && continue
        F = first(local_free_energy(f.factor, msgs, p, st2[f.name]))
        Fb = first(belief_free_energy(f.factor, b, p, st2[f.name]))
        @test collect(values(Fb.parts)) ≈ collect(values(F.parts)) atol = 1e-12
        n += 1
    end
    return n
end
function check_gradient(g, ps, st; atol = 1e-6)
    reference = converged_gradient(g, ps, st)
    for ad in (nothing, AutoForwardDiff(), AutoZygote())
        @test graph_gradient(g, ps, st; ad) ≈ reference atol = atol
    end
end

@testset "LP-020: the graph parameter gradient holds the factor beliefs fixed" begin
    @testset "linear constraint: the smallest example, and training to maximum likelihood" begin
        # u ~ N(0, 1), z = a u + c + ε, ε ~ N(0, q), z observed.
        q, z = 0.3, [1.7]
        b = GraphBuilder()
        variable!(b, :u, 1); variable!(b, :z, 1)
        factor!(b, :pu, GaussianPrior(:u, 0.0, 1.0))
        factor!(b, :lc, LinearConstraintFactor((z = 1, u = 1), 1; noise = q))
        factor!(b, :data, DataFactor(:z, z))
        connect!(b, :pu, :u, :u); connect!(b, :lc, :u, :u); connect!(b, :lc, :z, :z; direction = Absorbing())
        connect!(b, :data, :z, :z; direction = Emitting())
        g = build(b)
        ps0 = LuxCore.initialparameters(Xoshiro(0), g)
        ps(a, c) = merge(ps0, (lc = (A = (z = fill(1.0, 1, 1), u = fill(-a, 1, 1)), c = [c]),))
        nlp(a, c) = ((z[1] - c)^2 / (a^2 + q) + log(2π * (a^2 + q))) / 2           # −log p(z)
        h = 1e-6
        exact(a, c) = (A_u = -(nlp(a + h, c) - nlp(a - h, c)) / 2h, c = (nlp(a, c + h) - nlp(a, c - h)) / 2h)
        graph(a, c) = (p = ps(a, c); (_, _, st, s) = infer!(g, tree_schedule(g), p, NamedTuple());
                       gr = LenticulumCore.free_energy_gradient(s, g, p, st; ad = nothing); (A_u = gr.lc.A.u[1], c = gr.lc.c[1]))

        a, c = 0.8, 0.4
        _, _, st, s = infer!(g, tree_schedule(g), ps(a, c), NamedTuple())
        @test first(LenticulumCore.scalar_free_energy(s, g, ps(a, c), st)) ≈ nlp(a, c)   # Bethe is exact here
        # Holding the messages fixed gave (−0.146, −0.441); the gradient is (0.679, −1.383).
        @test graph(a, c).A_u ≈ exact(a, c).A_u rtol = 1e-6
        @test graph(a, c).c ≈ exact(a, c).c rtol = 1e-6
        @test exact(a, c).A_u ≈ 0.679 atol = 1e-3
        # differentiating through inference (sample_gradient, nothing sampled) agrees
        sampler = GibbsSampler(g, ())
        buf = SampleBuffer(WindowFilter(1))
        record!(buf, sampler, NamedTuple(), ps(a, c), NamedTuple())
        gi = sample_gradient(buf, sampler, ps(a, c), NamedTuple())
        @test gi.lc.A.u[1] ≈ exact(a, c).A_u rtol = 1e-6
        @test gi.lc.c[1] ≈ exact(a, c).c rtol = 1e-6
        # The gradient vanishes at the maximum-likelihood a, and alternating inference with it gets
        # there. Holding the messages fixed, it stopped at a = 0.754 instead of 1.179.
        aml = sqrt((z[1] - c)^2 - q)
        @test abs(graph(aml, c).A_u) < 1e-6
        at = 0.8
        for _ in 1:2000
            at += 0.05 * graph(at, c).A_u                                           # gradient descent on A_u = −a
        end
        @test at ≈ aml atol = 1e-6
        @test check_frozen_identity(g, ps(a, c), NamedTuple()) == 2
        check_gradient(g, ps(a, c), NamedTuple())
    end

    @testset "Gaussian factor and prior with learnable noise" begin
        b = GraphBuilder()
        variable!(b, :u, 1); variable!(b, :y, 2)
        factor!(b, :pu, GaussianPrior(:u, [0.3], fill(0.8, 1, 1); learn_noise = true))
        factor!(b, :model, GaussianFactor(1 => 2; noise = 0.5, learn_noise = true, channels = (:u, :y)))
        factor!(b, :data, DataFactor(:y, [0.9, -0.4]))
        connect!(b, :pu, :u, :u); connect!(b, :model, :u, :u)
        connect!(b, :model, :y, :y; direction = Absorbing()); connect!(b, :data, :y, :y; direction = Emitting())
        g = build(b)
        ps, st = LuxCore.setup(Xoshiro(3), g)
        ps = merge(ps, (model = merge(ps.model, (A = reshape([0.7, -1.2], 2, 1), b = [0.1, 0.2],
                                                noise = (log_diagonal = [0.2, -0.3], lower = [0.15]))),))
        @test check_frozen_identity(g, ps, st) == 2
        check_gradient(g, ps, st)
    end

    @testset "collapsed Gaussian composite" begin
        a = GaussianFactor(1 => 1; noise = 0.3, channels = (:x, :u))
        c = GaussianFactor(1 => 1; noise = 0.2, channels = (:u, :y))
        b = GraphBuilder()
        variable!(b, :x, 1); variable!(b, :y, 1)
        factor!(b, :px, GaussianPrior(:x, [0.2], fill(1.5, 1, 1)))
        factor!(b, :comp, ComposedFactor(a, c))
        factor!(b, :data, DataFactor(:y, [1.2]))
        connect!(b, :px, :x, :x); connect!(b, :comp, :x, :x)
        connect!(b, :comp, :y, :y; direction = Absorbing()); connect!(b, :data, :y, :y; direction = Emitting())
        g = build(b)
        ps, st = LuxCore.setup(Xoshiro(4), g)
        ps = merge(ps, (comp = (first = (A = fill(0.8, 1, 1), b = [0.1]), second = (A = fill(1.1, 1, 1), b = [-0.2])),))
        @test check_frozen_identity(g, ps, st) == 2
        check_gradient(g, ps, st)
    end

    @testset "switching constraint: A, b and the log-weights" begin
        # k ~ categorical, A_x x + y = b_k + ε with y observed and x latent. x has no other factor:
        # a Gaussian neighbour would receive the factor's mixture message, whose free energy no
        # Gaussian factor has in closed form. Its belief is proper all the same, as A_x is invertible.
        K = 3
        b = GraphBuilder()
        variable!(b, :x, 2); variable!(b, :y, 2); variable!(b, :k, K)
        factor!(b, :sw, SwitchingConstraintFactor((x = 2, y = 2), K, 2; noise = [0.3 0.1; 0.1 0.2]))
        prior = [0.5, 0.25, 0.25]
        factor!(b, :pk, PriorFactor(:k, CategoricalBelief(prior), k -> -log(prior[k])))
        factor!(b, :data, DataFactor(:y, [0.4, 0.9]))
        connect!(b, :sw, :x, :x)
        connect!(b, :sw, :y, :y; direction = Absorbing()); connect!(b, :data, :y, :y; direction = Emitting())
        connect!(b, :sw, :k, :k); connect!(b, :pk, :k, :k)
        g = build(b)
        ps, st = LuxCore.setup(Xoshiro(0), g)
        ps = merge(ps, (sw = (A = (x = [-1.0 0.2; 0.0 -0.9], y = Matrix(1.0I, 2, 2)),
                              b = [-1.0 1.0 0.0; 0.0 0.5 2.0], logw = log.([0.2, 0.3, 0.5])),))
        @test check_frozen_identity(g, ps, st) == 1
        check_gradient(g, ps, st)
        # the log-weights' gradient is −(q − w), the EM direction, with q the factor's outcome posterior
        _, _, st2, s = infer!(g, tree_schedule(g), ps, st)
        fid = findfirst(f -> f.name === :sw, g.factors)
        q = first(switch_statistics(g.factors[fid].factor, messages_into(s, g, fid), ps.sw))
        w = exp.(ps.sw.logw) ./ sum(exp.(ps.sw.logw))
        @test LenticulumCore.free_energy_gradient(s, g, ps, st2).sw.logw ≈ -(q .- w) atol = 1e-8
    end
end
