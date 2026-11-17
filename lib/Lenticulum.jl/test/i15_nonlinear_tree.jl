# I15: a BP NonlinearFactor reads the target cavity for its expansion, so one collect/distribute
# pass need not be a fixed point. The tree shortcut of `propagate!` is therefore restricted to
# messages that do not read the cavity (`message_uses_cavity`), and such a tree is iterated with
# its residual checked. Oracles: a second sweep from a fresh store moves the marginal (both root
# choices), and a tree whose messages do not read the cavity keeps the one-sweep exact report.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random

@testset "I15: a nonlinear tree is iterated, not declared a one-sweep fixed point" begin
    b = GraphBuilder(); variable!(b, :x, 1); variable!(b, :y, 1)
    factor!(b, :prior, GaussianPrior(:x, 2., 1.)); connect!(b, :prior, :x, :x)
    factor!(b, :nl, NonlinearFactor(x -> x .^ 2, 1 => 1; noise = 0.1))
    connect!(b, :nl, :x, :x); connect!(b, :nl, :y, :y)
    factor!(b, :data, DataFactor(:y, [9.])); connect!(b, :data, :y, :y; direction = Emitting())
    g = build(b); ps = LuxCore.initialparameters(Xoshiro(1), g)
    @test Mycelium.message_uses_cavity(only(fn.factor for fn in g.factors if fn.factor isa NonlinearFactor))
    for root in (:prior, :data)
        sched = tree_schedule(g; root = root)
        _, rep, st, _ = infer!(g, sched, ps, (;))
        @test !rep.exact
        @test !(rep.converged && rep.sweeps == 1)      # one pass is not claimed a fixed point
        # and with root :prior it is not one: another sweep from a fresh store moves the
        # marginal (with root :data the observation is incorporated in the first sweep and
        # the second changes nothing)
        s2 = MessageStore(g)
        _, st = sweep!(s2, g, sched, ps, st)
        m1 = belief_mean(marginal(s2, g, g.variable_index[:x]))
        _, st = sweep!(s2, g, sched, ps, st)
        m2 = belief_mean(marginal(s2, g, g.variable_index[:x]))
        if root === :prior
            @test norm(m2 .- m1) > 1e-8
        else
            @test norm(m1 .- [2.0]) > 1e-8
        end
    end
    # a tree whose messages do not read the cavity keeps the one-sweep exact report
    gb = GraphBuilder(); variable!(gb, :x, 1); variable!(gb, :y, 1)
    factor!(gb, :prior, GaussianPrior(:x, 2., 1.)); connect!(gb, :prior, :x, :x)
    factor!(gb, :lin, GaussianFactor(1 => 1; noise = 0.5)); connect!(gb, :lin, :x, :x); connect!(gb, :lin, :y, :y)
    factor!(gb, :data, DataFactor(:y, [9.])); connect!(gb, :data, :y, :y; direction = Emitting())
    gl = build(gb); psl = LuxCore.initialparameters(Xoshiro(1), gl)
    _, rep, _, _ = infer!(gl, tree_schedule(gl), psl, (;))
    @test rep.converged && rep.sweeps == 1 && rep.exact
end
