# Structure edits on a built graph (Extending Mycelium §6, L-SPEC-010). Oracles: the same graph
# built directly, and its exact free energy (every factor in one cluster node).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using MyceliumLearning

const SE_D = 2
se_eye() = Matrix(1.0I, SE_D, SE_D)
se_sum(L) = LinearConstraintFactor(NamedTuple{(:z, (Symbol(:u, l) for l in 1:L)...)}(ntuple(_ -> SE_D, L + 1)), SE_D; noise = 0.2)
se_sum_ps(L) = (A = NamedTuple{(:z, (Symbol(:u, l) for l in 1:L)...)}((se_eye(), ntuple(_ -> -se_eye(), L)...)), c = zeros(SE_D))
se_sw() = SwitchingConstraintFactor((u = SE_D,), 2, SE_D; noise = 0.01)

# the additive wiring with L switches, built directly
function se_direct(L, x)
    b = GraphBuilder(); variable!(b, :z, SE_D)
    factor!(b, :data, DataFactor(:z, x)); connect!(b, :data, :z, :z; direction = Emitting())
    factor!(b, :sum, se_sum(L)); connect!(b, :sum, :z, :z)
    for l in 1:L
        variable!(b, Symbol(:u, l), SE_D); variable!(b, Symbol(:k, l), 2)
        factor!(b, Symbol(:sw, l), se_sw())
        connect!(b, :sum, Symbol(:u, l), Symbol(:u, l)); connect!(b, Symbol(:sw, l), :u, Symbol(:u, l))
        connect!(b, Symbol(:sw, l), :k, Symbol(:k, l))
    end
    return build(b)
end
# the move "add a switch", as an edit: the sum node gains a channel, a switch and its variables
function se_add_switch(g, L)
    b = GraphBuilder(g)
    replace_factor!(b, :sum, se_sum(L + 1))
    u, k, sw = Symbol(:u, L + 1), Symbol(:k, L + 1), Symbol(:sw, L + 1)
    variable!(b, u, SE_D); variable!(b, k, 2); factor!(b, sw, se_sw())
    connect!(b, :sum, u, u); connect!(b, sw, :u, u); connect!(b, sw, :k, k)
    return build(b)
end
# the move "remove switch L"
function se_remove_switch(g, L)
    b = GraphBuilder(g)
    remove_factor!(b, Symbol(:sw, L)); remove_variable!(b, Symbol(:u, L)); remove_variable!(b, Symbol(:k, L))
    replace_factor!(b, :sum, se_sum(L - 1))
    return build(b)
end
const SE_W = [2.0 0.0 1.0; 0.0 2.0 -1.0]
se_ps(L) = merge((data = NamedTuple(), sum = se_sum_ps(L)),
    NamedTuple{Tuple(Symbol(:sw, l) for l in 1:L)}(Tuple((A = (u = se_eye(),), b = hcat(zeros(SE_D), SE_W[:, l]), logw = [0.3, -0.3]) for l in 1:L)))
function se_exact_F(g, L)
    gc = cluster(g, [:all => [:sum; [Symbol(:sw, l) for l in 1:L]]])
    psc = cluster_parameters(gc, se_ps(L))
    _, _, _, s = infer!(gc, tree_schedule(gc), psc, NamedTuple())
    return first(scalar_free_energy(s, gc, psc, NamedTuple()))
end
se_signature(g) = (Set(v.name for v in g.variables), Set(f.name for f in g.factors),
    Set((g.factors[e.factor].name, e.channel, g.variables[e.variable].name) for e in g.edges))

@testset "Edits reproduce the graphs built directly" begin
    x = [1.7, 2.4]
    g1 = se_direct(1, x)
    g2 = se_add_switch(g1, 1)
    @test se_signature(g2) == se_signature(se_direct(2, x))
    @test se_exact_F(g2, 2) ≈ se_exact_F(se_direct(2, x), 2) rtol = 1e-14
    g3 = se_add_switch(g2, 2)
    @test se_signature(g3) == se_signature(se_direct(3, x))
    g2b = se_remove_switch(g3, 3)
    @test se_signature(g2b) == se_signature(se_direct(2, x))
    @test se_exact_F(g2b, 2) ≈ se_exact_F(se_direct(2, x), 2) rtol = 1e-14
    # disconnect!, and an edited graph is consistent
    b = GraphBuilder(g2); disconnect!(b, :sw2, :k); gd = build(b)
    @test !any(e -> gd.factors[e.factor].name === :sw2 && e.channel === :k, gd.edges)
    @test all(e -> gd.factors[e.factor].id == e.factor, gd.edges)
    @test_throws ArgumentError remove_factor!(GraphBuilder(g2), :nope)
    @test_throws ArgumentError disconnect!(GraphBuilder(g2), :sw2, :nope)
    bp = GraphBuilder(); plate!((p, i) -> (variable!(p, :x, 1); factor!(p, :pr, GaussianPrior(:x, 0.0, 1.0)); connect!(p, :pr, :x, :x)), bp, :pl, 2)
    @test_throws ArgumentError GraphBuilder(build(bp))
end

@testset "transfer: a warm start after an edit" begin
    x = [1.7, 2.4]
    g2 = se_direct(2, x)
    sched2 = SequentialSchedule(all_tasks(g2))
    s2 = MessageStore(g2)
    rep2, _ = propagate!(s2, g2, sched2, se_ps(2), NamedTuple(); maxsweeps = 500, tol = 1e-10)
    @test rep2.converged
    g3 = se_add_switch(g2, 2)
    sched3 = SequentialSchedule(all_tasks(g3))
    t = transfer(s2, g2, g3)
    kept = count(i -> t.to_variable[i] !== nothing, 1:nedges(g3))
    @test kept == count(i -> s2.to_variable[i] !== nothing, 1:nedges(g2))   # every old edge survives the move
    cold = MessageStore(g3)
    rc, _ = propagate!(cold, g3, sched3, se_ps(3), NamedTuple(); maxsweeps = 500, tol = 1e-10)
    rw, _ = propagate!(t, g3, sched3, se_ps(3), NamedTuple(); maxsweeps = 500, tol = 1e-10)
    @test rc.converged && rw.converged
    @test rw.sweeps <= rc.sweeps
    for v in (:u1, :u2, :u3)
        qc, qw = moment_match(marginal(cold, g3, g3.variable_index[v])), moment_match(marginal(t, g3, g3.variable_index[v]))
        @test belief_mean(qc) ≈ belief_mean(qw) atol = 1e-8
    end
    # pins survive by name
    pin!(s2, g2, :k1, 2)
    @test !isempty(transfer(s2, g2, g3).pins)
    # a variable re-added with another dimension keeps none of its messages
    b = GraphBuilder(g2); remove_factor!(b, :sw2); remove_variable!(b, :k2); variable!(b, :k2, 3)
    factor!(b, :sw2, SwitchingConstraintFactor((u = SE_D,), 3, SE_D; noise = 0.01)); connect!(b, :sw2, :u, :u2); connect!(b, :sw2, :k, :k2)
    g4 = build(b); t4 = transfer(s2, g2, g4)
    @test all(i -> t4.to_variable[i] === nothing, [i for (i, e) in enumerate(g4.edges) if g4.variables[e.variable].name === :k2])
end
