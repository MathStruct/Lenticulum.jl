# I17 tests written from the specification (Mistral, 2026-11-21), against the code before the change and
# without its implementation: reverse-mode AD (Zygote) through the three replay paths of MyceliumLearning,
# values and derivatives pinned (`lib/MyceliumLearning.jl/src/gradient_couplings.md`, "Reverse mode by
# message versions"). The session fixed four mistakes of the test (marked "session fix") and trimmed
# I17-1's instances (4 min 50 s for three seeds); every testset but I17-9 passed on the code before.
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, Test, LinearAlgebra, Random, LuxCore,
      ForwardDiff, Zygote, ADTypes, DifferentiationInterface
using Krylov

fl(p) = LenticulumCore._flatten(p)

# Values on the code before I17 (dev-danielb 3ff93526, 2026-11-21), printed by this file (session).
const SCORE_VALUE_LITERAL = [0.6804401698805057]
const SCORE_SE_LITERAL = [0.21288705815329476]
const UNROLLED_LITERAL = [-0.4808106209805771, -0.6726710324815285, 2.1749365441831245, -4.65569780832623, 0.22401252842559816, 0.1699317630441762, 0.3218643260286864, -0.30638323932869027]
const IMPLICIT_LITERAL = [0.07361556531587538, 2.0718965587215195, 0.20825139328034833, -0.24077800860612658, -0.09702927707103956, 0.25486108642345723]
const INFER_LITERAL = [-4.509875138402768, 1.2610826588850397, 0.0, 0.0, 0.20275513146659055, -0.1616610729228418, 0.9771282512149242, -0.7790855912352264]

# Raw `Zygote.gradient` gives `nothing` for a key or field with no gradient; fill it with zeros shaped
# like `ps` before flattening (session fix).
_zeros(p::NamedTuple) = map(_zeros, p)
_zeros(p) = zero(p)
zfill(gr, p::NamedTuple) = gr === nothing ? _zeros(p) : NamedTuple{keys(p)}(map(k -> zfill(get(gr, k, nothing), p[k]), keys(p)))
zfill(gr, p) = gr === nothing ? zero(p) : gr

include(joinpath(@__DIR__, "..", "..", "..", "bench", "kalman_chain.jl"))

# ---------------------------------------------------------------------------
# Graph families. Factor NAMES are fixed across instances so that the `ps` type is
# constant: Zygote compiles the replay once per family, not once per instance.
# Trees: n scalar variables, a prior on every variable, one pair factor per
# non-root variable, data factors on `data_on`.
function spec_tree(rng; n = 5, coupling = ExactCoupling(), data_on = (2, 3),
                   detach_child = 0, diagonal_children = (), force_parent = ())
    b = GraphBuilder()
    vs = [Symbol(:x, i) for i in 1:n]
    for (i, v) in enumerate(vs)
        variable!(b, v, 1)
        p = Symbol(:p, i)
        factor!(b, p, GaussianPrior(:x, [0.1 * randn(rng)], fill(1.0, 1, 1)))
        connect!(b, p, :x, v; coupling)
    end
    for i in 2:n
        par = isempty(force_parent) ? rand(rng, 1:(i - 1)) :
              get(force_parent, i, rand(rng, 1:(i - 1)))
        f = Symbol(:f, i)
        factor!(b, f, GaussianFactor(1 => 1; noise = 0.3, channels = (:a, :b)))
        c = i in diagonal_children ? DiagonalCoupling() : coupling
        connect!(b, f, :a, vs[par]; coupling = c)
        connect!(b, f, :b, vs[i]; coupling = c, detach = i == detach_child)
    end
    for i in data_on
        d = Symbol(:d, i)
        factor!(b, d, DataFactor(:v, [randn(rng)]))
        connect!(b, d, :v, vs[i]; direction = Emitting(), coupling)
    end
    g = build(b)
    ps, st = LuxCore.setup(rng, g)
    kv = Dict{Symbol,Any}(pairs(ps))
    for fnode in g.factors
        fnode.factor isa GaussianFactor || continue
        k = fnode.params
        kv[k] = (A = [randn(rng) / 1.5 + 0.3;;], b = [0.1 * randn(rng)])
    end
    ks = keys(ps)
    return g, NamedTuple{ks}(Tuple(kv[k] for k in ks)), st
end

# Loops: a cycle of k scalar variables, a prior on every variable, one pair factor
# per cycle edge, one data factor.
function spec_loop(rng; k = 3, coupling = ExactCoupling())
    b = GraphBuilder()
    vs = [Symbol(:x, i) for i in 1:k]
    for (i, v) in enumerate(vs)
        variable!(b, v, 1)
        p = Symbol(:p, i)
        factor!(b, p, GaussianPrior(:x, [0.1 * randn(rng)], fill(1.0, 1, 1)))
        connect!(b, p, :x, v; coupling)
    end
    for i in 1:k
        j = mod1(i + 1, k)
        f = Symbol(:f, i)
        factor!(b, f, GaussianFactor(1 => 1; noise = 0.4, channels = (:a, :b)))
        connect!(b, f, :a, vs[i]; coupling)
        connect!(b, f, :b, vs[j]; coupling)
    end
    factor!(b, :d, DataFactor(:v, [randn(rng)]))
    connect!(b, :d, :v, vs[2]; direction = Emitting(), coupling)
    g = build(b)
    ps, st = LuxCore.setup(rng, g)
    kv = Dict{Symbol,Any}(pairs(ps))
    for fnode in g.factors
        fnode.factor isa GaussianFactor || continue
        k2 = fnode.params
        kv[k2] = (A = [randn(rng) / 1.5;;], b = [0.1 * randn(rng)])
    end
    ks = keys(ps)
    return g, NamedTuple{ks}(Tuple(kv[k] for k in ks)), st
end

# The score-coupling shape of lib/MyceliumLearning.jl/test/gradient_couplings.jl: a
# learnable terminal-free source emitting a sampleable Gaussian, a LossFactor absorbing
# on the same variable with ScoreFunctionCoupling.
struct SpecSource <: AbstractLenticulumFactor end
LenticulumCore.channels(::SpecSource) = (LenticulumCore.Channel(:x, nothing),)
LenticulumCore.supported_polarities(::SpecSource) = (Polarity(x = Unobserved()),)
LenticulumCore.supports_polarity(::SpecSource, p::Polarity) = p == Polarity(x = Unobserved())
LenticulumCore.islearnable(::SpecSource) = true
Mycelium.factor_message(::SpecSource, target::Symbol, pol, inputs, prior, ps, st) =
    (GaussianBelief([ps.μ], fill(1.0, 1, 1)), st)
Mycelium.local_free_energy(::SpecSource, beliefs, ps, st) = (0.0, st)

function spec_score_graph()
    b = GraphBuilder(); variable!(b, :x)
    factor!(b, :source, SpecSource())
    factor!(b, :loss, LossFactor((:x,), x -> (only(x) + 0.2)^2 / 2))
    connect!(b, :source, :x, :x; direction = Emitting(), coupling = ExactCoupling())
    connect!(b, :loss, :x, :x; direction = Absorbing(), coupling = ScoreFunctionCoupling())
    g = build(b)
    return g, (source = (μ = 0.8,), loss = (;)), (source = (;), loss = (;)), forward_backward_schedule(g)   # score sampling refuses flooding (session fix)
end

# Gaussian marginals only: an observed variable's marginal is a point (session fix)
marginal_loss(margs) = sum(b -> b isa GaussianBelief ? sum(abs2, LenticulumCore.belief_mean(b)) : 0.0, values(margs))

# ---------------------------------------------------------------------------
@testset "I17 coupled spec" begin
@testset "I17-1 unrolled replay: Zygote = ForwardDiff = reference differences" begin
    # random trees, every edge ExactCoupling
    for seed in (11,)   # (11, 12, 13) in the run before the change
        g, ps, st = spec_tree(Xoshiro(seed))
        scheds = Any[SequentialSchedule(tasks(tree_schedule(g))),
                     flooding_schedule(g),
                     ResidualSchedule(maxtasks = length(all_tasks(g)))]
        for sched in scheds, nsweeps in (sched isa SequentialSchedule ? (1, 2, 3) : (2, 3)),
                damping in (0.0, 0.3)
            gz = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps, damping)
            gf = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoForwardDiff(), schedule = sched, nsweeps, damping)
            gr = free_energy_gradient(MessageStore(g), g, ps, st; ad = nothing, schedule = sched, nsweeps, damping)
            @test fl(gz) ≈ fl(gf) rtol = 1e-8
            @test fl(gr) ≈ fl(gf) rtol = 1e-4
        end
    end
    # loopy graphs: cycles of 3 and 5 variables
    for (k, seed) in ((3, 21), (5, 22))   # k in (3, 5) × seed in (21, 22) before
        g, ps, st = spec_loop(Xoshiro(seed); k)
        for sched in (flooding_schedule(g), ResidualSchedule(maxtasks = length(all_tasks(g)))),
                nsweeps in (2, 3), damping in (0.0, 0.3)
            gz = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps, damping)
            gf = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoForwardDiff(), schedule = sched, nsweeps, damping)
            gr = free_energy_gradient(MessageStore(g), g, ps, st; ad = nothing, schedule = sched, nsweeps, damping)
            @test fl(gz) ≈ fl(gf) rtol = 1e-8
            @test fl(gr) ≈ fl(gf) rtol = 1e-4
        end
    end
end

@testset "I17-2 tied keys: the shared gradient is the sum of the untied ones" begin
    function chain_graph(rng; tied)
        θrng = copy(rng)   # θ not from `rng`: `setup` draws differently with one key fewer (session fix)
        b = GraphBuilder()
        for v in (:x1, :x2, :x3); variable!(b, v, 1); end
        factor!(b, :prior, GaussianPrior(:x, [0.0], fill(1.0, 1, 1)))
        connect!(b, :prior, :x, :x1; coupling = ExactCoupling())
        factor!(b, :f1, GaussianFactor(1 => 1; noise = 0.3, channels = (:a, :b)); params = tied ? :shared : :f1)
        factor!(b, :f2, GaussianFactor(1 => 1; noise = 0.3, channels = (:a, :b)); params = tied ? :shared : :f2)
        connect!(b, :f1, :a, :x1; coupling = ExactCoupling())
        connect!(b, :f1, :b, :x2; coupling = ExactCoupling())
        connect!(b, :f2, :a, :x2; coupling = ExactCoupling())
        connect!(b, :f2, :b, :x3; coupling = ExactCoupling())
        factor!(b, :d, DataFactor(:v, [randn(rng)]))
        connect!(b, :d, :v, :x3; direction = Emitting(), coupling = ExactCoupling())
        g = build(b)
        ps, st = LuxCore.setup(rng, g)
        kv = Dict{Symbol,Any}(pairs(ps))
        θ = (A = [randn(θrng) / 1.5 + 0.3;;], b = [0.1 * randn(θrng)])   # one value for both (session fix)
        for fnode in g.factors
            fnode.factor isa GaussianFactor || continue
            kv[fnode.params] = θ
        end
        ks = keys(ps)
        return g, NamedTuple{ks}(Tuple(kv[k] for k in ks)), st
    end
    for seed in (31, 32), ad in (AutoZygote(), AutoForwardDiff())
        gt, pt, stt = chain_graph(Xoshiro(seed); tied = true)
        gu, pu, stu = chain_graph(Xoshiro(seed); tied = false)
        scht = SequentialSchedule(tasks(tree_schedule(gt)))
        schiu = SequentialSchedule(tasks(tree_schedule(gu)))
        for nsweeps in (1, 2), damping in (0.0, 0.3)
            gtied = free_energy_gradient(MessageStore(gt), gt, pt, stt; ad, schedule = scht, nsweeps, damping)
            g1 = free_energy_gradient(MessageStore(gu), gu, pu, stu; ad, schedule = schiu, nsweeps, damping)
            @test fl(gtied.shared) ≈ fl(g1.f1) .+ fl(g1.f2) rtol = 1e-9 atol = 1e-12
        end
    end
end

@testset "I17-3 keys in ps: shuffled order, extra key, parameterless factors" begin
    g, ps, st = spec_tree(Xoshiro(41))
    sched = SequentialSchedule(tasks(tree_schedule(g)))
    gz = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps = 2)
    # shuffled key order gives the same gradient per key
    ks = collect(keys(ps))
    ps_shuf = NamedTuple{Tuple(reverse(ks))}(Tuple(ps[k] for k in reverse(ks)))
    gsh = free_energy_gradient(MessageStore(g), g, ps_shuf, st; ad = AutoZygote(), schedule = sched, nsweeps = 2)
    for k in ks
        @test fl(gz[k]) ≈ fl(gsh[k]) rtol = 1e-10
    end
    # an extra key not used by the graph: absent from the envelope driver's result,
    # refused by the replay (coupled) driver
    gd, pd, sd = spec_tree(Xoshiro(42); coupling = DiagonalCoupling())
    _, _, _, s = infer!(gd, tree_schedule(gd), pd, sd; tol = 1e-12, maxsweeps = 100)
    ge = free_energy_gradient(s, gd, merge(pd, (; extra = (x = 1.0,))), sd; ad = AutoZygote())
    @test !haskey(ge, :extra)
    @test keys(ge) == keys(free_energy_gradient(s, gd, pd, sd; ad = AutoZygote()))
    @test_throws ArgumentError free_energy_gradient(MessageStore(g), g,
        merge(ps, (; extra = (x = 1.0,))), st; ad = AutoZygote(), schedule = sched, nsweeps = 2)
    # factors without parameters are fine and get empty gradient entries
    @test haskey(gz, :p1) && isempty(fl(gz.p1))
    @test haskey(gz, :d2) && isempty(fl(gz.d2))
    @test !isempty(fl(gz.f2))
end

@testset "I17-4 detached edges and diagonal cuts in the unrolled replay" begin
    # one detached edge, every other edge ExactCoupling
    for seed in (51, 52)
        g, ps, st = spec_tree(Xoshiro(seed); detach_child = 3)
        sched = SequentialSchedule(tasks(tree_schedule(g)))
        for nsweeps in (1, 2), damping in (0.0, 0.3)
            gz = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps, damping)
            gf = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoForwardDiff(), schedule = sched, nsweeps, damping)
            @test fl(gz) ≈ fl(gf) rtol = 1e-8
        end
    end
    # a graph mixing DiagonalCoupling and ExactCoupling edges (it warns, LP-021)
    for seed in (53, 54)
        g, ps, st = spec_tree(Xoshiro(seed); diagonal_children = (5,), force_parent = (5 => 4,))
        sched = SequentialSchedule(tasks(tree_schedule(g)))
        gz = @test_logs (:warn, r"LP-021") match_mode = :any free_energy_gradient(
            MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps = 2, damping = 0.3)
        gf = @test_logs (:warn, r"LP-021") match_mode = :any free_energy_gradient(
            MessageStore(g), g, ps, st; ad = AutoForwardDiff(), schedule = sched, nsweeps = 2, damping = 0.3)
        @test fl(gz) ≈ fl(gf) rtol = 1e-8
    end
end

@testset "I17-5 implicit adjoint: dense = ForwardDiff, krylov = dense, = unrolled at convergence" begin
    # tree
    g, ps, st = spec_tree(Xoshiro(61))
    sched = tree_schedule(g)
    _, rep, _, s = infer!(g, sched, ps, st; tol = 1e-12, maxsweeps = 1000)
    @test rep.converged
    giz = free_energy_gradient(s, g, ps, st; ad = AutoZygote(), adjoint = :implicit, schedule = sched,
        linear_solver = :dense, tol = 1e-12, linear_tol = 1e-10)
    gif = free_energy_gradient(s, g, ps, st; ad = AutoForwardDiff(), adjoint = :implicit, schedule = sched,
        linear_solver = :dense, tol = 1e-12, linear_tol = 1e-10)
    gik = free_energy_gradient(s, g, ps, st; ad = AutoZygote(), adjoint = :implicit, schedule = sched,
        linear_solver = :krylov, tol = 1e-12, linear_tol = 1e-10)
    gu = free_energy_gradient(s, g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps = 10)
    @test fl(giz) ≈ fl(gif) rtol = 1e-8
    @test fl(gik) ≈ fl(giz) rtol = 1e-6
    @test fl(giz) ≈ fl(gu) rtol = 1e-6
    # loop
    gl, pl, stl = spec_loop(Xoshiro(62); k = 3)
    sl = flooding_schedule(gl)
    _, repl, _, sl_store = infer!(gl, sl, pl, stl; tol = 1e-12, maxsweeps = 2000)
    @test repl.converged
    liz = free_energy_gradient(sl_store, gl, pl, stl; ad = AutoZygote(), adjoint = :implicit, schedule = sl,
        linear_solver = :dense, tol = 1e-12, linear_tol = 1e-10)
    lif = free_energy_gradient(sl_store, gl, pl, stl; ad = AutoForwardDiff(), adjoint = :implicit, schedule = sl,
        linear_solver = :dense, tol = 1e-12, linear_tol = 1e-10)
    lik = free_energy_gradient(sl_store, gl, pl, stl; ad = AutoZygote(), adjoint = :implicit, schedule = sl,
        linear_solver = :krylov, tol = 1e-12, linear_tol = 1e-10)
    @test fl(liz) ≈ fl(lif) rtol = 1e-8
    @test fl(lik) ≈ fl(liz) rtol = 1e-6
end

@testset "I17-6 through infer!: Zygote = ForwardDiff on the marginal loss" begin
    # tree, SequentialSchedule
    g, ps, st = spec_tree(Xoshiro(71))
    sched = SequentialSchedule(tasks(tree_schedule(g)))
    zg = only(Zygote.gradient(p -> marginal_loss(first(infer!(g, sched, p, st; maxsweeps = 100, tol = 1e-10))), ps))
    fd = parameter_gradient(p -> marginal_loss(first(infer!(g, sched, p, st; maxsweeps = 100, tol = 1e-10))), ps; ad = AutoForwardDiff())
    @test fl(zfill(zg, ps)) ≈ fl(fd) rtol = 1e-8
    # loop, FloodingSchedule with damping 0.3
    gl, pl, stl = spec_loop(Xoshiro(72); k = 3)
    sl = flooding_schedule(gl)
    zg = only(Zygote.gradient(p -> marginal_loss(first(infer!(gl, sl, p, stl; maxsweeps = 300, tol = 1e-9, damping = 0.3))), pl))
    fd = parameter_gradient(p -> marginal_loss(first(infer!(gl, sl, p, stl; maxsweeps = 300, tol = 1e-9, damping = 0.3))), pl; ad = AutoForwardDiff())
    @test fl(zfill(zg, pl)) ≈ fl(fd) rtol = 1e-8
end

@testset "I17-7 score coupling: same seed same value, fixed instance literal" begin
    g, ps, st, sched = spec_score_graph()
    e1 = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, rng = Xoshiro(11), nsamples = 64)
    e2 = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, rng = Xoshiro(11), nsamples = 64)
    @test e1 isa GradientEstimate
    @test e1.nsamples == 64
    @test fl(e1.value) == fl(e2.value)
    @test fl(e1.standard_error) == fl(e2.standard_error)
    # value of this fixed instance, computed on the current code (2026-11-21):
    if isdefined(Main, :SCORE_VALUE_LITERAL)
        @test fl(e1.value) ≈ SCORE_VALUE_LITERAL rtol = 1e-10
    else
        println("SCORE_VALUE_LITERAL = ", repr(fl(e1.value)))
    end
    if isdefined(Main, :SCORE_SE_LITERAL)
        @test fl(e1.standard_error) ≈ SCORE_SE_LITERAL rtol = 1e-10
    else
        println("SCORE_SE_LITERAL = ", repr(fl(e1.standard_error)))
    end
end

@testset "I17-8 fixed gradient values per path (AutoZygote, current code)" begin
    # path 1: unrolled replay
    g, ps, st = spec_tree(Xoshiro(101))
    sched = SequentialSchedule(tasks(tree_schedule(g)))
    gu = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps = 2, damping = 0.3)
    if isdefined(Main, :UNROLLED_LITERAL)
        @test fl(gu) ≈ UNROLLED_LITERAL rtol = 1e-10
    else
        println("UNROLLED_LITERAL = ", repr(fl(gu)))
    end
    # path 2: implicit adjoint
    gl, pl, stl = spec_loop(Xoshiro(102); k = 3)
    sl = flooding_schedule(gl)
    _, _, _, s = infer!(gl, sl, pl, stl; tol = 1e-12, maxsweeps = 2000)
    gi = free_energy_gradient(s, gl, pl, stl; ad = AutoZygote(), adjoint = :implicit, schedule = sl,
        linear_solver = :dense, tol = 1e-12, linear_tol = 1e-10)
    if isdefined(Main, :IMPLICIT_LITERAL)
        @test fl(gi) ≈ IMPLICIT_LITERAL rtol = 1e-10
    else
        println("IMPLICIT_LITERAL = ", repr(fl(gi)))
    end
    # path 3: through infer!
    g3, ps3, st3 = spec_tree(Xoshiro(103))
    sched3 = SequentialSchedule(tasks(tree_schedule(g3)))
    zg = only(Zygote.gradient(p -> marginal_loss(first(infer!(g3, sched3, p, st3; maxsweeps = 100, tol = 1e-10))), ps3))
    if isdefined(Main, :INFER_LITERAL)
        @test fl(zfill(zg, ps3)) ≈ INFER_LITERAL rtol = 1e-10
    else
        println("INFER_LITERAL = ", repr(fl(zfill(zg, ps3))))
    end
end

@testset "I17-9 scaling: Zygote unrolled gradient bytes at 4T over T (kalman chain)" begin
    function kalman(T; coupling = ExactCoupling())
        g, ps, st = chain(T)
        for i in eachindex(g.edges)
            e = g.edges[i]
            g.edges[i] = Edge(e.factor, e.channel, e.variable, e.direction, coupling, e.detach)
        end
        return g, ps, st
    end
    function zygote_bytes(T; nsweeps = 1)
        g, ps, st = kalman(T)
        sched = tree_schedule(g)
        free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps)  # warm-up
        GC.gc()
        return @allocated free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched, nsweeps)
    end
    b4 = zygote_bytes(4)
    b16 = zygote_bytes(16)
    ratio = b16 / b4
    @info "I17-9 Zygote unrolled gradient allocation on the kalman chain" b4 b16 ratio
    @test ratio < 8
end
end
