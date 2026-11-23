# I4: the residual schedule's priority queue against the full scan it replaced
# (`Mycelium._residual_sweep_scan!`). With the dependencies right, the queue probes the same residuals
# the scan computes, so on any graph both commit the same tasks in the same order and leave the same
# store, bit for bit. Random loopy graphs with every kind of reader the dependencies distinguish:
# belief-propagation nodes, mean-field nodes (read marginals), expectation-propagation nodes and a
# nonlinear factor (read the target's cavity), damping (a message reads its own previous value).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using LenticulumCore: GaussianBelief, MixtureBelief

# Structural equality, arrays by value: the beliefs and messages of two stores.
rq_same(a, b) = typeof(a) === typeof(b) && isequal(a, b)
rq_same(a::T, b::T) where {T<:Number} = isequal(a, b)
rq_same(a::T, b::T) where {T<:Union{Float16,Float32,Float64}} = a === b
rq_same(a::T, b::T) where {T<:AbstractArray} =
    size(a) == size(b) && all(rq_same(x, y) for (x, y) in zip(a, b))
function rq_same(a::T, b::T) where {T}
    isstructtype(T) && fieldcount(T) > 0 || return a == b
    return all(rq_same(getfield(a, k), getfield(b, k)) for k in 1:fieldcount(T))
end
rq_same_store(s1, s2) = s1.iteration == s2.iteration && isequal(s1.pins, s2.pins) &&
    all(rq_same(s1.to_variable[e], s2.to_variable[e]) && rq_same(s1.to_factor[e], s2.to_factor[e]) for e in eachindex(s1.to_variable))

# A clutter likelihood as a mixture message: a Gaussian bump and a flat component.
function rq_clutter(y, w)
    comps = [GaussianBelief([y], fill(1.0, 1, 1), true), GaussianBelief([0.0], zeros(1, 1), true)]
    return MixtureBelief(Val(:log), comps, [log(1 - w) - y^2 / 2, log(w) - 3.0])
end

# A unary source whose message reads its state, a counter that every message it sends increments:
# a commit changes the messages of every node that shares its parameter key.
struct RQCounter <: AbstractLenticulumFactor end
LenticulumCore.channels(::RQCounter) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supported_polarities(::RQCounter) = (LenticulumCore.Polarity(x = LenticulumCore.Unobserved()),)
LenticulumCore.supports_polarity(::RQCounter, p::LenticulumCore.Polarity) = p[:x] isa LenticulumCore.Unobserved
LuxCore.initialstates(::AbstractRNG, ::RQCounter) = (n = 0,)
Mycelium.repeatable_messages(::RQCounter) = true
Mycelium.factor_message(::RQCounter, ::Symbol, _, _, _, ps, st) =
    (n = get(st, :n, 0); (Gaussian(1 / (1 + n), 2.0), (n = n + 1,)))

# A unary source whose message reads the target's cavity and declares neither `exact_messages` nor
# `message_uses_cavity` (as `DEQFactor` and `DiffusionFactor` do): its message must be probed again
# when the other messages into its variable change.
struct RQCavity <: AbstractLenticulumFactor end
LenticulumCore.channels(::RQCavity) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supported_polarities(::RQCavity) = (LenticulumCore.Polarity(x = LenticulumCore.Unobserved()),)
LenticulumCore.supports_polarity(::RQCavity, p::LenticulumCore.Polarity) = p[:x] isa LenticulumCore.Unobserved
Mycelium.repeatable_messages(::RQCavity) = true
Mycelium.factor_message(::RQCavity, ::Symbol, _, _, prior, ps, st) =
    (Gaussian(prior isa GaussianBelief ? 0.5 * belief_mean(prior)[1] : 0.0, 3.0), st)

function rq_graph(rng; nv = rand(rng, 3:6), nf = rand(rng, nv:nv + 4), meanfield = true, ep = true, nonlinear = true,
                  stateful = false, cavity = false)
    b = GraphBuilder()
    for i in 1:nv
        variable!(b, Symbol(:x, i), 1)
        factor!(b, Symbol(:p, i), GaussianPrior(Symbol(:x, i), randn(rng), 0.5 + 2rand(rng)))
        connect!(b, Symbol(:p, i), Symbol(:x, i), Symbol(:x, i))
    end
    for j in 1:nf
        k = rand(rng) < 0.25 ? 3 : 2
        vs = randperm(rng, nv)[1:min(k, nv)]
        chans = Tuple(Symbol(:c, i) for i in eachindex(vs))
        rule = meanfield && rand(rng) < 0.3 ? MeanField() : BeliefPropagation()
        name = Symbol(:f, j)
        factor!(b, name, LinearConstraintFactor(NamedTuple{chans}(ntuple(_ -> 1, length(vs))), 1;
                                                noise = 0.2 + rand(rng)); rule)
        for (c, v) in zip(chans, vs); connect!(b, name, c, Symbol(:x, v)); end
    end
    if ep
        for j in 1:rand(rng, 1:2)
            name = Symbol(:ep, j)
            factor!(b, name, PriorFactor(:x, rq_clutter(2randn(rng), 0.2), x -> 0.0); rule = ExpectationPropagationRule())
            connect!(b, name, :x, Symbol(:x, rand(rng, 1:nv)))
        end
    end
    if nonlinear && nv >= 2
        a, c = randperm(rng, nv)[1:2]
        factor!(b, :nl, NonlinearFactor(x -> tanh.(x), 1 => 1; noise = 0.5))
        connect!(b, :nl, :x, Symbol(:x, a)); connect!(b, :nl, :y, Symbol(:x, c))
    end
    if cavity
        factor!(b, :cav, RQCavity())
        connect!(b, :cav, :x, Symbol(:x, rand(rng, 1:nv)))
    end
    if stateful                         # two counters tied to one key, and one of its own
        for (j, key) in ((1, :tied), (2, :tied), (3, :own))
            name = Symbol(:cnt, j)
            factor!(b, name, RQCounter(); params = key)
            connect!(b, name, :x, Symbol(:x, rand(rng, 1:nv)))
        end
    end
    g = build(b)
    return g, LuxCore.initialparameters(rng, g), LuxCore.initialstates(rng, g)
end

@testset "I4: the residual queue commits what the full scan commits" begin
    ngraphs = 0; nfailed = 0
    for seed in 1:40
        rng = Xoshiro(seed)
        g, ps, st = rq_graph(rng; meanfield = seed % 4 != 1, ep = seed % 3 != 0, nonlinear = seed % 2 == 0,
                            stateful = seed % 5 < 2, cavity = seed % 3 == 1)
        n = length(all_tasks(g))
        for damping in (0.0, 0.3), maxtasks in (n, 3)
            sched = ResidualSchedule(; maxtasks, reuse = false)
            s1, s2 = MessageStore(g), MessageStore(g)
            r1, r2 = MessageTask[], MessageTask[]
            st1, st2 = st, st
            ok = true
            for sweep in 1:8
                # A random graph can fail (an expectation-propagation site with a negative precision
                # reaching a factor that refuses it): then both must fail, after the same commits.
                a = try sweep!(s1, g, sched, ps, st1; damping, record = r1) catch err; err end
                b = try Mycelium._residual_sweep_scan!(s2, g, sched, ps, st2; damping, record = r2) catch err; err end
                if a isa Exception || b isa Exception
                    ok &= a isa Exception && b isa Exception && typeof(a) == typeof(b) && r1 == r2
                    nfailed += 1
                    break
                end
                (res1, st1), (res2, st2) = a, b
                ok &= r1 == r2 && rq_same(res1, res2) && rq_same_store(s1, s2) && st1 == st2
            end
            @test ok
            ok || @info "queue and scan differ" seed damping maxtasks length(r1) length(r2) findfirst(i -> r1[i] != r2[i], 1:min(length(r1), length(r2)))
            ngraphs += 1
        end
    end
    @test ngraphs == 160 && nfailed < 20     # most runs go through

    # The queue reaches a fixed point on this graph.
    rng = Xoshiro(7)
    g, ps, st = rq_graph(rng; ep = false)
    a = propagate!(MessageStore(g), g, ResidualSchedule(), ps, st; maxsweeps = 500, tol = 1e-10)[1]
    @test a.converged
end

@testset "I31: retained and fresh queues agree" begin
    failures = zeros(Int, 2)
    for seed in 1:40
        g, ps0, st = rq_graph(Xoshiro(seed); meanfield = seed % 4 != 1, ep = seed % 3 != 0,
                              nonlinear = seed % 2 == 0, stateful = seed % 5 < 2, cavity = seed % 3 == 1)
        @test all(fn -> repeatable_messages(fn.factor), g.factors)
        n = length(all_tasks(g))
        for damping in (0.0, 0.3), maxtasks in (n, 3), mutate in (false, true)
            ps = ps0
            s1, s2 = MessageStore(g), MessageStore(g)
            on = ResidualSchedule(; maxtasks)
            off = ResidualSchedule(; maxtasks, reuse = false)
            r1, r2 = MessageTask[], MessageTask[]
            st1, st2 = st, st
            ok = true
            for sweep in 1:8
                if mutate
                    if sweep == 2
                        for s in (s1, s2)
                            s.to_variable[1] = Mycelium.Message(Gaussian([7.0], ones(1, 1)), s.iteration)
                        end
                    elseif sweep == 3
                        e = first(g.edges_of_factor[g.factor_index[:f1]])
                        for s in (s1, s2)
                            s.to_factor[e] = Mycelium.Message(Gaussian([-5.0], ones(1, 1)), s.iteration)
                        end
                    elseif sweep == 4
                        c = copy(ps.f1.c)
                        c[1] += 2.0
                        ps = merge(ps, (f1 = merge(ps.f1, (; c)),))
                    elseif sweep == 5
                        pin!(s1, g, :x1, [0.25])
                        pin!(s2, g, :x1, [0.25])
                    elseif sweep == 6
                        # Copy an independently advanced store into both existing stores.
                        source = MessageStore(g)
                        sweep!(source, g, off, ps, st; damping)
                        for s in (s1, s2)
                            copyto!(s.to_factor, source.to_factor)
                            copyto!(s.to_variable, source.to_variable)
                            s.iteration = source.iteration
                        end
                    elseif sweep == 7 && haskey(st, :tied)
                        st1 = merge(st1, (tied = (n = 1,),))
                        st2 = merge(st2, (tied = (n = 1,),))
                    end
                end
                # A one-sweep report also checks the reported residual and convergence flag.
                a = try propagate!(s1, g, on, ps, st1; damping, maxsweeps = 1, record = r1) catch err; err end
                b = try propagate!(s2, g, off, ps, st2; damping, maxsweeps = 1, record = r2) catch err; err end
                ok &= r1 == r2 && rq_same_store(s1, s2)
                if a isa Exception || b isa Exception
                    ok &= a isa Exception && b isa Exception && typeof(a) == typeof(b)
                    failures[1 + mutate] += 1
                    break
                end
                (rep1, st1), (rep2, st2) = a, b
                ok &= rq_same(rep1, rep2) && rq_same(st1, st2)
            end
            @test ok
            ok || @info "retained and fresh queues differ" seed damping maxtasks mutate
        end
    end
    @test all(<(20), failures)
end

@testset "I31: cache settings and the partial-sweep report" begin
    g, ps, st = rq_graph(Xoshiro(7); ep = false, cavity = true, stateful = true)
    s1, s2 = MessageStore(g), MessageStore(g)
    on, off = ResidualSchedule(3), ResidualSchedule(3; reuse = false)
    r1, r2 = MessageTask[], MessageTask[]
    for (run, (graph, damping, rtol)) in enumerate(((g, 0.0, 0.0), (g, 0.3, 0.0),
                                                  (g, 0.3, 1e-6), (g, 0.3, 2e-6),
                                                  (deepcopy(g), 0.3, 2e-6)))
        kw = (; damping, rtol, tol = 1e-8, maxsweeps = 4)
        if run == 4
            # A new store copied from a store already seen by the schedule.
            s1, s2 = deepcopy(s1), deepcopy(s2)
        end
        rep1, _ = propagate!(s1, graph, on, ps, st; kw..., record = r1)
        rep2, _ = propagate!(s2, graph, off, ps, st; kw..., record = r2)
        @test r1 == r2 && rq_same_store(s1, s2) && rq_same(rep1, rep2)
    end

    g, ps, st = rq_graph(Xoshiro(7); ep = false)
    s1, s2 = MessageStore(g), MessageStore(g)
    r1, r2 = MessageTask[], MessageTask[]
    kw = (; criterion = :marginals, tol = 1e-8, maxsweeps = 500)
    rep1, st1 = propagate!(s1, g, on, ps, st; kw..., record = r1)
    rep2, st2 = propagate!(s2, g, off, ps, st; kw..., record = r2)
    @test rep1.converged && rq_same(rep1, rep2)
    @test r1 == r2 && rq_same_store(s1, s2) && rq_same(st1, st2)
    @test isequal(Mycelium._max_pending(s1, g, ps, st1, 0.0, on),
                  Mycelium._max_pending(s2, g, ps, st2, 0.0))
end

struct RQRandomSource <: AbstractLenticulumFactor
    rng::Xoshiro
end
LenticulumCore.channels(::RQRandomSource) = LenticulumCore.channels(RQCounter())
LenticulumCore.supported_polarities(::RQRandomSource) = LenticulumCore.supported_polarities(RQCounter())
LenticulumCore.supports_polarity(::RQRandomSource, p::LenticulumCore.Polarity) = LenticulumCore.supports_polarity(RQCounter(), p)
Mycelium.factor_message(f::RQRandomSource, ::Symbol, _, _, _, ps, st) = (Gaussian(rand(f.rng), 2.0), st)

@testset "I31: an RNG factor keeps fresh probes" begin
    function rng_graph()
        b = GraphBuilder()
        variable!(b, :x, 1)
        factor!(b, :random, RQRandomSource(Xoshiro(1)))
        connect!(b, :random, :x, :x; direction = Emitting())
        return build(b)
    end
    g1, g2 = rng_graph(), rng_graph()
    s1, s2 = MessageStore(g1), MessageStore(g2)
    on, off = ResidualSchedule(), ResidualSchedule(; reuse = false)
    ps, st = LuxCore.setup(Xoshiro(2), g1)
    r1, r2 = MessageTask[], MessageTask[]
    @test !repeatable_messages(only(g1.factors).factor)
    for _ in 1:8
        a = sweep!(s1, g1, on, ps, st; record = r1)
        b = sweep!(s2, g2, off, ps, st; record = r2)
        @test rq_same(a, b) && rq_same_store(s1, s2) && r1 == r2
        @test isempty(on.memory)
    end
    @test only(g1.factors).factor.rng == only(g2.factors).factor.rng
end

@testset "I31: a fresh scan and a retained queue both reject a contradictory cavity" begin
    b = GraphBuilder()
    variable!(b, :x, 1)
    variable!(b, :y, 1)
    for (name, f, target) in ((:first, DataFactor(:p, [1.0]), :x),
                              (:second, DataFactor(:p, [2.0]), :x),
                              (:independent, GaussianPrior(:p, 0.0, 1.0), :y),
                              (:third, DataFactor(:p, [3.0]), :x))
        factor!(b, name, f)
        connect!(b, name, :p, target; direction = Emitting())
    end
    g = build(b)
    ps, st = LuxCore.setup(Xoshiro(1), g)
    for external in (false, true)
        s1, s2 = MessageStore(g), MessageStore(g)
        r1, r2 = MessageTask[], MessageTask[]
        for (s, record, reuse) in ((s1, r1, true), (s2, r2, false))
            sched = ResidualSchedule(; maxtasks = 1, reuse)
            sweep!(s, g, sched, ps, st; record)
            if external
                s.to_variable[2] = Mycelium.Message(LenticulumCore.DiracBelief([2.0]), s.iteration)
            else
                sweep!(s, g, sched, ps, st; record)
            end
            @test_throws ArgumentError sweep!(s, g, sched, ps, st; record)
            @test s.to_variable[3] === nothing  # error before the independent commit
        end
        @test r1 == r2 && rq_same_store(s1, s2)
    end
end

# The findings of the review (LP-072), from Codex's probes: each was a difference between the queue
# and the scan, and is now their regression test.
function rq_paired(g, ps, st; maxtasks = length(all_tasks(g)), damping = 0.0, sweeps = 1, setup = identity)
    s1, s2 = setup(MessageStore(g)), setup(MessageStore(g))
    r1, r2 = MessageTask[], MessageTask[]
    st1, st2 = st, st
    sched = ResidualSchedule(; maxtasks)
    ok = true
    for _ in 1:sweeps
        a = try sweep!(s1, g, sched, ps, st1; damping, record = r1) catch err; err end
        b = try Mycelium._residual_sweep_scan!(s2, g, sched, ps, st2; damping, record = r2) catch err; err end
        (a isa Exception || b isa Exception) && return (; ok = false, a, b, r1, r2, s1, s2)
        ok &= r1 == r2 && isequal(a[1], b[1]) && rq_same_store(s1, s2)
        st1, st2 = a[2], b[2]
    end
    return (; ok, r1, r2, s1, s2)
end

struct RQPotential <: LenticulumCore.AbstractPotentialBelief end     # no distance: `Inf` to itself
Mycelium.belief_logdensity(::RQPotential, x) = -sum(abs2, x)

struct RQBigPrior <: AbstractLenticulumFactor end                   # residuals in BigFloat
LuxCore.parameterlength(::RQBigPrior) = 1
LenticulumCore.channels(::RQBigPrior) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supported_polarities(::RQBigPrior) = (LenticulumCore.Polarity(x = LenticulumCore.Unobserved()),)
LenticulumCore.supports_polarity(::RQBigPrior, p::LenticulumCore.Polarity) = p[:x] isa LenticulumCore.Unobserved
Mycelium.factor_message(::RQBigPrior, ::Symbol, _, _, _, ps, st) = (GaussianBelief(BigFloat[ps.eta], ones(BigFloat, 1, 1)), st)

@testset "I4: the review's findings (LP-072)" begin
    # F1: a committed message's distance to itself is not always 0 (`Inf` without a distance).
    b = GraphBuilder(); variable!(b, :x, 1); variable!(b, :y, 1)
    factor!(b, :p, PriorFactor(:x, RQPotential(), x -> sum(abs2, x))); connect!(b, :p, :x, :x; direction = Emitting())
    factor!(b, :q, GaussianPrior(:y, 0.0, 1.0)); connect!(b, :q, :y, :y; direction = Emitting())
    g = validate(build(b)); ps, st = LuxCore.setup(Xoshiro(1), g)
    @test rq_paired(g, ps, st; sweeps = 3).ok

    # F1: NaN to itself, with a NaN message already stored on the second edge.
    b = GraphBuilder()
    for n in (:x, :y)
        variable!(b, n, 1); factor!(b, n, DataFactor(:p, [NaN])); connect!(b, n, :p, n; direction = Emitting())
    end
    g = validate(build(b)); ps, st = LuxCore.setup(Xoshiro(1), g)
    r = rq_paired(g, ps, st; setup = s -> (step!(s, g, all_tasks(g)[2], ps, st); s))
    @test r.ok && r.r1 == r.r2
    s = MessageStore(g)
    sched = ResidualSchedule(1)
    sweep!(s, g, sched, ps, st)
    # The pending check must still report the NaN even with an Inf at the top of the heap.
    @test isnan(Mycelium._max_pending(s, g, ps, st, 0.0, sched))

    # F2: BigFloat residuals that round to one Float64, or to 0.
    b = GraphBuilder()
    for n in (:x, :y)
        variable!(b, n, 1); factor!(b, n, RQBigPrior()); connect!(b, n, :x, n; direction = Emitting())
    end
    g = validate(build(b)); st = (x = NamedTuple(), y = NamedTuple())
    ps0 = (x = (eta = big"0.0",), y = (eta = big"0.0",))
    warm(s) = (sweep!(s, g, SequentialSchedule(all_tasks(g)), ps0, st); s)
    for (x, y, mt) in ((big"1.0", 1 + big"2.0"^(-80), 1), (big"2.0"^(-1200), big"2.0"^(-1201), 2))
        r = rq_paired(g, (x = (eta = x,), y = (eta = y,)), st; setup = warm, maxtasks = mt)
        @test r.ok && length(r.r1) == mt
    end

    # F3: no probe after the last commit of the budget (an expectation-propagation probe there threw
    # where the scan returned). F4, recorded and not changed: with BeliefPropagation the graph
    # (contradictory hard clamps) throws in both, at different commits.
    function hard_graph(rule)
        b = GraphBuilder(); variable!(b, :x, 1); variable!(b, :y, 1)
        for (n, v) in ((:clamp1, [1.0]), (:clamp2, [2.0]))
            factor!(b, n, DataFactor(:x, v)); connect!(b, n, :x, :x; direction = Emitting())
        end
        factor!(b, :prior, GaussianPrior(:y, 0.0, 1.0)); connect!(b, :prior, :y, :y; direction = Emitting())
        factor!(b, :hard, HardConstraintFactor((x = 1,), 1); rule); connect!(b, :hard, :x, :x; direction = Emitting())
        return validate(build(b))
    end
    g = hard_graph(ExpectationPropagationRule()); ps, st = LuxCore.setup(Xoshiro(1), g)
    @test rq_paired(g, ps, st; maxtasks = 2).ok
    g = hard_graph(BeliefPropagation()); ps, st = LuxCore.setup(Xoshiro(1), g)
    r = rq_paired(g, ps, st; maxtasks = 4)
    @test r.a isa ArgumentError && r.b isa ArgumentError
    # the same graphs with the variable pinned agree, and a commit at a pinned variable has no dependents
    for rule in (BeliefPropagation(), ExpectationPropagationRule())
        g = hard_graph(rule); ps, st = LuxCore.setup(Xoshiro(1), g)
        @test rq_paired(g, ps, st; maxtasks = 2, sweeps = 3, setup = s -> (pin!(s, g, :x, [0.0]); s)).ok
    end
end
