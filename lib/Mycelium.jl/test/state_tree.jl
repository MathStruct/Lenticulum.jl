# The graph-wide state tree is written back only when a factor changed its state, and the check of
# tied state is linear (LP-024: the free energy was quadratic in the number of factors).
using LuxCore, Random

# A unary source with a counter in its state: every message it sends increments it.
struct CountingSource <: AbstractLenticulumFactor end
LenticulumCore.channels(::CountingSource) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supported_polarities(::CountingSource) = (Polarity(x = Unobserved()),)
LenticulumCore.supports_polarity(::CountingSource, p::Polarity) = p[:x] isa Unobserved
LuxCore.initialstates(::AbstractRNG, ::CountingSource) = (n = 0,)
Mycelium.factor_message(::CountingSource, ::Symbol, _, _, _, ps, st) = (Gaussian(0.0, 1.0), (n = get(st, :n, 0) + 1,))

# A source that counts in its state when `count` is set and hands its state back unchanged otherwise.
struct MaybeCount <: AbstractLenticulumFactor
    count::Bool
end
LenticulumCore.channels(::MaybeCount) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supported_polarities(::MaybeCount) = (Polarity(x = Unobserved()),)
LenticulumCore.supports_polarity(::MaybeCount, p::Polarity) = p[:x] isa Unobserved
_maybe(f::MaybeCount, st) = f.count ? (n = get(st, :n, 0) + 1,) : st
Mycelium.factor_message(f::MaybeCount, ::Symbol, _, _, _, ps, st) = (Gaussian(0.0, 1.0), _maybe(f, st))
Mycelium.local_free_energy(f::MaybeCount, beliefs, ps, st) = (0.0, _maybe(f, st))

# `k` separate variables, each with a stateless `DataFactor` (so the flooding schedule); with
# `counting`, a `CountingSource` on the first.
function state_graph(k; counting = false, tied = false)
    b = GraphBuilder()
    for i in 1:k
        variable!(b, Symbol(:x, i), 1)
        factor!(b, Symbol(:d, i), DataFactor(:x, [Float64(i)])); connect!(b, Symbol(:d, i), :x, Symbol(:x, i))
    end
    counting && (factor!(b, :c, CountingSource()); connect!(b, :c, :x, :x1))
    tied && for i in 1:2                                    # two counting sources sharing one key
        factor!(b, Symbol(:t, i), CountingSource(); params = :t); connect!(b, Symbol(:t, i), :x, Symbol(:x, i))
    end
    g = build(b)
    return g, LuxCore.initialparameters(Xoshiro(1), g), LuxCore.initialstates(Xoshiro(1), g)
end

@testset "state tree: written back only when a factor changes it" begin
    nt = (a = (;), b = (n = 1,))
    @test Mycelium._setsubtree(nt, :a, nt.a) === nt                  # unchanged: the same tree
    @test Mycelium._setsubtree(nt, :b, nt.b) === nt
    @test Mycelium._setsubtree(nt, :b, (n = 2,)) == (a = (;), b = (n = 2,))
    @test Mycelium._setsubtree(nt, :c, (;)) == (a = (;), b = (n = 1,), c = (;))   # a missing key is added
    @test Mycelium._setsubtree(nothing, :a, (;)) == NamedTuple()
    @test Mycelium._writeback(nt, :a, Mycelium._stored(nt, :a), (;)) === nt
    @test Mycelium._writeback(nt, :c, Mycelium._stored(nt, :c), (;)) == (a = (;), b = (n = 1,), c = (;))

    # A stateless graph: a sweep and the free energy hand back the very tree they were given.
    g, ps, st = state_graph(20)
    s = MessageStore(g); sched = flooding_schedule(g)
    _, st1 = sweep!(s, g, sched, ps, st)
    @test st1 === st
    @test last(Mycelium._factor_terms(s, g, ps, st)) === st

    # A stateful factor still has its state written back, and only its own key changes.
    g, ps, st = state_graph(5; counting = true)
    s = MessageStore(g); sched = flooding_schedule(g)
    _, st1 = sweep!(s, g, sched, ps, st)
    _, st2 = sweep!(s, g, sched, ps, st1)
    @test st.c.n == 0 && st1.c.n == 1 && st2.c.n == 2
    @test keys(st2) == keys(st)
    @test all(getfield(st2, k) === getfield(st, k) for k in keys(st) if k !== :c)

    # Keys missing from the tree are added as before: callers read `st[name]` after `infer!`.
    g, ps, _ = state_graph(3; counting = true)
    s = MessageStore(g); sched = flooding_schedule(g)
    _, st1 = sweep!(s, g, sched, ps, NamedTuple())
    @test st1.c == (n = 1,) && haskey(st1, :d1) && st1.d1 == NamedTuple()
    g, ps, _ = state_graph(3)
    s = MessageStore(g); sweep!(s, g, flooding_schedule(g), ps, NamedTuple())
    @test haskey(last(Mycelium._factor_terms(s, g, ps, NamedTuple())), :d2)

    # Tied factors that create state from an empty one: the free energy's factor terms all read the
    # original tree, and the last factor's state is kept, as before the write-back was skipped
    # (Mistral's review, 2026-11-11: `t` came out `(n = 1,)` instead of `(;)`).
    b = GraphBuilder(); variable!(b, :x1, 1); variable!(b, :x2, 1)
    factor!(b, :d1, DataFactor(:x, [2.0])); connect!(b, :d1, :x, :x1)
    factor!(b, :t1, MaybeCount(true); params = :t); connect!(b, :t1, :x, :x1)
    factor!(b, :t2, MaybeCount(false); params = :t); connect!(b, :t2, :x, :x2)
    g = build(b); ps = LuxCore.initialparameters(Xoshiro(1), g)
    s = MessageStore(g); sweep!(s, g, flooding_schedule(g), ps, (d1 = (;), t = (;)))
    @test last(Mycelium._factor_terms(s, g, ps, (d1 = (;), t = (;)))).t == NamedTuple()

    # Tied nodes with a non-empty state are refused, whatever the number of factors.
    g, ps, st = state_graph(3; tied = true)
    @test_throws ArgumentError Mycelium._check_tied_state(g, merge(st, (t = (n = 1,),)))
    @test Mycelium._check_tied_state(g, merge(st, (t = (;),))) === nothing

    # The check that tied nodes are stateless is linear in the number of factors: it counted the
    # factors of every key, O(keys × factors), half the free energy of a 2 401-factor chain
    # (`bench/state_scaling.jl`). Four times the factors: 16 times the time before, 4 now.
    check_time(k) = begin
        g, _, st = state_graph(k)
        Mycelium._check_tied_state(g, st)
        minimum(_ -> @elapsed(Mycelium._check_tied_state(g, st)), 1:5)
    end
    small, large = check_time(2000), check_time(8000)
    @test large < 8 * small
end

# The trees by position (I5; `passing.md`, "The trees by position"): a lookup by a run-time
# `Symbol` scanned the tree's names, so a sweep and the free energy were quadratic in the number
# of factors (×3.3–3.5 per doubling at 2 401 factors, `bench/state_scaling.jl`).

# `k` variables, each with a `DataFactor` and, with `stateful`, a counting `MaybeCount`.
function counting_graph(k; stateful = true)
    b = GraphBuilder()
    for i in 1:k
        variable!(b, Symbol(:x, i), 1)
        factor!(b, Symbol(:d, i), DataFactor(:x, [Float64(i)])); connect!(b, Symbol(:d, i), :x, Symbol(:x, i))
        stateful && (factor!(b, Symbol(:m, i), MaybeCount(true)); connect!(b, Symbol(:m, i), :x, Symbol(:x, i)))
    end
    g = build(b)
    return g, LuxCore.initialparameters(Xoshiro(1), g), LuxCore.initialstates(Xoshiro(1), g)
end

@testset "state tree: linear in the number of factors" begin
    # Four times the factors: about 16 times the time when quadratic, 4 when linear.
    function times(k; stateful)
        g, ps, st = counting_graph(k; stateful)
        s = MessageStore(g); sched = flooding_schedule(g)
        sweep!(s, g, sched, ps, st); Mycelium._factor_terms(s, g, ps, st)
        ts = minimum(_ -> @elapsed(sweep!(s, g, sched, ps, st)), 1:5)
        tf = minimum(_ -> @elapsed(Mycelium._factor_terms(s, g, ps, st)), 1:5)
        return ts, tf
    end
    for stateful in (false, true)
        (s1, f1), (s4, f4) = times(500; stateful), times(2000; stateful)
        @test s4 < 8 * s1
        @test f4 < 8 * f1
    end
end

@testset "state tree: write-backs by position match the write-backs by name" begin
    # The old write-back, one `merge` per write, as the reference for the free energy's one `merge`.
    function reference(g, st, results)
        foldl(zip(g.factors, results); init = st) do acc, (f, (_, new, old))
            Mycelium._writeback(acc, f.params, f.params === f.name ? old : nothing, new)
        end
    end
    rng = Xoshiro(7)
    for trial in 1:50
        b = GraphBuilder(); n = rand(rng, 2:8)
        for i in 1:n; variable!(b, Symbol(:x, i), 1); end
        for j in 1:rand(rng, 2:10)
            name, x = Symbol(:f, j), Symbol(:x, rand(rng, 1:n))
            if rand(rng) < 0.3                                  # a tied key: stateless, written anyway
                factor!(b, name, MaybeCount(false); params = Symbol(:t, rand(rng, 1:2)))
            else
                factor!(b, name, MaybeCount(rand(rng, Bool)))
            end
            connect!(b, name, :x, x)
        end
        g = build(b); ps = LuxCore.initialparameters(rng, g)
        full = LuxCore.initialstates(rng, g)
        ks = Tuple(k for k in keys(full) if rand(rng) < 0.6)    # some keys missing, in a shuffled order
        ks = ks[randperm(rng, length(ks))]
        st = NamedTuple{ks}(Tuple(getfield(full, k) for k in ks))
        s = MessageStore(g); sweep!(s, g, flooding_schedule(g), ps, st)
        F, st1 = Mycelium._factor_terms(s, g, ps, st)
        results = map(g.factors) do f                          # the same terms, read by name
            stf0 = Mycelium._stored(st, f.params)
            E, new = Mycelium.local_free_energy(f.factor, Mycelium.messages_into(s, g, f.id),
                                                Mycelium._subtree(ps, f.params), something(stf0, NamedTuple()))
            (E, new, stf0)
        end
        ref = reference(g, st, results)
        @test keys(st1) == keys(ref)
        @test st1 == ref

        # A sweep: the buffer against `step!` on the NamedTuple, task by task.
        sched = SequentialSchedule(tasks(flooding_schedule(g)))
        s1, s2 = MessageStore(g), MessageStore(g)
        _, a = sweep!(s1, g, sched, ps, st)
        r = st; s2.iteration += 1
        for t in tasks(sched); _, _, r = step!(s2, g, t, ps, r); end
        @test keys(a) == keys(r) && a == r
    end
end

@testset "state tree: the residual schedule's probes do not write state" begin
    g, ps, st = state_graph(3; counting = true)
    s = MessageStore(g); rec = Mycelium.MessageTask[]
    _, st1 = sweep!(s, g, ResidualSchedule(maxtasks = 5), ps, st; record = rec)
    c = only(e for e in g.edges_of_factor[g.factor_index[:c]])
    @test st1.c.n == count(t -> t.kind === :to_variable && t.edge == c, rec)
end
