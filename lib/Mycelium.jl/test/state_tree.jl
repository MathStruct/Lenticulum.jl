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
