# The planned factor -> variable message (I18; `src/polarity_resolution.md`, difficulty 3). `step!`
# keeps, per (edge, set of informative inputs), the polarity, its legality and the input edges. The
# test compares it with the code it replaced (resolve, check, gather the inputs by name, call the
# rule), on random wirings, edge directions and store contents, for a factor with a few channels
# (planned, cached) and one with more than 64 (planned afresh on every message).
using Random

# A factor on `n` channels that records each call: what it was asked, and with which inputs.
struct EchoFactor <: AbstractLenticulumFactor
    names::Tuple{Vararg{Symbol}}
    free::Bool   # whether it emits with nothing observed
end
const ECHO_CALLS = Any[]
const ECHO_LOCK = ReentrantLock()
LenticulumCore.channels(f::EchoFactor) = map(c -> LenticulumCore.Channel(c, nothing), f.names)
LenticulumCore.hasnopolarity(::EchoFactor) = false
LenticulumCore.supports_polarity(f::EchoFactor, p::Polarity) =
    f.free || !isempty(LenticulumCore.observed_channels(p))
function Mycelium.factor_message(::EchoFactor, target::Symbol, pol, inputs, prior, ps, st)
    @lock ECHO_LOCK push!(ECHO_CALLS, (target, pol, inputs, prior))
    return DiracBelief(1.0), st
end

function echo_graph(rng, n; free, rule)
    names = Tuple(Symbol(:c, i) for i in 1:n)
    b = GraphBuilder()
    factor!(b, :f, EchoFactor(names, free); rule)
    for (i, c) in enumerate(names)
        v = Symbol(:v, i)
        variable!(b, v)
        connect!(b, :f, c, v; direction = rand(rng, (Bidirectional(), Bidirectional(), Absorbing(), Emitting())))
        # a second factor on the variable, so that the target has a prior to exclude
        factor!(b, Symbol(:d, i), DataFactor(v, Float64(i)))
        connect!(b, Symbol(:d, i), v, v; direction = Emitting())
    end
    return build(b)
end

# The store: each input of `f` is missing, uninformative or a point; each data edge holds its point
# or nothing.
function fill_store!(rng, s, g)
    for ei in g.edges_of_factor[g.factor_index[:f]]
        r = rand(rng, 1:3)
        s.to_factor[ei] = r == 1 ? nothing : Mycelium.Message(r == 2 ? TrivialBelief() : DiracBelief(rand(rng)), 0)
    end
    for ei in eachindex(g.edges)
        g.edges[ei].factor == g.factor_index[:f] && continue
        s.to_variable[ei] = rand(rng, Bool) ? Mycelium.Message(DiracBelief(rand(rng)), 0) : nothing
    end
    return s
end

# The code `_step!` ran before the plans: an exception, `:trivial`, or the recorded call.
function reference_message(s, g, ei)
    e = g.edges[ei]; fn = factornode(g, e.factor)
    try
        chans = available_channels(s, g, e.factor; skip = ei)
        pol = Mycelium._resolve_polarity(g, e.factor, e.channel, chans)
        isempty(chans) && !issink(fn.factor) && !LenticulumCore.supports_polarity(fn.factor, pol) && return :trivial
        check_legal(g, e.factor, pol)
        inputs = Mycelium._inputs(s, g, e.factor, chans)
        prior = excluded_marginal(s, g, e.variable, ei)
        empty!(ECHO_CALLS)
        Mycelium._rule_message(fn.rule, fn.factor, e.channel, pol, inputs, prior, NamedTuple(), NamedTuple())
        return only(ECHO_CALLS)
    catch err
        err isa Mycelium.PolarityError || rethrow()
        return (err.factor, err.channel, err.reason)
    end
end

function planned_message(s, g, ei)
    empty!(ECHO_CALLS)
    try
        b, _, _ = step!(s, g, Mycelium.MessageTask(ei, :to_variable), NamedTuple(), NamedTuple(); commit = false)
        b isa TrivialBelief && return :trivial
        return only(ECHO_CALLS)
    catch err
        err isa Mycelium.PolarityError || rethrow()
        return (err.factor, err.channel, err.reason)
    end
end

@testset "planned messages agree with the unplanned path (I18)" begin
    rng = Xoshiro(18)
    # Each new set of inputs is a new `NamedTuple` type and compiles, so the wide factors get few trials.
    for n in (1, 2, 4, 64, 65, 70), trial in 1:(n <= 4 ? 12 : 2)
        g = echo_graph(rng, n; free = rand(rng, Bool), rule = rand(rng, (BeliefPropagation(), ExpectationPropagationRule())))
        s = MessageStore(g)
        fid = g.factor_index[:f]
        for _ in 1:(n <= 4 ? 4 : 2)
            fill_store!(rng, s, g)
            for ei in g.edges_of_factor[fid]
                can_emit(g, ei) || continue
                ref = reference_message(s, g, ei)
                @test planned_message(s, g, ei) == ref
                @test planned_message(s, g, ei) == ref   # from the stored plan
            end
        end
        if n <= 64
            # every stored plan is the one a fresh planning builds
            for ((ei, mask), plan) in g.emit_plans
                eofs = g.edges_of_factor[fid]
                chans = Tuple(g.edges[eofs[i]].channel for i in eachindex(eofs) if isodd(mask >> (i - 1)))
                fresh = Mycelium._make_plan(g, fid, ei, chans)
                @test typeof(plan) == typeof(fresh)
                @test plan === nothing || (plan.polarity == fresh.polarity && plan.edges == fresh.edges)
            end
        else
            @test isempty(g.emit_plans)          # past 64 edges nothing is stored
        end
    end
end

@testset "an illegal plan is not stored" begin
    # A message with no inputs that the factor does not support is planned as `nothing` and stored;
    # a message whose input sits on an edge that cannot absorb is refused by `check_legal` and not
    # stored, so it throws on every call.
    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y)
    factor!(b, :f, EchoFactor((:x, :y), false))
    connect!(b, :f, :x, :x; direction = Emitting())
    connect!(b, :f, :y, :y; direction = Emitting())
    g = build(b)
    s = MessageStore(g)
    ex, ey = g.edges_of_factor[1]
    # `x` has no incoming message (its edge only emits), so the message on `y` has no inputs and
    # this factor does not emit freely: the plan is `nothing`, a TrivialBelief, and it is stored.
    @test planned_message(s, g, ey) === :trivial
    @test haskey(g.emit_plans, (ey, UInt64(0))) && g.emit_plans[(ey, UInt64(0))] === nothing

    b = GraphBuilder()
    variable!(b, :x); variable!(b, :y)
    factor!(b, :f, EchoFactor((:x, :y), false))
    connect!(b, :f, :x, :x; direction = Emitting())
    connect!(b, :f, :y, :y; direction = Bidirectional())
    g = build(b)
    s = MessageStore(g)
    ex, ey = g.edges_of_factor[1]
    s.to_factor[ex] = Mycelium.Message(DiracBelief(1.0), 0)   # an input on an edge that cannot absorb
    for _ in 1:2
        @test planned_message(s, g, ey) isa Tuple{Symbol,Symbol,String}   # check_legal: x cannot be Observed
    end
    @test isempty(g.emit_plans)
end

@testset "plans under concurrent tasks and several stores (I18)" begin
    # Several tasks, each with its own store, compute messages on one graph at once: every result is
    # the old path's, and every stored plan is the one a fresh planning builds (review of I18, LP-027).
    rng = Xoshiro(27)
    g = echo_graph(rng, 6; free = true, rule = BeliefPropagation())
    fid = g.factor_index[:f]
    em = [ei for ei in g.edges_of_factor[fid] if can_emit(g, ei)]
    stores = [fill_store!(Xoshiro(k), MessageStore(g), g) for k in 1:8]
    refs = [[reference_message(s, g, ei) for ei in em] for s in stores]
    # `ECHO_CALLS` is shared, so the tasks compare the beliefs `step!` returns instead
    got = Vector{Any}(undef, length(stores))
    Threads.@sync for k in eachindex(stores)
        Threads.@spawn begin
            out = Any[]
            for _ in 1:20, ei in em
                b, _, _ = try
                    step!(stores[k], g, Mycelium.MessageTask(ei, :to_variable), NamedTuple(), NamedTuple(); commit = false)
                catch err
                    err isa Mycelium.PolarityError || rethrow()
                    (err, nothing, nothing)
                end
                push!(out, b)
            end
            got[k] = out
        end
    end
    for k in eachindex(stores)
        expected = repeat([r === :trivial ? TrivialBelief() : r isa Tuple{Symbol,Symbol,String} ? r : DiracBelief(1.0) for r in refs[k]], 20)
        actual = [b isa Mycelium.PolarityError ? (b.factor, b.channel, b.reason) : b for b in got[k]]
        @test actual == expected
    end
    for ((ei, mask), plan) in g.emit_plans
        g.edges[ei].factor == fid || continue
        eofs = g.edges_of_factor[fid]
        fresh = Mycelium._make_plan(g, fid, ei, Tuple(g.edges[eofs[i]].channel for i in eachindex(eofs) if isodd(mask >> (i - 1))))
        @test typeof(plan) == typeof(fresh)
        @test plan === nothing || (plan.polarity == fresh.polarity && plan.edges == fresh.edges)
    end
end
