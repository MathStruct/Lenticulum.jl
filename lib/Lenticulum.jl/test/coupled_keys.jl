module CoupledKeysTests

using Test, Random, LinearAlgebra
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LuxCore
using LenticulumCore: DiracBelief, TrivialBelief
using Mycelium: factornode, available_channels, excluded_marginal, messages_into,
    can_emit, can_absorb, issink, damp
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

const ML = MyceliumLearning
const LC = LenticulumCore

# The replay and graph-adjoint implementation before I17.
struct _ReplayStore <: Mycelium.AbstractMessageStore
    to_variable::Vector{Any}
    to_factor::Vector{Any}
    iteration::Int
end
_ReplayStore(g::FactorGraph) = _ReplayStore(Vector{Any}(nothing, nedges(g)), Vector{Any}(nothing, nedges(g)), 0)
Zygote.ChainRulesCore.@non_differentiable _ReplayStore(g::FactorGraph)
_buffer(s, buffer, t::MessageTask) = (s, s)
function _buffer(s, buffer, t::ML._ReplayTask)
    saved = t.first ? s : buffer
    return (t.buffered ? saved : s, saved)
end

const _task = ML._task
const _gradient_message = ML._gradient_message
const _subtree = ML._subtree
const _setsubtree = ML._setsubtree
const _variable_nodes = ML._variable_nodes
const _factor_nodes = ML._factor_nodes
const _local_scalar = ML._local_scalar
const _belief_scalar = ML._belief_scalar
const _score_baseline = ML._score_baseline
const _factor_parameters = ML._factor_parameters
const _key_node = ML._key_node
const _active_indices = ML._active_indices
const _replace_active = ML._replace_active
const _gather_active = ML._gather_active
const _restore_message = ML._restore_message
const _check_pathwise = ML._check_pathwise
const _same_message_layout = ML._same_message_layout
const _pack_message = ML._pack_message
const _inputs = ML._inputs
const _variable_term = ML._variable_term

function _old_candidate(s, g, t, ps, st, ad=nothing)
    e = g.edges[t.edge]
    if t.kind === :to_factor
        can_absorb(g, t.edge) || throw(ArgumentError("schedule sends to an emitting-only factor port"))
        return excluded_marginal(s, g, e.variable, t.edge), st
    end
    can_emit(g, t.edge) || throw(ArgumentError("schedule sends from an absorbing-only factor port"))
    f = factornode(g, e.factor)
    chans = available_channels(s, g, f.id; skip = t.edge)
    pol = resolve_polarity(g, f.id, e.channel, chans)
    if isempty(chans) && !issink(f.factor) && !LenticulumCore.supports_polarity(f.factor, pol)
        return LenticulumCore.TrivialBelief(), st
    end
    check_legal(g, f.id, pol)
    b, state = _gradient_message(ad, f.factor, e.channel, pol, _inputs(s, g, f.id, chans),
        excluded_marginal(s, g, e.variable, t.edge), _subtree(ps, f.params), _subtree(st, f.params))
    return b, _setsubtree(st, f.params, state)
end

function _old_damped_candidate(s, g, t, ps, st, damping, ad=nothing)
    b, state = _old_candidate(s, g, t, ps, st, ad)
    old = t.kind === :to_variable ? s.to_variable[t.edge] : nothing
    if old !== nothing && damping > 0
        b = damp(b, old.belief, 1 - damping)
    end
    return b, state
end

function _old_functional_infer(g, ts, ps, st, report, damping=0.0)
    final_s, final_state, _ = foldl(ts; init=(_ReplayStore(g), st, _ReplayStore(g))) do acc, op
        store, state, buffer = acc
        read, buffer = _buffer(store, buffer, op)
        t = _task(op)
        e = g.edges[t.edge]
        psnames = keys(ps)
        pinned = NamedTuple{psnames}(map(name -> _factor_parameters(_key_node(g,name),ps[name]), psnames))
        b, state = _old_damped_candidate(read, g, t, pinned, state, damping, Val(:chainrules))
        b = e.detach ? LenticulumCore.stop_gradient(b) : b
        return _old_with_message(store,t,b), LenticulumCore.stop_gradient(state), buffer
    end
    names = Tuple(v.name for v in _variable_nodes(g))
    final_store = MessageStore(copy(final_s.to_variable),copy(final_s.to_factor),report.sweeps)
    return NamedTuple{names}(map(v -> marginal(final_s,g,v.id), _variable_nodes(g))), report, final_state, final_store
end

function _old_with_message(s, t, belief)
    msg = Message(belief, s.iteration + 1)
    tv = t.kind === :to_variable ? [i == t.edge ? msg : m for (i,m) in enumerate(s.to_variable)] : s.to_variable
    tf = t.kind === :to_factor ? [i == t.edge ? msg : m for (i,m) in enumerate(s.to_factor)] : s.to_factor
    return _ReplayStore(tv, tf, s.iteration + 1)
end

function _old_graph_scalar(s, g, ps, st, frozen = nothing)
    terms = map(_factor_nodes(g)) do f
        b = frozen === nothing ? nothing : frozen[f.id]
        b === nothing ?
            _local_scalar(_subtree(ps, f.params), f.factor, messages_into(s, g, f.id), _subtree(st, f.params)) :
            _belief_scalar(_subtree(ps, f.params), f.factor, b, _subtree(st, f.params))
    end
    correction = sum(map(v -> _variable_term(s, g, v.id), _variable_nodes(g)))
    return sum(terms) + correction
end

function _old_replay(ps, g, ts, tape, st, ad, damping, frozen = nothing; outputs=false)
    final_s, final_state, correction, _ = foldl(zip(ts, tape); init=(_ReplayStore(g), st, 0.0, _ReplayStore(g))) do acc, item
        store, state, score, buffer = acc
        op, ref = item
        read, buffer = _buffer(store, buffer, op)
        t = _task(op)
        e = g.edges[t.edge]
        b, state = _old_damped_candidate(read, g, t, ps, state, damping, ad)
        if e.detach || e.coupling isa DiagonalCoupling
            b = ref.belief
        elseif e.coupling isa LenticulumCore.ScoreFunctionCoupling
            score += (ref.loss - _score_baseline(e)) * (belief_logdensity(b, ref.sample) - ref.logp)
            b = ref.belief
        end
        return _old_with_message(store, t, b), state, score, buffer
    end
    outputs && return final_s, final_state, correction
    return _old_graph_scalar(final_s, g, ps, final_state, frozen) + correction
end

function _old_pinned_parameters(ps, reference_ps, g)
    names = keys(reference_ps)
    pinned = NamedTuple{names}(map(names) do name
        f = _key_node(g, name)
        indices = _active_indices(f, reference_ps[name])
        θ = LenticulumCore._flatten_active(ps[name])
        _replace_active(reference_ps[name], indices, _gather_active(θ, indices))
    end)
    return pinned
end

function _old_coordinate_store(x, ctx)
    return foldl(eachindex(ctx.ts); init=_ReplayStore(ctx.g)) do s, i
        _old_with_message(s, ctx.ts[i], _restore_message(x[ctx.ranges[i]], ctx.beliefs[i]))
    end
end

function _old_graph_update(x, ps, ctx)
    s = _old_coordinate_store(x, ctx)
    pinned = _old_pinned_parameters(ps, ctx.ps, ctx.g)
    parts = map(eachindex(ctx.ts)) do i
        t = ctx.ts[i]
        e = ctx.g.edges[t.edge]
        if e.detach || e.coupling isa DiagonalCoupling
            return ctx.x[ctx.ranges[i]]
        end
        b, _ = _old_damped_candidate(s, ctx.g, t, pinned, ctx.st, ctx.damping, ctx.ad)
        e.coupling isa LenticulumCore.PathwiseCoupling && _check_pathwise(b)
        _same_message_layout(b, ctx.beliefs[i]) || throw(ArgumentError(
            "implicit BP update must preserve message family, shape and precision structure"))
        return _pack_message(b)
    end
    return reduce(vcat, parts)
end

const _draw = ML._draw

function _old_reference_tape(g, ts, ps, st, rng, damping=0.0)
    s = _ReplayStore(g)
    buffer = s
    tape = Any[]
    for op in ts
        read, buffer = _buffer(s, buffer, op)
        t = _task(op)
        b, st = _old_damped_candidate(read, g, t, ps, st, damping)
        e = g.edges[t.edge]
        if !e.detach && e.coupling isa LenticulumCore.PathwiseCoupling
            _check_pathwise(b)
        end
        if !e.detach && e.coupling isa LenticulumCore.ScoreFunctionCoupling
            e.direction isa Absorbing && t.kind === :to_factor || throw(ArgumentError(
                "score coupling currently requires an Absorbing edge: sampling in bidirectional BP changes its density/entropy objective"))
            all(ei -> !emits(g.edges[ei].direction), g.edges_of_factor[e.factor]) || throw(ArgumentError(
                "score sampling currently requires a terminal factor with only Absorbing ports"))
            ports = g.edges_of_factor[e.factor]
            count(ei -> g.edges[ei].variable == e.variable,ports) == 1 || throw(ArgumentError(
                "a sampled terminal factor cannot read the same variable on multiple ports without a joint sampling rule"))
            _score_baseline(e)
            x = _draw(rng, b)
            entry = (belief=LenticulumCore.DiracBelief(x), sample=x, logp=belief_logdensity(b, x))
        else
            entry = (belief=b, sample=nothing, logp=0.0)
        end
        push!(tape, entry)
        s = _old_with_message(s, t, entry.belief)
    end
    weighted = map(zip(ts,tape)) do (op,entry)
        t = _task(op)
        e = g.edges[t.edge]
        f = g.factors[e.factor]
        loss = !e.detach && e.coupling isa LenticulumCore.ScoreFunctionCoupling ?
            _local_scalar(_subtree(ps, f.params),f.factor,messages_into(s,g,f.id),_subtree(st, f.params)) : 0.0
        merge(entry,(loss=loss,))
    end
    return weighted, s, st
end

# The same fixture exercises shuffled parameters, absent parameterless keys,
# tied transitions and a mix of cut and live message paths.
function random_graph(seed; loopy, mixed, frozen, T=Float64)
    rng = Xoshiro(seed)
    b = GraphBuilder()
    for v in (:x, :y, :z)
        variable!(b, v, 1)
        f = Symbol(:prior_, v)
        factor!(b, f, GaussianPrior(v, T(randn(rng)), T(1 + rand(rng))))
        connect!(b, f, v, v; coupling=mixed && v === :z ? DiagonalCoupling() : ExactCoupling())
    end
    for (f, x, y) in ((:ab, :x, :y), (:bc, :y, :z), (:ca, :z, :x))
        f === :ca && !loopy && continue
        factor!(b, f, GaussianFactor(1 => 1; noise=T(0.8 + rand(rng)), channels=(:a, :b));
            params=:tied, freeze=frozen ? (:b,) : ())
        connect!(b, f, :a, x; coupling=mixed && f === :ab ? DiagonalCoupling() : ExactCoupling())
        connect!(b, f, :b, y; coupling=ExactCoupling(), detach=mixed && f === :bc)
    end
    g = build(b)
    tied = (A=fill(T(0.2 + 0.3rand(rng)), 1, 1), b=T.(randn(rng, 1)))
    ps = (prior_z=(;), tied, prior_x=(;))  # :prior_y is parameterless and absent
    st = (prior_z=(;), tied=(;), extra=(visit=4,))  # preserve extra state and append missing keys
    return g, ps, st
end

function check_store(new, old)
    @test new.iteration == old.iteration
    for field in (:to_factor, :to_variable)
        a, b = getproperty(new, field), getproperty(old, field)
        for i in eachindex(b)
            if b[i] === nothing
                @test a[i] === nothing
            else
                @test a[i].iteration == b[i].iteration
                check_belief(a[i].belief, b[i].belief)
            end
        end
    end
end

check_belief(a::GaussianBelief, b::GaussianBelief) =
    @test (typeof(a), a.η, a.Λ, a.improper) == (typeof(b), b.η, b.Λ, b.improper)
check_belief(a::DiracBelief, b::DiracBelief) = @test a.value == b.value
check_belief(a, b) = @test a == b

zgrad(f, p) = LC._flatten(LC._parameter_gradient(AutoZygote(), f, p))
fgrad(f, p) = LC._flatten(LC._parameter_gradient(AutoForwardDiff(), f, p))

@testset "I17 replay values and parameter cotangents" begin
    for seed in 1:4
        loopy = iseven(seed)
        g, ps, st = random_graph(seed; loopy, mixed=seed > 2, frozen=seed > 2, T=seed == 4 ? Float32 : Float64)
        schedules = (SequentialSchedule(all_tasks(g)), flooding_schedule(g), ResidualSchedule(maxtasks=length(all_tasks(g))))
        loopy || (schedules = (schedules..., tree_schedule(g)))
        for schedule in schedules, nsweeps in 1:3, damping in (0.0, 0.3)
            ts = ML._realised_tasks(g, schedule, nsweeps, ps, st; damping)
            ref, sr, str = ML._reference_tape(g, ts, ps, st, nothing, damping)
            oldref, so, sto = _old_reference_tape(g, ts, ps, st, nothing, damping)
            check_store(sr, so)
            @test str == sto
            frozen = ML._frozen_beliefs(sr, g, ps, str)
            ad = AutoZygote()
            old(p) = _old_replay(_old_pinned_parameters(p, ps, g), g, ts, oldref, st, ad, damping, frozen)
            new(p) = ML._coupled_loss(p, ps, g, ts, ref, st, ad, damping, frozen)
            # A partial sweep can leave both sides of a Gaussian relation
            # uninformative, so its joint belief is improper.
            expected = try
                old(ps)
            catch err
                err isa ArgumentError && occursin("belief_mean needs a proper belief", err.msg) || rethrow()
                err
            end
            if expected isa ArgumentError
                @test_throws ArgumentError new(ps)
            else
                @test new(ps) == expected
            end

            report = Mycelium.ConvergenceReport(false, nsweeps, Inf, "")
            a, _, sn, store = ML._functional_infer(g, ts, ps, st, report, damping)
            b, _, sx, oldstore = _old_functional_infer(g, ts, ps, st, report, damping)
            @test sn == sx
            check_store(store, oldstore)
            foreach(check_belief, values(a), values(b))

            # Sample each schedule for reverse mode while avoiding repeated compilation on Float32.
            # The old and new values are equal above; their derivatives are compared through
            # ForwardDiff on the new code, as Zygote through the old replay compiled for minutes.
            if nsweeps == 3 && damping == 0.3 && seed <= 3 && expected isa Real
                got = free_energy_gradient(MessageStore(g), g, ps, st; ad, schedule, nsweeps, damping)
                if seed <= 2
                    @test LC._flatten(got) ≈ fgrad(new, ps) rtol=1e-10 atol=1e-11
                else
                    # ForwardDiff fails on a frozen key, before I17 as well (`Float64(::Dual)` in
                    # `_rank_deficient`; Feature Backlog I29): reference differences instead.
                    fd = free_energy_gradient(MessageStore(g), g, ps, st; ad=nothing, schedule, nsweeps, damping)
                    @test LC._flatten(got) ≈ LC._flatten(fd) rtol=1e-5 atol=1e-7
                    @test_broken try
                        fgrad(new, ps); true
                    catch
                        false
                    end
                end
            end
        end
    end

    g, ps, _ = random_graph(17; loopy=false, mixed=false, frozen=true)
    shuffled = NamedTuple{reverse(keys(ps))}(reverse(values(ps)))
    @test ML._pinned_parameters(merge(shuffled, (extra=(x=1.0,),)), ps, g) ==
        _old_pinned_parameters(merge(shuffled, (extra=(x=1.0,),)), ps, g)
    @test_throws FieldError ML._pinned_parameters((prior_x=(;),), ps, g)
    @test_throws FieldError _old_pinned_parameters((prior_x=(;),), ps, g)
    extra = merge(ps, (unknown=(;),))
    @test_throws ArgumentError ML._pinned_parameters(extra, extra, g)
    @test_throws ArgumentError _old_pinned_parameters(extra, extra, g)
end

function marginal_loss(marginals)
    return sum(b -> sum(abs2, belief_mean(b)), values(marginals))
end

@testset "I17 reverse mode through infer!" begin
    for loopy in (false, true)
        g, ps, st = random_graph(21; loopy, mixed=false, frozen=false)
        for schedule in (loopy ? flooding_schedule(g) : tree_schedule(g), ResidualSchedule(maxtasks=length(all_tasks(g))))
            damping = 0.3
            _, report, _, _ = infer!(g, schedule, ps, st; maxsweeps=3, tol=0.0, damping)
            ts = ML._realised_tasks(g, schedule, report.sweeps, ps, st; damping)
            old(p) = marginal_loss(first(_old_functional_infer(g, ts, p, st, report, damping)))
            new(p) = marginal_loss(first(infer!(g, schedule, p, st; maxsweeps=3, tol=0.0, damping)))
            # A residual order is a discrete choice. ForwardDiff on the fixed replay uses the
            # same recorded choices as the pullback of infer!.
            @test zgrad(new, ps) ≈ fgrad(schedule isa ResidualSchedule ? old : new, ps) rtol=1e-10 atol=1e-11
        end
    end
end

struct StatefulSource <: AbstractLenticulumFactor end
LenticulumCore.channels(::StatefulSource) = (LenticulumCore.Channel(:x, nothing),)
LenticulumCore.supported_polarities(::StatefulSource) = (Polarity(x=Unobserved()),)
LenticulumCore.supports_polarity(f::StatefulSource, p::Polarity) = p in supported_polarities(f)
LenticulumCore.islearnable(::StatefulSource) = true
function Mycelium.factor_message(::StatefulSource, target::Symbol, pol, inputs, prior, ps, st)
    memory = get(st, :memory, zero(ps.μ)) + ps.μ
    return GaussianBelief([memory], ones(1, 1)), (; memory)
end
Mycelium.local_free_energy(::StatefulSource, beliefs, ps, st) = (get(st, :memory, zero(ps.μ))^2 / 2, st)

struct QuadraticSink <: AbstractLenticulumFactor end
LenticulumCore.channels(::QuadraticSink) = (LenticulumCore.Channel(:x, nothing),)
LenticulumCore.supported_polarities(::QuadraticSink) = ()
LenticulumCore.supports_polarity(::QuadraticSink, p::Polarity) = false
LenticulumCore.islearnable(::QuadraticSink) = false
Mycelium.uses_cavity(::QuadraticSink) = true
function Mycelium.local_free_energy(::QuadraticSink, beliefs, ps, st)
    x = beliefs.x
    x isa Union{Nothing,TrivialBelief} && return (0.0, st)
    value = x isa DiracBelief ? x.value : belief_mean(x)
    return (only(value) + 0.2)^2 / 2, st
end

function score_graph(c)
    b = GraphBuilder()
    for v in (:x, :y)
        variable!(b, v)
        factor!(b, v, StatefulSource(); params=:tied)
        connect!(b, v, :x, v; direction=Emitting(), coupling=ExactCoupling())
        loss = Symbol(:loss_, v)
        factor!(b, loss, QuadraticSink())
        connect!(b, loss, :x, v; direction=Absorbing(), coupling=c)
    end
    g = build(b)
    return g, (tied=(μ=0.73,),), (tied=(;),)
end

@testset "I17 state versions and seeded score replay" begin
    g, ps, st = score_graph(ExactCoupling())
    ts = ML._realised_tasks(g, flooding_schedule(g), 3, ps, st; damping=0.3)
    tape, _, _ = _old_reference_tape(g, ts, ps, st, nothing, 0.3)
    ad = AutoZygote()
    s, state, score = ML._replay_run(Val(:replay), g, ts, Mycelium._subtrees(ps, g.factors),
        st, tape, 0.3, ad, length(ts))
    so, sto, sco = _old_replay(ps, g, ts, tape, st, ad, 0.3; outputs=true)
    @test state == sto == (tied=(memory=foldl(+, fill(ps.tied.μ, 6)),),)
    @test score == sco
    check_store(s, so)
    new(p) = ML._replay(p, g, ts, tape, st, ad, 0.3)
    old(p) = _old_replay(p, g, ts, tape, st, ad, 0.3)
    @test new(ps) == old(ps)
    @test zgrad(new, ps) ≈ fgrad(old, ps) rtol=1e-10

    # A repeat within a buffered sweep keeps reading the sweep's initial
    # messages. A literal nothing state discards the first update, as before.
    tasks = all_tasks(g)
    ts = ML._realised_tasks(g, FloodingSchedule([tasks; first(tasks)]), 2, ps, nothing; damping=0.3)
    _, s, state = ML._reference_tape(g, ts, ps, nothing, nothing, 0.3)
    _, so, sto = _old_reference_tape(g, ts, ps, nothing, nothing, 0.3)
    check_store(s, so)
    @test state == sto

    g, ps, st = score_graph(ScoreFunctionCoupling(baseline=0.37))
    sched = forward_backward_schedule(g)
    ts = ML._realised_tasks(g, sched, 1, ps, st)
    rng, nsamples = Xoshiro(912), 5
    estimate = free_energy_gradient(MessageStore(g), g, ps, st; ad, schedule=sched, rng, nsamples)
    rng = Xoshiro(912)
    draws = map(1:nsamples) do _
        tape, s, sr = _old_reference_tape(g, ts, ps, st, rng)
        frozen = ML._frozen_beliefs(s, g, ps, sr)
        fgrad(p -> _old_replay(_old_pinned_parameters(p, ps, g), g, ts, tape, st, ad, 0.0, frozen), ps)
    end
    μ = sum(draws) / nsamples
    se = sqrt.(sum(v -> abs2.(v - μ), draws) / (nsamples * (nsamples - 1)))
    @test LC._flatten(estimate.value) ≈ μ rtol=1e-10
    @test LC._flatten(estimate.standard_error) ≈ se rtol=1e-10
end

function coordinate_context(g, ps, st, schedule; damping=0.0)
    s = MessageStore(g)
    for _ in 1:10
        _, st = sweep!(s, g, schedule, ps, st; damping)
    end
    ts = all_tasks(g)
    beliefs = map(t -> ML._stored_belief(s, t), ts)
    parts = map(ML._message_coordinates, beliefs)
    lengths = length.(parts)
    starts = cumsum(vcat(1, lengths[1:end-1]))
    ranges = map((i, n) -> i:i+n-1, starts, lengths)
    x = reduce(vcat, parts)
    frozen = ML._frozen_beliefs(s, g, ps, st)
    return (; g, ps, st, ts, beliefs, ranges, x, damping, frozen, ad=AutoZygote())
end

old_loss(x, p, ctx) = _old_graph_scalar(_old_coordinate_store(x, ctx), ctx.g,
    _old_pinned_parameters(p, ctx.ps, ctx.g), ctx.st, ctx.frozen)

function old_dense(ctx)
    F(x) = old_loss(x, ctx.ps, ctx)
    update(x) = _old_graph_update(x, ctx.ps, ctx)
    rhs = ForwardDiff.gradient(F, ctx.x)
    J = ForwardDiff.jacobian(update, ctx.x)
    λ = (I - J') \ rhs
    return fgrad(p -> old_loss(ctx.x, p, ctx) + dot(λ, _old_graph_update(ctx.x, p, ctx)), ctx.ps)
end

@testset "I17 graph-adjoint scalar and Krylov product" begin
    for loopy in (false, true)
        g, ps, st = random_graph(31; loopy, mixed=loopy, frozen=false)
        schedule = loopy ? flooding_schedule(g) : tree_schedule(g)
        st = merge(st, (extra=(;),))
        ctx = coordinate_context(g, ps, st, schedule; damping=loopy ? 0.3 : 0.0)
        x = ctx.x
        check_store(ML._coordinate_store(x, ctx), _old_coordinate_store(x, ctx))
        @test ML._graph_update(x, ps, ctx) == _old_graph_update(x, ps, ctx)
        v = randn(Xoshiro(31), length(x))
        for (f, f0) in (((x, p) -> ML._graph_map_dot(x, p, ctx, v), (x, p) -> dot(v, _old_graph_update(x, p, ctx))),
                        ((x, p) -> ML._graph_loss(x, p, ctx), (x, p) -> old_loss(x, p, ctx)))
            @test f(x, ps) == f0(x, ps)
            dx, _ = Zygote.gradient(f, x, ps)
            dp = zgrad(p -> f(x, p), ps)
            @test dx ≈ ForwardDiff.gradient(x -> f(x, ps), x) rtol=1e-10 atol=1e-11
            @test dp ≈ fgrad(p -> f(x, p), ps) rtol=1e-10 atol=1e-11
        end
        if !loopy
            # Krylov is exercised by MyceliumLearning's suite. For Lenticulum
            # we also check the product directly and compare the dense solves.
            grad(ad) = LC._flatten(free_energy_gradient(MessageStore(g), g, ps, st;
                schedule, ad, adjoint=:implicit, linear_solver=:dense))
            @test grad(AutoZygote()) ≈ old_dense(ctx) rtol=1e-10 atol=1e-11
            @test grad(AutoZygote()) ≈ grad(AutoForwardDiff()) rtol=1e-10 atol=1e-11
        end
    end
end

@testset "I17 coordinate pullback layouts" begin
    g, _, _ = random_graph(42; loopy=false, mixed=false, frozen=false)
    beliefs = [
        GaussianBelief([1., 2.], [3. 0.25; 0.25 5.]),
        GaussianBelief([2., 3.], Diagonal([4., 5.])),
        DiracBelief(0.4), DiracBelief([0.3 0.2; 0.7 0.8]), TrivialBelief(),
    ]
    ts = [MessageTask(i, :to_variable) for i in eachindex(beliefs)]
    parts = ML._message_coordinates.(beliefs)
    lengths = length.(parts)
    starts = cumsum(vcat(1, lengths[1:end-1]))
    ranges = map((i, n) -> i:i+n-1, starts, lengths)
    ctx = (; g, ts, beliefs, ranges)
    function loss(s; dense)
        b1, b2, b3, b4 = map(i -> s.to_variable[i].belief, 1:4)
        d = dense ? sum(Matrix(b2.Λ) .* [2. 7.; 11. 3.]) : sum(abs2, b2.Λ.diag)
        return sum(abs2, b1.η) + sum(b1.Λ .* [2. 5.; 7. 3.]) + d + b3.value^2 + sum(abs2, b4.value)
    end
    for T in (Float64, Float32), dense in (true, false)
        x = T.(reduce(vcat, parts))
        new(x) = loss(ML._coordinate_store(x, ctx); dense)
        old(x) = loss(_old_coordinate_store(x, ctx); dense)
        @test new(x) == old(x)
        dx = only(Zygote.gradient(new, x))
        @test dx == only(Zygote.gradient(old, x))
        @test dx ≈ ForwardDiff.gradient(new, x)
    end
    # Drive the rule with a dense precision cotangent as well as with Zygote's
    # structured one from the two matrix expressions above.
    CRC = Zygote.ChainRulesCore
    x = reduce(vcat, parts)
    s, back = CRC.rrule(ML._coordinate_store, x, ctx)
    messages = Any[nothing for _ in 1:nedges(g)]
    messages[2] = (; belief=(; Λ=[2. 7.; 11. 3.]))
    _, dx, _ = back(CRC.Tangent{typeof(s)}(; to_variable=messages))
    @test dx[ranges[2]] == [0., 0., 2., 3.]
    @test count(!iszero, dx) == 2
end

# Use the benchmark's position–velocity model for the allocation regression.
include(joinpath(@__DIR__, "../../../bench/kalman_chain.jl"))

function replay_allocations(T)
    g, ps, st = chain(T; coupling=ExactCoupling())
    schedule = tree_schedule(g)
    ad = AutoZygote()
    ctx = coordinate_context(g, ps, st, schedule)
    seed = ones(length(ctx.x))
    ts = ML._realised_tasks(g, schedule, 1, ps, st)
    tape, _, _ = ML._reference_tape(g, ts, ps, st, nothing)
    unrolled() = Zygote.gradient(p -> ML._coupled_loss(p, ps, g, ts, tape, st, ad, 0.0), ps)
    product() = Zygote.gradient((x, p) -> ML._graph_map_dot(x, p, ctx, seed), ctx.x, ps)
    unrolled()
    product()
    return ((@allocated unrolled()), (@allocated product()))
end

@testset "I17 replay and Krylov allocation growth" begin
    # At 10 and 40 steps (`bench/coupled_scaling.jl`, 2026-11-21): 4.5× (unrolled) and 4.1× (product).
    # Before I17 the unrolled gradient grew 8.9×; the product grew 5.9× in the first version, whose
    # by-key rule read `ps` specialised and whose update captured `ctx` (`gradient_couplings.md`). At 6
    # and 24 steps with a bound of 8 that version passed: the quadratic part only shows at this size.
    T = 10
    small, large = replay_allocations(T), replay_allocations(4T)
    @info "I17 allocations" steps=(T, 4T) unrolled=(small[1], large[1]) product=(small[2], large[2]) ratio=large ./ small
    @test large[1] < 5small[1]
    @test large[2] < 5small[2]
end

end
