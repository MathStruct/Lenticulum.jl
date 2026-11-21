module MyceliumLearningChainRulesCoreExt
using MyceliumLearning, Mycelium, LenticulumCore, ChainRulesCore
using Mycelium: FactorGraph, MessageTask, ResidualSchedule, infer!
using LenticulumCore: DiracBelief, TrivialBelief
using ChainRulesCore: backing
ChainRulesCore.@non_differentiable MyceliumLearning._frozen_beliefs(s, g, ps, st)
ChainRulesCore.@non_differentiable MyceliumLearning._check_unnormalised(g)

const ML = MyceliumLearning
const LA = ML.LinearAlgebra
_mycelium_rules() = Base.get_extension(Mycelium, :MyceliumChainRulesCoreExt)

function ChainRulesCore.rrule(::typeof(Mycelium._local_messages), s::ML._CoordinateStore, reads)
    return _mycelium_rules()._local_messages_rrule(s, reads, (;))
end
ChainRulesCore.@non_differentiable ML._task_reads_of(g, ts)
ChainRulesCore.@non_differentiable ML._task_nodes(g, ts)
ChainRulesCore.@non_differentiable ML._fixed_parts(ctx)
ChainRulesCore.@non_differentiable ML._marginal_reads(g, vs)
ChainRulesCore.@non_differentiable ML._parameter_nodes(g, names, supplied)
# Checks, no derivative. Traced by Zygote, the layout check generated a pullback per task at run
# time and was the larger part of the Krylov product's forward pass (I17; 45.6 of 139.8 MB at 241 factors).
ChainRulesCore.@non_differentiable ML._same_message_layout(a, b)
ChainRulesCore.@non_differentiable ML._check_pathwise(b)

_iszero(x) = x === nothing || x isa AbstractZero
_part(x, i) = _iszero(x) ? ZeroTangent() : unthunk(x[i])
function _part(x, name::Symbol)
    x = unthunk(x)
    _iszero(x) && return ZeroTangent()
    x = x isa Tangent ? backing(x) : x
    return unthunk(get(x, name, ZeroTangent()))
end

function _tree_cotangents(Δ, names)
    Δ = unthunk(Δ)
    _iszero(Δ) && return Vector{Any}(nothing, length(names))
    Δ = Δ isa Tangent ? backing(Δ) : Δ
    keys(Δ) === names && return Any[unthunk(getfield(Δ, i)) for i in 1:length(names)]
    positions = Mycelium._keypositions(Δ)
    return Any[haskey(positions, name) ? unthunk(getfield(Δ, positions[name])) : nothing for name in names]
end

function ChainRulesCore.rrule(::typeof(ML._named_tree), names, vals)
    y = ML._named_tree(names, vals)
    return y, Δ -> (NoTangent(), NoTangent(), _tree_cotangents(Δ, names))
end

function _store_messages(Δ, field)
    ds = _part(Δ, field)
    return ds isa AbstractVector ? ds : _part(ds, :msgs)
end

function _seed_versions!(dest, versions, Δ)
    _iszero(Δ) && return
    for (j, v) in enumerate(versions)
        v == 0 && continue
        dest[v] = _part(_part(Δ, j), :belief)
    end
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(ML._replay_run),
        mode, g, ts, params, st, tape, damping, ad, iteration)
    steps = Vector{Any}(undef, length(ts))
    function evaluate(i, fid, tv, tf, sv, save, args...)
        y, back = rrule_via_ad(config, ML._task_step, mode, args...)
        steps[i] = (; fid, tv, tf, sv, back, state=save ? y[2] : nothing, scored=y[3] !== nothing)
        return y
    end
    y, (tv, tf, live_state) = ML._replay_forward(evaluate, mode, g, ts, params, st, tape, damping, ad, iteration)
    # Use the I16 tangent addition. A pullback can omit fields of a message or subtree.
    rules = _mycelium_rules()
    return y, Δ -> _replay_pullback(rules, steps, params, st, mode, keys_or_empty(y[2]), tv, tf, live_state, unthunk(Δ))
end

keys_or_empty(st) = keys(st)
keys_or_empty(::Nothing) = ()

function _replay_pullback(rules, steps, params, @nospecialize(st), mode, names, tv, tf, live_state, Δ)
    nt, z = NoTangent(), ZeroTangent()
    _iszero(Δ) && return (nt, nt, nt, nt, z, z, nt, nt, nt, nt)
    dm = Vector{Any}(nothing, length(steps))
    ds = Vector{Any}(nothing, length(steps))
    dparams = params === nothing ? z : Vector{Any}(nothing, length(params))
    dinitial = Vector{Any}(nothing, length(keys_or_empty(st)))
    store = _part(Δ, 1)
    _seed_versions!(dm, tv, _store_messages(store, :to_variable))
    _seed_versions!(dm, tf, _store_messages(store, :to_factor))
    for (v, d) in zip(live_state, _tree_cotangents(_part(Δ, 2), names))
        if v > 0
            ds[v] = d
        elseif !(mode isa Val{:infer}) || isempty(steps)
            dinitial[-v] = d
        end
    end
    dscore = _part(Δ, 3)
    for i in reverse(eachindex(steps))
        step = steps[i]
        db = dm[i]
        dstate = ds[i]
        dc = step.scored ? dscore : z
        _iszero(db) && _iszero(dstate) && _iszero(dc) && continue
        update = _iszero(dstate) ? z : Tangent{typeof(step.state)}(; value=dstate)
        _, _, dlocal, _, _, dp, dst, _, _, _ = step.back((db, update, dc))
        dparams[step.fid] = rules._acc(dparams[step.fid], dp)
        if step.sv > 0
            ds[step.sv] = rules._acc(ds[step.sv], dst)
        elseif step.sv < 0
            dinitial[-step.sv] = rules._acc(dinitial[-step.sv], dst)
        end
        _scatter_versions!(rules, dm, step.tv, _part(dlocal, :to_variable))
        _scatter_versions!(rules, dm, step.tf, _part(dlocal, :to_factor))
    end
    dst = if all(_iszero, dinitial)
        z
    else
        fields = ML._named_tree(keys(st), Any[_iszero(d) ? z : d for d in dinitial])
        Tangent{typeof(st),typeof(fields)}(fields)
    end
    return (nt, nt, nt, nt, dparams, dst, nt, nt, nt, nt)
end

function _scatter_versions!(rules, dm, versions, slice)
    msgs = _part(slice, :msgs)
    _iszero(msgs) && return
    for (j, v) in enumerate(versions)
        v == 0 && continue
        dm[v] = rules._acc(dm[v], _part(_part(msgs, j), :belief))
    end
end

function ChainRulesCore.rrule(::typeof(ML._coordinate_store), x, ctx)
    s = ML._coordinate_store(x, ctx)
    function coordinate_pullback(Δ)
        dtv = _store_messages(Δ, :to_variable)
        dtf = _store_messages(Δ, :to_factor)
        (_iszero(dtv) && _iszero(dtf)) && return (NoTangent(), ZeroTangent(), NoTangent())
        dx = zero(x)
        for (i, t) in enumerate(ctx.ts)
            messages, tangents = t.kind === :to_factor ? (s.to_factor, dtf) : (s.to_variable, dtv)
            _iszero(tangents) && continue
            messages[t.edge].iteration == i || continue
            db = _part(_part(tangents, t.edge), :belief)
            _iszero(db) || _pack_cotangent!(view(dx, ctx.ranges[i]), ctx.beliefs[i], db)
        end
        return NoTangent(), dx, NoTangent()
    end
    return s, coordinate_pullback
end

function _pack_cotangent!(dx, b::GaussianBelief, db)
    n = length(b.η)
    _add_coordinates!(view(dx, 1:n), _part(db, :η))
    dΛ = _part(db, :Λ)
    _iszero(dΛ) && return
    if b.Λ isa LA.Diagonal
        dΛ = dΛ isa AbstractMatrix ? LA.diag(dΛ) : _part(dΛ, :diag)
    end
    _add_coordinates!(view(dx, n+1:length(dx)), dΛ)
    return
end
_pack_cotangent!(dx, ::DiracBelief, db) = _add_coordinates!(dx, _part(db, :value))
_pack_cotangent!(dx, ::TrivialBelief, db) = nothing
function _add_coordinates!(dx, d)
    _iszero(d) && return
    d = d isa Number ? (d,) : vec(d)
    for (i, v) in enumerate(d)
        _iszero(v) || (dx[i] += v)
    end
end

# Reverse mode through `infer!`: the realised schedule is replayed functionally and that replay is
# differentiated (`gradient_couplings.jl`).
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(infer!), g::FactorGraph, schedule, ps, st; kwargs...)
    chosen = MessageTask[]
    result = schedule isa ResidualSchedule ?
        infer!(g, schedule, ps, st; kwargs..., record=chosen) : infer!(g, schedule, ps, st; kwargs...)
    damping = get(kwargs, :damping, 0.0)
    ts = schedule isa ResidualSchedule ? chosen :
        MyceliumLearning._realised_tasks(g, schedule, result[2].sweeps, ps, st; damping)
    _, back = rrule_via_ad(config, MyceliumLearning._functional_infer, g, ts, ps, st, result[2], damping)
    function pullback(dy)
        d = back(dy)
        return NoTangent(), NoTangent(), NoTangent(), d[4], NoTangent()
    end
    return result, pullback
end
end
