# ---------------------------------------------------------------------------
# Executing a schedule.
#
# The one identity worth remembering:
#
#     a factor -> variable message IS an inversion c'_π
#
# The polarity is determined by which channel is the target; the prior π is the excluded
# marginal at the target variable; the observation y is the pooled incoming messages.
#
# See `passing.md` and `Messages are Inversions.md`.
# ---------------------------------------------------------------------------

"""
    factor_message(factor, target::Symbol, polarity, inputs::NamedTuple, prior, ps, st)
        -> (belief, st)

Compute the outgoing belief on channel `target`.

The generic implementation is `assemble(factor, polarity, ps, st)` followed by
`invert(lens, prior, inputs, ps, st)` — i.e. the composite of `LenticulumCore`'s two
interface functions. Structural factors (`DataFactor`, `LossFactor`, …) implement this
directly instead, because for them the "lens" is trivial and building one would be
ceremony.

`inputs` is a `NamedTuple` of the incoming beliefs keyed by channel, and `prior` is the
excluded marginal at the target variable — the ``\\pi`` of ``c'_\\pi``.
"""
function factor_message(factor, target::Symbol, polarity, inputs, prior, ps, st)
    lens, st = LenticulumCore.assemble(factor, polarity, ps, st)
    return LenticulumCore.invert(lens, prior, inputs, ps, st)
end

"""
    available_channels(store, g, fid; skip = 0) -> Tuple{Vararg{Symbol}}

Channels of factor `fid` that currently have an incoming (variable → factor) message,
optionally excluding one edge. These become `Observed()` in the resolved polarity.
"""
function available_channels(s::MessageStore, g::FactorGraph, fid::Int; skip::Int = 0)
    out = Symbol[]
    for ei in g.edges_of_factor[fid]
        ei == skip && continue
        has_to_factor(s, ei) && push!(out, g.edges[ei].channel)
    end
    return Tuple(out)
end

function _inputs(s::MessageStore, g::FactorGraph, fid::Int, chans)
    vals = map(chans) do c
        ei = findfirst(e -> g.edges[e].channel === c, g.edges_of_factor[fid])
        s.to_factor[g.edges_of_factor[fid][ei]].belief
    end
    return NamedTuple{chans}(vals)
end

"""
    step!(store, g, task, ps, st; damping = 0.0, commit = true) -> (belief, residual, st)

Execute one [`MessageTask`](@ref).

`ps` and `st` are the graph-wide parameter and state trees, keyed by factor name — the same
nested `NamedTuple` shape `LuxCore.setup` produces.

Returns the new belief, the [`belief_distance`](@ref) to the message it replaces (the
*residual*, used for convergence and for [`ResidualSchedule`](@ref)), and the updated state.
With `commit = false` the message is computed but not stored, which is what
[`FloodingSchedule`](@ref) needs for double buffering.
"""
function step!(s::MessageStore, g::FactorGraph, t::MessageTask, ps, st;
               damping::Real = 0.0, commit::Bool = true)
    e = g.edges[t.edge]
    if t.kind === :to_factor
        can_absorb(g, t.edge) || throw(PolarityError(
            factornode(g, e.factor).name, e.channel,
            "edge is Emitting; a variable -> factor message is not legal here"))
        belief = excluded_marginal(s, g, e.variable, t.edge)
        old = s.to_factor[t.edge]
        res = belief_distance(old === nothing ? nothing : old.belief, belief)
        if commit
            s.to_factor[t.edge] = Message(belief, s.iteration)
        end
        return belief, res, st
    else
        can_emit(g, t.edge) || throw(PolarityError(
            factornode(g, e.factor).name, e.channel,
            "edge is Absorbing; a factor -> variable message is not legal here"))
        fn = factornode(g, e.factor)
        # EXCLUSION, factor side: the target channel's own incoming message must not be
        # used to compute the outgoing message on that same channel. Without this the
        # target would be resolved as both Observed and Unobserved, and even if it were
        # not, the factor would be feeding its own previous claim back to itself.
        chans = available_channels(s, g, e.factor; skip = t.edge)
        pol = resolve_polarity(g, e.factor, e.channel, chans)
        check_legal(g, e.factor, pol)
        inputs = _inputs(s, g, e.factor, chans)
        prior = excluded_marginal(s, g, e.variable, t.edge)
        psf = _subtree(ps, fn.name)
        stf = _subtree(st, fn.name)
        belief, stf = factor_message(fn.factor, e.channel, pol, inputs, prior, psf, stf)
        old = s.to_variable[t.edge]
        if old !== nothing && damping > 0
            belief = damp(belief, old.belief, 1 - damping)
        end
        res = belief_distance(old === nothing ? nothing : old.belief, belief)
        if commit
            s.to_variable[t.edge] = Message(belief, s.iteration)
        end
        return belief, res, _setsubtree(st, fn.name, stf)
    end
end

_subtree(nt::NamedTuple, name::Symbol) = haskey(nt, name) ? getfield(nt, name) : NamedTuple()
_subtree(::Nothing, ::Symbol) = NamedTuple()
_setsubtree(nt::NamedTuple, name::Symbol, v) = merge(nt, NamedTuple{(name,)}((v,)))
_setsubtree(::Nothing, ::Symbol, v) = NamedTuple()

"""
    sweep!(store, g, sched, ps, st; damping = 0.0) -> (maxresidual, st)

One pass of the schedule.

`SequentialSchedule`, `TreeSchedule` and `ForwardBackwardSchedule` execute **in place**, so
later tasks see earlier results in the same sweep. `FloodingSchedule` is **double
buffered**: every task is computed against the previous sweep's messages and all results are
committed together.
"""
function sweep! end

function sweep!(s::MessageStore, g::FactorGraph, sched::AbstractSchedule, ps, st;
                damping::Real = 0.0)
    s.iteration += 1
    maxres = 0.0
    for t in tasks(sched)
        _, res, st = step!(s, g, t, ps, st; damping)
        maxres = max(maxres, res)
    end
    return maxres, st
end

function sweep!(s::MessageStore, g::FactorGraph, sched::FloodingSchedule, ps, st;
                damping::Real = 0.0)
    s.iteration += 1
    ts = tasks(sched)
    results = Vector{Any}(undef, length(ts))
    maxres = 0.0
    for (i, t) in enumerate(ts)
        b, res, st = step!(s, g, t, ps, st; damping, commit = false)
        results[i] = b
        maxres = max(maxres, res)
    end
    for (i, t) in enumerate(ts)
        if t.kind === :to_factor
            s.to_factor[t.edge] = Message(results[i], s.iteration)
        else
            s.to_variable[t.edge] = Message(results[i], s.iteration)
        end
    end
    return maxres, st
end

"""
    ConvergenceReport(converged, sweeps, residual, reason)

What [`propagate!`](@ref) reports. `converged == false` is not an error — on a loopy graph it
is the expected outcome, and callers must decide whether to trust the marginals.
"""
struct ConvergenceReport
    converged::Bool
    sweeps::Int
    residual::Float64
    reason::String
end
Base.show(io::IO, r::ConvergenceReport) = print(io,
    "ConvergenceReport(", r.converged ? "converged" : "NOT converged",
    " after ", r.sweeps, " sweep(s), residual ", r.residual, "; ", r.reason, ")")

"""
    propagate!(store, g, sched, ps, st; maxsweeps = 100, tol = 1e-8, damping = 0.0)
        -> (ConvergenceReport, st)

Run the schedule to convergence, or to `maxsweeps`.

For a [`TreeSchedule`](@ref) one sweep is enough and is **exact**; the loop stops after one
and says so. For loopy graphs there is no guarantee: see `Loopy Message Passing.md`.

`damping ∈ [0,1)` mixes each new message with the previous one
(`0` = none, higher = more inertia), the standard remedy for oscillation. Damping only
applies where [`can_damp`](@ref) is true; elsewhere it is silently a no-op, which is recorded
in the report's `reason`.
"""
function propagate!(s::MessageStore, g::FactorGraph, sched::AbstractSchedule, ps, st;
                    maxsweeps::Int = 100, tol::Real = 1e-8, damping::Real = 0.0)
    if sched isa TreeSchedule
        _, st = sweep!(s, g, sched, ps, st; damping)
        return ConvergenceReport(true, 1, 0.0, "tree schedule: one sweep is exact"), st
    end
    res = Inf
    for i in 1:maxsweeps
        res, st = sweep!(s, g, sched, ps, st; damping)
        if res <= tol
            return ConvergenceReport(true, i, float(res), "residual below tol"), st
        end
    end
    reason = isfinite(res) ? "hit maxsweeps" :
        "residual is Inf: no `belief_distance` method for these belief types, so convergence \
         cannot be verified (see messages.md). Ran to maxsweeps rather than claim success."
    return ConvergenceReport(false, maxsweeps, float(res), reason), st
end

"""
    infer!(g, sched, ps, st; kwargs...) -> (marginals::NamedTuple, report, st)

Convenience: fresh store, propagate, read out every variable's marginal.
"""
function infer!(g::FactorGraph, sched::AbstractSchedule, ps, st; kwargs...)
    s = MessageStore(g)
    report, st = propagate!(s, g, sched, ps, st; kwargs...)
    names = Tuple(v.name for v in g.variables)
    vals = Tuple(marginal(s, g, v.id) for v in g.variables)
    return NamedTuple{names}(vals), report, st, s
end
