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
    return LenticulumCore.likelihood_message(lens, prior, inputs, ps, st)
end

"""
    factor_messages(factors, target, polarity, inputs, priors, ps, st) -> (beliefs, st)

The factor → variable messages to channel `target` of several tied replicas at once: `factors[i]`
is replica `i`'s factor (all of one type, with shared parameters `ps`; their other fields, such
as data, may differ), and `inputs[i]` and `priors[i]` are what [`factor_message`](@ref) receives
for it. `beliefs[i]` must be what `factor_message` returns. A plate sweep
([`PlateSchedule`](@ref)) calls this for a group of tied replicas. The default calls
`factor_message` for each in turn, threading `st`; a factor type overrides it (dispatching on
the element type of `factors`) to share the work that depends only on the parameters.
"""
function factor_messages(factors::AbstractVector, target, polarity, inputs::AbstractVector, priors::AbstractVector, ps, st)
    beliefs = Vector{Any}(undef, length(inputs))
    for i in eachindex(factors, inputs, priors)
        beliefs[i], st = factor_message(factors[i], target, polarity, inputs[i], priors[i], ps, st)
    end
    return beliefs, st
end

# Block inputs (a plate sweep on message blocks, `blocks.jl`): `inputs` is a NamedTuple of
# `BeliefBlock`s and the result one block. The default takes each replica's column and calls the
# method above, so a factor type opts in to block arithmetic by adding a method of this form.
function factor_messages(factors::AbstractVector, target, polarity, inputs::NamedTuple, priors::AbstractVector, ps, st)
    cols = [NamedTuple{keys(inputs)}(map(b -> replica(b, i), values(inputs))) for i in eachindex(factors)]
    beliefs, st = factor_messages(factors, target, polarity, cols, priors, ps, st)
    return belief_block(beliefs), st
end

# The factor -> variable message of a cavity-reading node under its rule (`passing.md`). An
# expectation-propagation node returns `nothing` for an update it skips.
_rule_message(::AbstractMessageRule, factor, target, pol, inputs, prior, ps, st) =
    factor_message(factor, target, pol, inputs, prior, ps, st)

function _rule_message(::ExpectationPropagationRule, factor, target, pol, inputs, prior, ps, st)
    any(_indefinite, values(inputs)) && return nothing, st
    m, st = factor_message(factor, target, pol, inputs, prior, ps, st)
    return ep_message(m, prior), st
end

function _rule_message(r::MixtureReduction, factor, target, pol, inputs, prior, ps, st)
    m, st = any(b -> b isa LenticulumCore.MixtureBelief, values(inputs)) ?
        mixture_message(factor, target, pol, inputs, prior, ps, st) :
        factor_message(factor, target, pol, inputs, prior, ps, st)
    return reduce_message(m, r.maxcomponents), st
end

# A cavity with a negative direction, which an expectation-propagation site can leave behind;
# a factor cannot take it as an input. `gaussian.jl` has the Gaussian method.
_indefinite(b) = false

"""
    ep_message(message, cavity) -> belief or nothing

The [`ExpectationPropagationRule`](@ref) message from the belief-propagation `message` to a target
whose cavity (excluded marginal) is `cavity`. A message that is not a `MixtureBelief` is in its
family and is returned as it is. A mixture is multiplied by the cavity (the tilted belief),
projected ([`ep_project`](@ref)) and divided by the cavity ([`ep_divide`](@ref)); an
uninformative cavity leaves the mixture itself as the tilted belief, and a point cavity (a
pinned variable) makes the message irrelevant, so it is returned unchanged. Returns `nothing`
when the update cannot be made: a Gaussian cavity that is not proper, or a tilted belief that
[`ep_project`](@ref) cannot normalise.
"""
ep_message(m, cavity) = m
function ep_message(m::LenticulumCore.MixtureBelief, cavity)
    cavity isa LenticulumCore.DiracBelief && return m
    if LenticulumCore.isuninformative(cavity)
        tilted = m
    else
        cavity isa GaussianBelief && !LenticulumCore.isproper(cavity) && return nothing
        tilted = combine(cavity, m)
    end
    p = ep_project(tilted)
    p === nothing && return nothing
    return ep_divide(p, cavity)
end

"""
    ep_project(tilted) -> belief or nothing

The projection step of expectation propagation: the member of the target's family closest to
`tilted` in ``\\mathrm{KL}(\\text{tilted} \\,\\|\\, q)``, i.e. the one with the same moments. Mycelium
implements it for a mixture of Gaussians (the total mean and covariance), returning `nothing`
when a component is not proper, and for a Gaussian (itself). A belief type opts in by adding a
method.
"""
ep_project(b::GaussianBelief) = b
ep_project(b) = throw(ArgumentError(
    "no expectation-propagation projection for $(typeof(b)): implement `Mycelium.ep_project` for it"))

"""
    ep_divide(p, cavity) -> belief

``p / \\pi``: the site that, combined with the cavity, gives `p`. Gaussians subtract canonical
parameters; the result may be improper, even indefinite, and is declared improper unless its
precision is positive definite. Dividing by an uninformative belief returns `p`.
"""
function ep_divide(p, cavity)
    LenticulumCore.isuninformative(cavity) && return p
    throw(ArgumentError("no expectation-propagation division of $(typeof(p)) by $(typeof(cavity))"))
end

"""
    mixture_message(factor, target::Symbol, polarity, inputs::NamedTuple, prior, ps, st) -> (belief, st)

The factor → variable message when some `inputs` are `MixtureBelief`s, computed exactly: each
mixture input is expanded into its components, the message of every combination of components
is weighted by the components' weights and by the evidence the combination gives, and the result
is one mixture. [`factor_message`](@ref) collapses such inputs to their moments instead.
[`MixtureReduction`](@ref) nodes call it. No generic implementation, since the evidence needs the
factor's form; Lenticulum implements it for `LinearConstraintFactor` and
`SwitchingConstraintFactor`.
"""
function mixture_message(factor, target, polarity, inputs, prior, ps, st)
    throw(ArgumentError(
        "$(nameof(typeof(factor))) has no `Mycelium.mixture_message`, so a MixtureReduction() node cannot take \
         its mixture inputs exactly; implement it, or use BeliefPropagation(), which collapses them"))
end

"""
    reduce_message(belief, maxcomponents) -> belief

A message reduced to at most `maxcomponents` mixture components, for [`MixtureReduction`](@ref) nodes. A
belief that is not a mixture, or has few enough components, is returned unchanged; a reduction
to one component returns the component. Lenticulum implements it for mixtures of Gaussians
(Runnalls' merge, `reduce_mixture`); anything else is refused.
"""
reduce_message(b, maxcomponents::Integer) = b
function reduce_message(m::LenticulumCore.MixtureBelief, maxcomponents::Integer)
    length(m.components) <= maxcomponents && return m
    throw(ArgumentError(
        "no mixture reduction for $(typeof(m)): implement `Mycelium.reduce_message` for it"))
end

"""
    mean_field_message(factor, target::Symbol, polarity, beliefs::NamedTuple, ps, st)
        -> (belief, st)

The outgoing message of a [`MeanField`](@ref) node on channel `target`:
``\\exp \\mathbb E_{\\prod_{i \\ne t} q_i}[\\log f(x)]`` as a function of ``x_t``, where `beliefs` holds the
current marginals ``q_i`` of the other channels' variables (not their cavity messages).

There is no generic implementation, because the expectation needs the factor's form; factors
opt in by adding a method. Lenticulum implements it for `LinearConstraintFactor` and
`SwitchingConstraintFactor`.
"""
function mean_field_message(factor, target, polarity, beliefs, ps, st)
    throw(ArgumentError(
        "$(nameof(typeof(factor))) has no mean-field message: implement `Mycelium.mean_field_message` \
         for it, or leave its node on BeliefPropagation()"))
end

"""
    mean_field_energy(factor, marginals::NamedTuple, ps, st) -> (𝐅, st)

A [`MeanField`](@ref) node's term in the graph free energy: the average energy
``\\mathbb E_{\\prod_i q_i}[-\\log f]`` under the product of its variables' current marginals
`marginals` (keyed by channel), graded like [`local_free_energy`](@ref), with **no entropy of its
own** (Riegler et al. 2013, eq. 17; the variables' entropies are counted at the variables,
[`counting_number`](@ref)). Its parameter derivative at fixed marginals is the node's part of the
graph gradient.

No generic implementation; a factor with a [`mean_field_message`](@ref) should add one, and
Lenticulum does for `LinearConstraintFactor` and `SwitchingConstraintFactor`.
"""
function mean_field_energy(factor, marginals, ps, st)
    throw(ArgumentError(
        "$(nameof(typeof(factor))) has no `Mycelium.mean_field_energy`, so a graph with it on a \
         MeanField() node has no free energy; implement it, or leave the node on BeliefPropagation()"))
end

# The other channels' current marginals, for a mean-field node: informative ones only, so that
# the rest resolve to Latent() exactly as uninformative cavities do for belief propagation.
function _marginal_inputs(s::AbstractMessageStore, g::FactorGraph, fid::Int, skip::Int)
    chans, vals = Symbol[], Any[]
    for ei in g.edges_of_factor[fid]
        ei == skip && continue
        b = marginal(s, g, g.edges[ei].variable)
        LenticulumCore.isuninformative(b) && continue
        push!(chans, g.edges[ei].channel); push!(vals, b)
    end
    return NamedTuple{Tuple(chans)}(Tuple(vals))
end

"""
    available_channels(store, g, fid; skip = 0) -> Tuple{Vararg{Symbol}}

Channels of factor `fid` that currently have an informative incoming (variable → factor)
message, optionally excluding one edge. Channels holding `nothing` or an uninformative belief
(`TrivialBelief()` or `uninformative(n)`) are excluded so they resolve to `Latent()`.
"""
function available_channels(s::AbstractMessageStore, g::FactorGraph, fid::Int; skip::Int = 0)
    out = Symbol[]
    for ei in g.edges_of_factor[fid]
        ei == skip && continue
        if has_to_factor(s, ei) && !LenticulumCore.isuninformative(s.to_factor[ei].belief)
            push!(out, g.edges[ei].channel)
        end
    end
    return Tuple(out)
end

function _inputs(s::AbstractMessageStore, g::FactorGraph, fid::Int, chans)
    vals = map(chans) do c
        ei = findfirst(e -> g.edges[e].channel === c, g.edges_of_factor[fid])
        s.to_factor[g.edges_of_factor[fid][ei]].belief
    end
    return NamedTuple{chans}(vals)
end

"""
    step!(store, g, task, ps, st; damping = 0.0, commit = true) -> (belief, residual, st)

Execute one [`MessageTask`](@ref).

`ps` and `st` are the graph-wide parameter and state trees, keyed by [`parameter_key`](@ref) (the factor name unless tied) — the same
nested `NamedTuple` shape `LuxCore.setup` produces.

Returns the new belief, the [`belief_distance`](@ref) to the message it replaces (the
*residual*, used for convergence and for [`ResidualSchedule`](@ref)), and the updated state.
With `commit = false` the message is computed but not stored, which is what
[`FloodingSchedule`](@ref) needs for double buffering. A skipped update (an
expectation-propagation node without a proper cavity) returns the edge's current message
(`TrivialBelief()` if it has none) and stores nothing, with either `commit`.
"""
step!(s::MessageStore, g::FactorGraph, t::MessageTask, @nospecialize(ps), @nospecialize(st); kwargs...) = Base.front(_step!(s, g, t, ps, st; kwargs...))

# `step!` and whether the update was skipped: the schedules that compute without committing
# (flooding, residual) must not store or select a skipped update.
#
# `ps` and `st` are the graph-wide trees, read only through `_subtree` by the node's key. They are
# not specialised on: their type changes with every new graph structure (a key per node), and
# inferring this body for a tree of a few hundred keys took 6.9 of the 9.1 s a new structure
# spent compiling its sweeps (`passing.md`, "Compilation per structure"). The factor's methods
# still specialise on its own subtree.
function _step!(s::MessageStore, g::FactorGraph, t::MessageTask, @nospecialize(ps), @nospecialize(st);
               damping::Real = 0.0, commit::Bool = true)
    0 <= damping < 1 || throw(ArgumentError("damping must be in [0, 1); got $damping"))
    e = g.edges[t.edge]
    if t.kind === :to_factor
        can_absorb(g, t.edge) || throw(PolarityError(
            factornode(g, e.factor).name, e.channel,
            "edge is Emitting; a variable -> factor message is not legal here"))
        belief = excluded_marginal(s, g, e.variable, t.edge)
        belief = e.detach ? LenticulumCore.stop_gradient(belief) : belief
        old = s.to_factor[t.edge]
        res = belief_distance(old === nothing ? nothing : old.belief, belief)
        if commit
            s.to_factor[t.edge] = Message(belief, s.iteration)
        end
        return belief, res, st, false
    else
        can_emit(g, t.edge) || throw(PolarityError(
            factornode(g, e.factor).name, e.channel,
            "edge is Absorbing; a factor -> variable message is not legal here"))
        fn = factornode(g, e.factor)
        psf = _factor_parameters(fn, _subtree(ps, fn.params))
        stf = _subtree(st, fn.params)
        # EXCLUSION, factor side: the target channel's own incoming message must not be
        # used to compute the outgoing message on that same channel. Without this the
        # target would be resolved as both Observed and Unobserved, and even if it were
        # not, the factor would be feeding its own previous claim back to itself.
        # A mean-field node reads the other variables' full marginals instead (`passing.md`).
        mf = fn.rule isa MeanField
        inputs = mf ? _marginal_inputs(s, g, e.factor, t.edge) : nothing
        chans = mf ? keys(inputs) : available_channels(s, g, e.factor; skip = t.edge)
        pol = resolve_polarity(g, e.factor, e.channel, chans)
        if mf
            check_legal(g, e.factor, pol)
            belief, stf = mean_field_message(fn.factor, e.channel, pol, inputs, psf, stf)
        elseif isempty(chans) && !issink(fn.factor) && !LenticulumCore.supports_polarity(fn.factor, pol)
            @warn "factor :$(fn.name) has no informative inputs when emitting on :$(e.channel); sending TrivialBelief()" maxlog = 1
            belief = LenticulumCore.TrivialBelief()
        else
            check_legal(g, e.factor, pol)
            inputs = _inputs(s, g, e.factor, chans)
            prior = excluded_marginal(s, g, e.variable, t.edge)
            belief, stf = _rule_message(fn.rule, fn.factor, e.channel, pol, inputs, prior, psf, stf)
        end
        old = s.to_variable[t.edge]
        if belief === nothing
            # An expectation-propagation update without a proper cavity is skipped: the edge keeps
            # its message (none yet: an uninformative one, not stored).
            @warn "factor :$(fn.name) skipped an expectation-propagation update on :$(e.channel): the cavity \
                   or the tilted belief is not a proper Gaussian; the edge keeps its previous message" maxlog = 1
            return (old === nothing ? LenticulumCore.TrivialBelief() : old.belief), (old === nothing ? Inf : 0.0),
                   _setsubtree(st, fn.params, stf), true
        end
        if old !== nothing && damping > 0
            belief = damp(belief, old.belief, 1 - damping)
        end
        belief = e.detach ? LenticulumCore.stop_gradient(belief) : belief
        res = belief_distance(old === nothing ? nothing : old.belief, belief)
        if commit
            s.to_variable[t.edge] = Message(belief, s.iteration)
        end
        return belief, res, _setsubtree(st, fn.params, stf), false
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

function sweep!(s::MessageStore, g::FactorGraph, sched::AbstractSchedule, @nospecialize(ps), @nospecialize(st);
                damping::Real = 0.0)
    s.iteration += 1
    maxres = 0.0
    for t in tasks(sched)
        _, res, st = step!(s, g, t, ps, st; damping)
        maxres = max(maxres, res)
    end
    return maxres, st
end

function sweep!(s::MessageStore, g::FactorGraph, sched::FloodingSchedule, @nospecialize(ps), @nospecialize(st);
                damping::Real = 0.0)
    s.iteration += 1
    ts = tasks(sched)
    results = Vector{Any}(undef, length(ts))
    skipped = falses(length(ts))
    maxres = 0.0
    for (i, t) in enumerate(ts)
        b, res, st, skipped[i] = _step!(s, g, t, ps, st; damping, commit = false)
        results[i] = b
        maxres = max(maxres, res)
    end
    for (i, t) in enumerate(ts)
        skipped[i] && continue              # a skipped update keeps the edge as it is (no message stays none)
        if t.kind === :to_factor
            s.to_factor[t.edge] = Message(results[i], s.iteration)
        else
            s.to_variable[t.edge] = Message(results[i], s.iteration)
        end
    end
    return maxres, st
end

function _task_age(s::MessageStore, t::MessageTask)
    m = t.kind === :to_factor ? s.to_factor[t.edge] : s.to_variable[t.edge]
    return m === nothing ? -1 : m.iteration
end

function sweep!(s::MessageStore, g::FactorGraph, sched::ResidualSchedule, @nospecialize(ps), @nospecialize(st);
                damping::Real = 0.0, record = nothing)
    s.iteration += 1
    ts = all_tasks(g)
    isempty(ts) && return 0.0, st
    nsteps = min(sched.maxtasks, length(ts))
    maxres = 0.0
    for _ in 1:nsteps
        best_idx = 0
        best_res = -1.0
        best_age = typemax(Int)
        for (i, t) in enumerate(ts)
            _, res, _, skip = _step!(s, g, t, ps, st; damping, commit = false)
            skip && continue                # cannot be committed: never the best task
            age = _task_age(s, t)
            if res > best_res || (res == best_res && age < best_age)
                best_idx = i
                best_res = res
                best_age = age
            end
        end
        best_idx == 0 && break
        record === nothing || push!(record, ts[best_idx])
        _, res, st = step!(s, g, ts[best_idx], ps, st; damping, commit = true)
        maxres = max(maxres, res)
        best_res == 0.0 && break
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

# One tree sweep is a fixed point when every message depends on upstream messages only: belief
# propagation and MixtureReduction() nodes. Expectation-propagation and mean-field messages also read the
# target side (the cavity, the marginals), so those graphs are iterated like any other.
_tree_exact(g::FactorGraph) = all(f -> f.rule isa BeliefPropagation || f.rule isa MixtureReduction, g.factors)

"""
    propagate!(store, g, sched, ps, st; maxsweeps = 100, tol = 1e-8, damping = 0.0,
               criterion = _default_criterion(sched))
        -> (ConvergenceReport, st)

Run the schedule to convergence, or to `maxsweeps`.

For a [`TreeSchedule`](@ref) without damping, on a graph of [`BeliefPropagation`](@ref) and
[`MixtureReduction`](@ref) nodes, one sweep is enough and is **exact** (for `MixtureReduction`, exact for the reduced
messages); the loop stops after one and says so. [`ExpectationPropagationRule`](@ref) and
[`MeanField`](@ref) messages also depend on the target's side, so such graphs are iterated. With damping a tree schedule is iterated and checked like any
other, since a damped sweep keeps part of each previously stored message. For loopy graphs there is no guarantee: see `Loopy Message Passing.md`.

`damping ∈ [0,1)` mixes each new message with the previous one
(`0` = none, higher = more inertia), the standard remedy for oscillation. Damping only
applies where [`can_damp`](@ref) is true; elsewhere it is a no-op. Callers can inspect
`can_damp`; the convergence report does not automatically record unsupported damping.

`criterion` is what must change by at most `tol` between sweeps: `:messages` (the largest
[`belief_distance`](@ref) of a sweep's messages to the ones they replace) or `:marginals` (the
largest change of a variable's marginal, compared by moments: means and covariances, and the
weights of a mixture or a categorical). Messages of mixtures with very narrow components have a
rounding floor in natural parameters that their moments do not, so hybrid graphs default to
`:marginals` ([`hybrid_schedule`](@ref)).
"""
function propagate!(s::MessageStore, g::FactorGraph, sched::AbstractSchedule, ps, st;
                    maxsweeps::Int = 100, tol::Real = 1e-8, damping::Real = 0.0,
                    record = nothing, criterion::Symbol = _default_criterion(sched))
    0 <= damping < 1 || throw(ArgumentError("damping must be in [0, 1); got $damping"))
    criterion in (:messages, :marginals) || throw(ArgumentError("criterion must be :messages or :marginals"))
    # Undamped, one tree sweep is exact. Damping keeps part of each stored message, so from a
    # warm start one sweep is not the fixed point: iterate and check like any schedule.
    if sched isa TreeSchedule && iszero(damping) && _tree_exact(g)
        _, st = sweep!(s, g, sched, ps, st; damping)
        return ConvergenceReport(true, 1, 0.0, "tree schedule: one sweep is exact"), st
    end
    res = Inf
    margs = criterion === :marginals ? _all_marginals(s, g) : nothing
    for i in 1:maxsweeps
        res, st = sched isa ResidualSchedule ?
            sweep!(s, g, sched, ps, st; damping, record) : sweep!(s, g, sched, ps, st; damping)
        if criterion === :marginals
            new = _all_marginals(s, g)
            res = maximum(v -> _moment_distance(margs[v], new[v]), eachindex(new); init = 0.0)
            margs = new
        end
        if res <= tol
            return ConvergenceReport(true, i, float(res),
                criterion === :marginals ? "marginals changed by less than tol" : "residual below tol"), st
        end
    end
    reason = isfinite(res) ? "hit maxsweeps" :
        "residual is Inf: no `belief_distance` method for these belief types, so convergence \
         cannot be verified (see messages.md). Ran to maxsweeps rather than claim success."
    return ConvergenceReport(false, maxsweeps, float(res), reason), st
end

"""
    infer!(g, sched, ps, st; kwargs...) -> (marginals::NamedTuple, report, st, store)

Convenience: fresh store, propagate, read out every variable's marginal. The store is returned
for the free energy and the gradients.
"""
function infer!(g::FactorGraph, sched::AbstractSchedule, ps, st; kwargs...)
    s = MessageStore(g)
    report, st = propagate!(s, g, sched, ps, st; kwargs...)
    names = Tuple(v.name for v in g.variables)
    vals = Tuple(marginal(s, g, v.id) for v in g.variables)
    return NamedTuple{names}(vals), report, st, s
end
