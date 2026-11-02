# ---------------------------------------------------------------------------
# Plates: one template, replicated over n observations, with tied parameters.
#
# A plate is built by calling a template function once per replica on a `PlateBuilder`, which
# names the replica's own variables and factors `name[i]` and resolves any other name to the
# enclosing graph (a shared variable or factor). The graph is unrolled: every replica is an
# ordinary set of nodes, so inference, the free energy and every gradient path are the ones of
# any graph. What the plate adds:
#
#   - parameter tying: a factor inside the plate reads the parameter entry of its template
#     name, so the replicas share one parameter vector and its gradient is their sum;
#   - a record of which node of which replica is which (`Plate`), checked to be the same
#     template in every replica;
#   - `plate_schedule`, which lifts a schedule of replica 1 to all replicas, as groups of the
#     same template task.
#
# See `plates.md`.
# ---------------------------------------------------------------------------

"""
    Plate

The record of one plate in a [`FactorGraph`](@ref): its `name`, its size `n`, and the node and
edge ids of each replica, as matrices with one column per replica and one row per template
position (`variables`, `factors`, `edges`). Row `k` of `edges` is the same template edge in
every replica. Read it with [`plates`](@ref).
"""
struct Plate
    name::Symbol
    n::Int
    variables::Matrix{Int}
    factors::Matrix{Int}
    edges::Matrix{Int}
end

"""
    PlateBuilder

What the template function of [`plate!`](@ref) receives: [`variable!`](@ref),
[`factor!`](@ref) and [`connect!`](@ref) on it add one replica's nodes and edges to the
enclosing [`GraphBuilder`](@ref). Its `index` field is the replica number.
"""
mutable struct PlateBuilder
    builder::GraphBuilder
    plate::Symbol
    index::Int
    variables::Vector{Symbol}
    factors::Vector{Symbol}
    signature::Vector{Any}
    vids::Vector{Int}
    fids::Vector{Int}
    eids::Vector{Int}
    created::Tuple{Set{Int},Set{Int}}
end

"""
    replica_name(name::Symbol, i::Integer) -> Symbol

The graph name of replica `i` of a plate node declared as `name`: `Symbol("name[i]")`.
"""
replica_name(name::Symbol, i::Integer) = Symbol(name, "[", i, "]")

"""
    variable!(p::PlateBuilder, name, space = nothing) -> Int

Add replica `p.index`'s copy of the plate variable `name`, as [`replica_name`](@ref)`(name, i)`.
"""
function variable!(p::PlateBuilder, name::Symbol, space = nothing)
    name in p.variables && throw(ArgumentError("duplicate plate variable :$name"))
    id = variable!(p.builder, replica_name(name, p.index), space)
    push!(p.variables, name); push!(p.vids, id); push!(p.created[1], id)
    return id
end

"""
    factor!(p::PlateBuilder, name, factor; params = name, tie = true, kwargs...) -> Int

Add replica `p.index`'s copy of the plate factor `name`. With `tie = true` (the default) every
replica reads the parameter entry `params`, so the replicas share one parameter vector; with
`tie = false` each reads its own, `replica_name(params, i)`. Other keywords are those of
[`factor!`](@ref).
"""
function factor!(p::PlateBuilder, name::Symbol, factor; params::Symbol = name, tie::Bool = true, kwargs...)
    name in p.factors && throw(ArgumentError("duplicate plate factor :$name"))
    key = tie ? params : replica_name(params, p.index)
    id = factor!(p.builder, replica_name(name, p.index), factor; params = key, kwargs...)
    push!(p.factors, name); push!(p.fids, id); push!(p.created[2], id)
    return id
end

"""
    connect!(p::PlateBuilder, factor, channel, variable; kwargs...) -> Int

Connect inside a plate. A name declared in this replica (by `variable!(p, ...)` or
`factor!(p, ...)`) refers to the replica's own node; any other name refers to a node of the
enclosing graph, shared by all replicas. Other keywords are those of [`connect!`](@ref).
"""
function connect!(p::PlateBuilder, factor::Symbol, channel::Symbol, variable::Symbol; kwargs...)
    fref, fname = _plate_ref(p, factor, p.factors, p.builder.factor_index, 2, "factor")
    vref, vname = _plate_ref(p, variable, p.variables, p.builder.variable_index, 1, "variable")
    fref[1] === :outer && vref[1] === :outer && throw(ArgumentError(
        "connect!(p, :$factor, :$channel, :$variable) joins two nodes outside plate :$(p.plate); \
         connect them on the GraphBuilder"))
    id = connect!(p.builder, fname, channel, vname; kwargs...)
    e = p.builder.edges[id]
    push!(p.signature, (fref, channel, vref, e.direction, e.coupling, e.detach))
    push!(p.eids, id)
    return id
end

function _plate_ref(p, name, locals, index, which, what)
    name in locals && return (:local, name), replica_name(name, p.index)
    haskey(index, name) || throw(ArgumentError("no $what :$name in plate :$(p.plate) or outside it"))
    index[name] in p.created[which] && throw(ArgumentError(
        "$what :$name belongs to another replica of plate :$(p.plate); refer to plate nodes by their template names"))
    return (:outer, name), name
end

"""
    plate!(template, b::GraphBuilder, name::Symbol, n::Integer) -> Plate

Replicate a template `n` times: `template(p, i)` is called for `i = 1:n` with a
[`PlateBuilder`](@ref) `p`, and adds replica `i`'s nodes through `variable!(p, ...)`,
`factor!(p, ...)` and `connect!(p, ...)`. Use it as a `do` block:

```julia
plate!(b, :obs, N) do p, i
    variable!(p, :z, k); variable!(p, :x, d)
    factor!(p, :prior, GaussianPrior(:z, zeros(k), I(k)))
    factor!(p, :lik, LinearConstraintFactor((x = d, z = k), d))   # parameters :lik, shared
    factor!(p, :data, DataFactor(:x, X[:, i]))
    connect!(p, :prior, :z, :z); connect!(p, :lik, :z, :z); connect!(p, :lik, :x, :x)
    connect!(p, :data, :x, :x; direction = Emitting())
end
```

Replica `i`'s nodes are named [`replica_name`](@ref)`(name, i)` (`:x` becomes `Symbol("x[3]")`).
Factors are tied across replicas by default (`factor!(p, ...; tie)`), so their gradient is the
sum over replicas. A name not declared in the replica is a node of the enclosing graph, shared
by all replicas: a shared latent variable couples the replicas, a shared parameter does not.
Every replica must declare the same nodes and edges in the same order, with the same
directions, couplings and shared nodes; otherwise this throws. The data may differ. Plates do
not nest.
"""
function plate!(template, b::GraphBuilder, name::Symbol, n::Integer)
    n >= 1 || throw(ArgumentError("a plate needs n ≥ 1 replicas; got $n"))
    any(pl -> pl.name === name, b.plates) && throw(ArgumentError("duplicate plate :$name"))
    created = (Set{Int}(), Set{Int}())
    reps = map(1:n) do i
        p = PlateBuilder(b, name, i, Symbol[], Symbol[], Any[], Int[], Int[], Int[], created)
        template(p, i)
        return p
    end
    r = first(reps)
    for p in reps
        p.variables == r.variables && p.factors == r.factors && isequal(p.signature, r.signature) ||
            throw(ArgumentError("replica $(p.index) of plate :$name does not have replica 1's nodes and edges \
                                 (the same names, channels, directions, couplings and shared nodes, in order)"))
    end
    cols(f) = reshape(reduce(vcat, map(f, reps); init = Int[]), length(f(r)), n)
    pl = Plate(name, n, cols(p -> p.vids), cols(p -> p.fids), cols(p -> p.eids))
    push!(b.plates, pl)
    return pl
end

"""
    plates(g) -> Vector{Plate}

The [`Plate`](@ref)s of a graph, in the order they were declared.
"""
plates(g::FactorGraph) = Plate[pl for pl in g.plates]

function _plate(g::FactorGraph, name)
    pls = plates(g)
    isempty(pls) && throw(ArgumentError("the graph has no plates"))
    name === nothing && length(pls) == 1 && return only(pls)
    name === nothing && throw(ArgumentError("the graph has $(length(pls)) plates; pass plate = name"))
    i = findfirst(pl -> pl.name === name, pls)
    i === nothing && throw(ArgumentError("no plate :$name"))
    return pls[i]
end

# --- Schedules ---------------------------------------------------------------

"""
    PlateSchedule(groups)

A static schedule of groups of tasks. A group is one template task of a plate lifted to every
replica (or a single task outside it). A sweep gives the same messages as the
[`SequentialSchedule`](@ref) of `tasks(sched)`. Where the members of a group are independent, a
group of factor → variable messages is computed in one call of [`factor_messages`](@ref), with
the polarity resolved once: a belief-propagation node, tied replicas (one parameter key), the
same available channels, and a different target variable for each member (a shared target
would see the earlier members' messages in its prior). Otherwise the members run one by one.
Build one with [`plate_schedule`](@ref), which also finds each lifted group's structure once
(`plans`); the polarity of such a group is resolved and checked on its first member. A schedule
built from `groups` alone checks every group at run time.
"""
struct PlateSchedule <: AbstractSchedule
    groups::Vector{Vector{MessageTask}}
    plans::Vector{Any}            # per group: a `_GroupPlan`, or `nothing` (checked at run time)
end
PlateSchedule(groups::Vector{Vector{MessageTask}}) = PlateSchedule(groups, Any[nothing for _ in groups])

# What a lifted group of factor -> variable messages needs, found once when the schedule is
# built: member i's factor `factors[i]` (tied, belief propagation, one type) and its edge on
# channel `channels[j]` at `edges[j, i]`, the same channel layout and directions in every member,
# the target channel's position, and distinct targets.
struct _GroupPlan{F}
    factors::Vector{F}
    channels::Vector{Symbol}
    edges::Matrix{Int}
    target::Int
end

function _group_plan(g::FactorGraph, grp::Vector{MessageTask})
    length(grp) > 1 && all(t -> t.kind === :to_variable && can_emit(g, t.edge), grp) || return nothing
    nodes = [factornode(g, g.edges[t.edge].factor) for t in grp]
    f1 = first(nodes)
    all(h -> h.params === f1.params && h.rule isa BeliefPropagation, nodes) || return nothing
    allunique(g.edges[t.edge].variable for t in grp) || return nothing
    E1 = g.edges_of_factor[f1.id]
    layout(h) = [(g.edges[e].channel, typeof(g.edges[e].direction)) for e in g.edges_of_factor[h.id]]
    l1 = layout(f1)
    all(h -> layout(h) == l1, nodes) || return nothing
    target = findfirst(==(first(grp).edge), E1)
    all(((t, h),) -> g.edges_of_factor[h.id][target] == t.edge, zip(grp, nodes)) || return nothing
    edges = reduce(hcat, [g.edges_of_factor[h.id] for h in nodes])
    factors = [h.factor for h in nodes]
    return _GroupPlan(identity.(factors), first.(l1), reshape(edges, length(E1), length(nodes)), target)
end
tasks(s::PlateSchedule) = reduce(vcat, s.groups; init = MessageTask[])

"""
    plate_schedule(g, replica_tasks; plate = nothing) -> PlateSchedule
    plate_schedule(g, sched::AbstractSchedule; plate = nothing) -> PlateSchedule
    plate_schedule(g; plate = nothing) -> PlateSchedule

Lift a schedule written for replica 1 to every replica. Each task on an edge of replica 1 becomes
the group of the same template task in replicas `1:n`; a task on an edge outside the plate stays
a group of one; a task on another replica's edge throws.

Without a schedule, each connected component gets its tree schedule, and replica 1's component
is lifted. This needs the replicas to be separate components (parameters shared, no shared
variable) and every component to be a tree; otherwise it throws. A graph with shared variables
can use `tree_schedule(g)` directly, or a hand-written replica schedule. `plate` names the plate
when the graph has more than one.
"""
function plate_schedule(g::FactorGraph, replica_tasks::AbstractVector{MessageTask}; plate = nothing)
    pl = _plate(g, plate)
    where_ = Dict{Int,Tuple{Int,Int}}()
    for i in 1:pl.n, k in axes(pl.edges, 1)
        where_[pl.edges[k, i]] = (k, i)
    end
    groups = map(replica_tasks) do t
        haskey(where_, t.edge) || return [t]
        k, i = where_[t.edge]
        i == 1 || throw(ArgumentError(
            "task on edge $(t.edge) belongs to replica $i of plate :$(pl.name); write the schedule for replica 1"))
        return [MessageTask(pl.edges[k, j], t.kind) for j in 1:pl.n]
    end
    return PlateSchedule(groups, Any[_group_plan(g, grp) for grp in groups])
end
plate_schedule(g::FactorGraph, sched::AbstractSchedule; plate = nothing) =
    plate_schedule(g, tasks(sched); plate)

function plate_schedule(g::FactorGraph; plate = nothing)
    pl = _plate(g, plate)
    replica_of = Dict{Tuple{Symbol,Int},Int}()
    for i in 1:pl.n
        foreach(v -> replica_of[(:variable, v)] = i, pl.variables[:, i])
        foreach(f -> replica_of[(:factor, f)] = i, pl.factors[:, i])
    end
    seen = Set{Tuple{Symbol,Int}}()
    groups = Vector{MessageTask}[]
    plans = Any[]
    for node in vcat([(:variable, v) for v in 1:nvariables(g)], [(:factor, f) for f in 1:nfactors(g)])
        node in seen && continue
        comp = _component(g, node)
        union!(seen, comp)
        reps = unique(get(replica_of, c, 0) for c in comp)
        length(reps) == 1 || throw(ArgumentError(
            "plate_schedule(g) needs plate :$(pl.name)'s replicas to be separate components, but replicas \
             $(filter(!=(0), reps)) share a variable or factor; use tree_schedule(g) or a replica schedule"))
        ne = sum(c -> c[1] === :factor ? factor_degree(g, c[2]) : 0, comp)
        ne == length(comp) - 1 || throw(ArgumentError(
            "a component of the graph has a loop; plate_schedule(g) builds tree schedules only"))
        only(reps) >= 2 && continue                      # lifted from replica 1
        root = something(findfirst(c -> (c[1] === :variable ? variable_degree(g, c[2]) : factor_degree(g, c[2])) == 1, comp), 1)
        ts = _tree_sweeps(g, comp[root])
        sub = plate_schedule(g, tasks(ts); plate = pl.name)
        append!(groups, sub.groups); append!(plans, sub.plans)
    end
    return PlateSchedule(groups, plans)
end

# The nodes reachable from `node`, breadth first.
function _component(g::FactorGraph, node)
    out = [node]; seen = Set([node]); i = 1
    while i <= length(out)
        kind, id = out[i]; i += 1
        nbrs = kind === :variable ?
            [(:factor, g.edges[e].factor) for e in g.edges_of_variable[id]] :
            [(:variable, g.edges[e].variable) for e in g.edges_of_factor[id]]
        for m in nbrs
            m in seen && continue
            push!(seen, m); push!(out, m)
        end
    end
    return out
end

# --- Grouped execution ---------------------------------------------------------

function sweep!(s::MessageStore, g::FactorGraph, sched::PlateSchedule, ps, st; damping::Real = 0.0)
    0 <= damping < 1 || throw(ArgumentError("damping must be in [0, 1); got $damping"))
    s.iteration += 1
    maxres = 0.0
    for (grp, plan) in zip(sched.groups, sched.plans)
        res, st = length(grp) == 1 ? _step_each!(s, g, grp, ps, st, damping) :
                  _to_factor_group(g, grp) ? (_step_to_factor_group!(s, g, grp), st) :
                  plan !== nothing ? _step_planned!(s, g, grp, plan, ps, st, damping) :
                  _batchable(s, g, grp) ? _step_group!(s, g, grp, ps, st, damping) :
                  _step_each!(s, g, grp, ps, st, damping)
        maxres = max(maxres, res)
    end
    return maxres, st
end

function _step_each!(s, g, grp, ps, st, damping)
    maxres = 0.0
    for t in grp
        _, res, st = step!(s, g, t, ps, st; damping)
        maxres = max(maxres, res)
    end
    return maxres, st
end

# Variable -> factor messages: each is the product of the other messages at its variable, which
# no member of the group writes (they write variable -> factor messages only).
_to_factor_group(g, grp) = all(t -> t.kind === :to_factor && can_absorb(g, t.edge), grp)

function _step_to_factor_group!(s, g, grp)
    maxres = 0.0
    for t in grp
        e = g.edges[t.edge]
        belief = excluded_marginal(s, g, e.variable, t.edge)
        belief = e.detach ? LenticulumCore.stop_gradient(belief) : belief
        old = s.to_factor[t.edge]
        maxres = max(maxres, belief_distance(old === nothing ? nothing : old.belief, belief))
        s.to_factor[t.edge] = Message(belief, s.iteration)
    end
    return maxres
end

# Whether the group's factor -> variable messages can be computed together (see `PlateSchedule`).
function _batchable(s, g, grp)
    t1 = first(grp)
    t1.kind === :to_variable && can_emit(g, t1.edge) || return false
    e1 = g.edges[t1.edge]
    fn = factornode(g, e1.factor)
    fn.rule isa BeliefPropagation || return false
    chans = available_channels(s, g, e1.factor; skip = t1.edge)
    # no inputs and no method for that polarity: `step!` sends TrivialBelief() with a warning
    isempty(chans) && !issink(fn.factor) &&
        !LenticulumCore.supports_polarity(fn.factor, resolve_polarity(g, e1.factor, e1.channel, chans)) && return false
    targets = Set{Int}()
    for t in grp
        t.kind === :to_variable && can_emit(g, t.edge) || return false
        e = g.edges[t.edge]
        h = factornode(g, e.factor)
        e.channel === e1.channel && h.params === fn.params && h.rule isa BeliefPropagation || return false
        e.variable in targets && return false
        push!(targets, e.variable)
        available_channels(s, g, e.factor; skip = t.edge) == chans || return false
    end
    return true
end

function _step_group!(s, g, grp, ps, st, damping)
    e1 = g.edges[first(grp).edge]
    fn = factornode(g, e1.factor)
    chans = available_channels(s, g, e1.factor; skip = first(grp).edge)
    pol = resolve_polarity(g, e1.factor, e1.channel, chans)
    foreach(t -> check_legal(g, g.edges[t.edge].factor, pol), grp)   # edge directions may differ
    psf = _factor_parameters(fn, _subtree(ps, fn.params))
    stf = _subtree(st, fn.params)
    inputs = [_inputs(s, g, g.edges[t.edge].factor, chans) for t in grp]
    priors = _LazyPriors(s, g, grp)
    factors = [factornode(g, g.edges[t.edge].factor).factor for t in grp]
    beliefs, stf = factor_messages(factors, e1.channel, pol, inputs, priors, psf, stf)
    length(beliefs) == length(grp) || throw(ArgumentError(
        "factor_messages of $(nameof(typeof(fn.factor))) returned $(length(beliefs)) messages for $(length(grp)) replicas"))
    maxres = 0.0
    for (t, belief) in zip(grp, beliefs)
        e = g.edges[t.edge]
        old = s.to_variable[t.edge]
        if old !== nothing && damping > 0
            belief = damp(belief, old.belief, 1 - damping)
        end
        belief = e.detach ? LenticulumCore.stop_gradient(belief) : belief
        maxres = max(maxres, belief_distance(old === nothing ? nothing : old.belief, belief))
        s.to_variable[t.edge] = Message(belief, s.iteration)
    end
    return maxres, _setsubtree(st, fn.params, stf)
end

# The `priors` of a group, computed when a factor reads them (most ignore them). Read before
# any member's message is committed, so they are the sequential schedule's priors.
struct _LazyPriors{S,G} <: AbstractVector{Any}
    s::S
    g::G
    grp::Vector{MessageTask}
end
Base.size(p::_LazyPriors) = (length(p.grp),)
Base.getindex(p::_LazyPriors, i::Int) = excluded_marginal(p.s, p.g, p.g.edges[p.grp[i].edge].variable, p.grp[i].edge)

_informative_input(s, e) = has_to_factor(s, e) && !LenticulumCore.isuninformative(s.to_factor[e].belief)

# A group with a `_GroupPlan`: availability is read through the plan's edges, the polarity is
# resolved and checked on the first member only (the plan holds one layout of channels and
# directions, and one factor type), and the inputs are gathered without searching edges.
function _step_planned!(s, g, grp, plan::_GroupPlan, ps, st, damping)
    E, n, nc = plan.edges, length(grp), length(plan.channels)
    mask = [j != plan.target && _informative_input(s, E[j, 1]) for j in 1:nc]
    for i in 2:n, j in 1:nc
        j == plan.target && continue
        _informative_input(s, E[j, i]) == mask[j] || return _step_each!(s, g, grp, ps, st, damping)
    end
    js = findall(mask)
    chans = Tuple(plan.channels[js])
    e1 = g.edges[first(grp).edge]
    fn = factornode(g, e1.factor)
    pol = resolve_polarity(g, e1.factor, e1.channel, chans)
    isempty(chans) && !issink(fn.factor) && !LenticulumCore.supports_polarity(fn.factor, pol) &&
        return _step_each!(s, g, grp, ps, st, damping)
    check_legal(g, e1.factor, pol)
    psf = _factor_parameters(fn, _subtree(ps, fn.params))
    stf = _subtree(st, fn.params)
    inputs = _gather_inputs(NamedTuple{chans}, s.to_factor, E, js)
    beliefs, stf = factor_messages(plan.factors, e1.channel, pol, inputs, _LazyPriors(s, g, grp), psf, stf)
    length(beliefs) == n || throw(ArgumentError(
        "factor_messages of $(nameof(typeof(fn.factor))) returned $(length(beliefs)) messages for $n replicas"))
    return _commit_group!(s, g, grp, beliefs, damping), _setsubtree(st, fn.params, stf)
end

function _gather_inputs(::Type{NT}, tf, E, js) where {NT}
    return [NT(Tuple(tf[E[j, i]].belief for j in js)) for i in axes(E, 2)]
end

function _commit_group!(s, g, grp, beliefs, damping)
    maxres = 0.0
    for (t, belief) in zip(grp, beliefs)
        e = g.edges[t.edge]
        old = s.to_variable[t.edge]
        if old !== nothing && damping > 0
            belief = damp(belief, old.belief, 1 - damping)
        end
        belief = e.detach ? LenticulumCore.stop_gradient(belief) : belief
        maxres = max(maxres, belief_distance(old === nothing ? nothing : old.belief, belief))
        s.to_variable[t.edge] = Message(belief, s.iteration)
    end
    return maxres
end
