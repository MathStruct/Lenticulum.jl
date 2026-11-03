# ---------------------------------------------------------------------------
# Structure edits on a built graph (Extending Mycelium §6, L-SPEC-010).
#
# A `FactorGraph` is immutable. An edit opens it as a builder (`GraphBuilder(g)`), changes it by
# name (the builder's `variable!`, `factor!`, `connect!`, and the removals below), and builds a new
# graph; `transfer` carries the messages of the edges both graphs share into a store for the new
# one, as a warm start. Nodes and edges are identified by name, so ids may change freely.
#
# See `edits.md`.
# ---------------------------------------------------------------------------

"""
    GraphBuilder(g::FactorGraph) -> GraphBuilder

A builder holding a copy of `g`'s variables, factors and edges, to be edited (`variable!`,
`factor!`, `connect!`, [`remove_factor!`](@ref), [`remove_variable!`](@ref),
[`disconnect!`](@ref), [`replace_factor!`](@ref)) and built again. Graphs with plates are refused:
their replica records would not survive the edits.
"""
function GraphBuilder(g::FactorGraph)
    isempty(g.plates) || throw(ArgumentError("GraphBuilder(g): graphs with plates cannot be edited"))
    b = GraphBuilder()
    b.variables = copy(g.variables)
    b.factors = copy(g.factors)
    b.edges = copy(g.edges)
    b.variable_index = copy(g.variable_index)
    b.factor_index = copy(g.factor_index)
    b.ports = Set((e.factor, e.channel) for e in g.edges)
    return b
end

# Keep the marked variables, factors and edges, renumbering ids (names stay).
function _keep!(b::GraphBuilder, keepv, keepf, keepe)
    vmap, fmap = zeros(Int, length(b.variables)), zeros(Int, length(b.factors))
    vs, fs = VariableNode[], FactorNode[]
    for v in b.variables
        keepv[v.id] || continue
        push!(vs, VariableNode(length(vs) + 1, v.name, v.space)); vmap[v.id] = length(vs)
    end
    for f in b.factors
        keepf[f.id] || continue
        push!(fs, FactorNode(length(fs) + 1, f.name, f.factor, f.freeze, f.rule, f.params)); fmap[f.id] = length(fs)
    end
    es = Edge[]
    for (i, e) in enumerate(b.edges)
        keepe[i] && keepv[e.variable] && keepf[e.factor] || continue
        push!(es, Edge(fmap[e.factor], e.channel, vmap[e.variable], e.direction, e.coupling, e.detach))
    end
    b.variables, b.factors, b.edges = vs, fs, es
    b.variable_index = Dict(v.name => v.id for v in vs)
    b.factor_index = Dict(f.name => f.id for f in fs)
    b.ports = Set((e.factor, e.channel) for e in es)
    return b
end

"""
    remove_factor!(b::GraphBuilder, name) -> GraphBuilder

Remove a factor node and its edges. Ids are renumbered; names stay.
"""
function remove_factor!(b::GraphBuilder, name::Symbol)
    haskey(b.factor_index, name) || throw(ArgumentError("no factor :$name"))
    fid = b.factor_index[name]
    return _keep!(b, trues(length(b.variables)), [f.id != fid for f in b.factors], trues(length(b.edges)))
end
"""
    remove_variable!(b::GraphBuilder, name) -> GraphBuilder

Remove a variable and its edges. A factor left with an unconnected channel is allowed, as at
construction (its channel then resolves as latent).
"""
function remove_variable!(b::GraphBuilder, name::Symbol)
    haskey(b.variable_index, name) || throw(ArgumentError("no variable :$name"))
    vid = b.variable_index[name]
    return _keep!(b, [v.id != vid for v in b.variables], trues(length(b.factors)), trues(length(b.edges)))
end
"""
    disconnect!(b::GraphBuilder, factor, channel) -> GraphBuilder

Remove the edge of `factor`'s `channel`; `connect!` can attach the channel again.
"""
function disconnect!(b::GraphBuilder, factor::Symbol, channel::Symbol)
    haskey(b.factor_index, factor) || throw(ArgumentError("no factor :$factor"))
    fid = b.factor_index[factor]
    i = findfirst(e -> e.factor == fid && e.channel === channel, b.edges)
    i === nothing && throw(ArgumentError("channel :$channel of factor :$factor is not connected"))
    return _keep!(b, trues(length(b.variables)), trues(length(b.factors)), [j != i for j in eachindex(b.edges)])
end

"""
    replace_factor!(b::GraphBuilder, name, factor; freeze, rule, params) -> GraphBuilder

Put `factor` in node `name`, keeping the node's edges whose channels `factor` still has and
dropping the others; `freeze`, `rule` and `params` default to the node's. Its new channels are
connected with `connect!` as usual. A sum node that gains an input is replaced this way.
"""
function replace_factor!(b::GraphBuilder, name::Symbol, factor; freeze = nothing, rule = nothing, params = nothing)
    haskey(b.factor_index, name) || throw(ArgumentError("no factor :$name"))
    fid = b.factor_index[name]
    f = b.factors[fid]
    b.factors[fid] = FactorNode(fid, name, factor, something(freeze, f.freeze), something(rule, f.rule), something(params, f.params))
    chans = Set(LenticulumCore.channelname(c) for c in LenticulumCore.channels(factor))
    return _keep!(b, trues(length(b.variables)), trues(length(b.factors)),
                  [e.factor != fid || e.channel in chans for e in b.edges])
end

"""
    transfer(s::MessageStore, g_old, g_new) -> MessageStore

A store for `g_new` holding `s`'s messages on every edge both graphs have: the same factor name,
channel, variable name and variable space, in either direction, and the pins of the variables both
have (with the same space). The
other edges start empty. A warm start: a message kept next to an edit (into a node whose
neighbourhood changed) is a starting point, and the schedule replaces it.
"""
function transfer(s::MessageStore, g_old::FactorGraph, g_new::FactorGraph)
    # the variable's space is part of the key: a variable re-added under its name with another
    # dimension keeps nothing
    key(g, e) = (g.factors[e.factor].name, e.channel, g.variables[e.variable].name, g.variables[e.variable].space)
    old = Dict(key(g_old, e) => i for (i, e) in enumerate(g_old.edges))
    t = MessageStore(g_new)
    for (i, e) in enumerate(g_new.edges)
        j = get(old, key(g_new, e), 0)
        j == 0 && continue
        m = s.to_variable[j]; m === nothing || (t.to_variable[i] = m)
        m = s.to_factor[j]; m === nothing || (t.to_factor[i] = m)
    end
    for (vid, value) in s.pins
        name = g_old.variables[vid].name
        v = get(g_new.variable_index, name, 0)
        v != 0 && isequal(g_new.variables[v].space, g_old.variables[vid].space) && (t.pins[v] = value)
    end
    t.iteration = s.iteration
    return t
end
