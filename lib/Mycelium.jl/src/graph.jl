# ---------------------------------------------------------------------------
# The factor graph.
#
# Bipartite: VARIABLE nodes (which carry beliefs) and FACTOR nodes (which are
# `AbstractLenticulumFactor`s, i.e. parameterized statistical games). Edges attach one
# NAMED CHANNEL of a factor to one variable.
#
# Edges are DIRECTED, and a direction is one of three things:
#
#     Emitting     factor -> variable only    (the factor can only produce here)
#     Absorbing    variable -> factor only    (the factor can only consume here)
#     Bidirectional                           (both; the implicit case)
#
# Two different acyclicity notions matter and must not be confused:
#
#     istree(g)  -- the UNDIRECTED bipartite graph is a tree.  Controls whether message
#                   passing is EXACT (two sweeps) or merely approximate (loopy BP).
#     isdag(g)   -- the DIRECTED multigraph has no directed cycle.  Controls whether the
#                   graph is Lux-compatible, i.e. can be run as a forward/backward pass.
#                   Any Bidirectional edge makes this false by itself.
#
# See `graph.md`.
# ---------------------------------------------------------------------------

"""
    abstract type EdgeDirection

Which way messages may travel along an edge. The graph is directed; bidirectional edges are
the special (and, for implicit factors, usual) case.
"""
abstract type EdgeDirection end

"""
    Bidirectional()

Messages flow both ways. The attached channel may be `Observed()` or `Unobserved()`
depending on the message being computed — this is what makes a factor implicit.
"""
struct Bidirectional <: EdgeDirection end

"""
    Emitting()

Factor → variable only. The factor *produces* on this channel and never reads it, so the
channel is never `Observed()`. A data source, a prior, and the output side of an ordinary
Lux-style layer are all emitting.
"""
struct Emitting <: EdgeDirection end

"""
    Absorbing()

Variable → factor only. The factor *consumes* on this channel and never writes it, so the
channel is never `Unobserved()`. A loss and the input side of an ordinary layer are
absorbing.
"""
struct Absorbing <: EdgeDirection end

emits(::Bidirectional) = true
emits(::Emitting) = true
emits(::Absorbing) = false

absorbs(::Bidirectional) = true
absorbs(::Emitting) = false
absorbs(::Absorbing) = true

"""
    VariableNode(id, name, space)

A wire. Carries a belief; has no behaviour of its own.

Per AutoBayes' Remark 24 (and the design note [[Everything is a Factor]]), *data is not a
variable* — data is a `DataFactor` of arity one attached to a variable. Variables are purely
structural.
"""
struct VariableNode{S}
    id::Int
    name::Symbol
    space::S
end

"""
    FactorNode(id, name, factor)

A node holding an `AbstractLenticulumFactor` — a parameterized statistical game.
"""
struct FactorNode{F}
    id::Int
    name::Symbol
    factor::F
end

"""
    Edge(factor, channel, variable, direction, coupling)

Attaches channel `channel` of factor node `factor` to variable node `variable`.

`coupling` is the `AbstractGradientCoupling` for this edge: which of the terms AutoBayes'
Definition 29 drops are restored here. Per-edge rather than global, because a conjugate edge
and a discrete-latent edge genuinely want different answers.
"""
struct Edge{D<:EdgeDirection,C<:AbstractGradientCoupling}
    factor::Int
    channel::Symbol
    variable::Int
    direction::D
    coupling::C
end

"""
    FactorGraph

A bipartite graph of variables and factors with named, directed channel attachments.

Build one with [`GraphBuilder`](@ref); do not construct directly, since the adjacency
indices must stay consistent with the edge list.
"""
struct FactorGraph{V<:Tuple,F<:Tuple}
    variables::Vector{VariableNode}
    factors::Vector{FactorNode}
    edges::Vector{Edge}
    edges_of_factor::Vector{Vector{Int}}
    edges_of_variable::Vector{Vector{Int}}
    variable_index::Dict{Symbol,Int}
    factor_index::Dict{Symbol,Int}
    _v::V
    _f::F
end

# --- Construction ----------------------------------------------------------

"""
    GraphBuilder()

Mutable builder. Use [`variable!`](@ref), [`factor!`](@ref), [`connect!`](@ref), then
[`build`](@ref).
"""
mutable struct GraphBuilder
    variables::Vector{VariableNode}
    factors::Vector{FactorNode}
    edges::Vector{Edge}
    variable_index::Dict{Symbol,Int}
    factor_index::Dict{Symbol,Int}
end
GraphBuilder() = GraphBuilder(
    VariableNode[], FactorNode[], Edge[], Dict{Symbol,Int}(), Dict{Symbol,Int}()
)

"""
    variable!(b::GraphBuilder, name::Symbol, space = nothing) -> Int

Add a variable node; returns its id.
"""
function variable!(b::GraphBuilder, name::Symbol, space = nothing)
    haskey(b.variable_index, name) && throw(ArgumentError("duplicate variable :$name"))
    id = length(b.variables) + 1
    push!(b.variables, VariableNode(id, name, space))
    b.variable_index[name] = id
    return id
end

"""
    factor!(b::GraphBuilder, name::Symbol, factor) -> Int

Add a factor node wrapping an `AbstractLenticulumFactor`; returns its id.
"""
function factor!(b::GraphBuilder, name::Symbol, factor)
    haskey(b.factor_index, name) && throw(ArgumentError("duplicate factor :$name"))
    id = length(b.factors) + 1
    push!(b.factors, FactorNode(id, name, factor))
    b.factor_index[name] = id
    return id
end

"""
    connect!(b, factor::Symbol, channel::Symbol, variable::Symbol;
             direction = Bidirectional(), coupling = DiagonalCoupling())

Attach one channel of a factor to a variable.

Each (factor, channel) pair may be connected at most once: a channel is a single port, not a
bus. Fanning a variable out to several factors is done by connecting *those factors*, which
is the graph-level form of the copier of `Copiers Cups and Caps.md`.
"""
function connect!(
    b::GraphBuilder, factor::Symbol, channel::Symbol, variable::Symbol;
    direction::EdgeDirection = Bidirectional(),
    coupling::AbstractGradientCoupling = DiagonalCoupling(),
)
    haskey(b.factor_index, factor) || throw(ArgumentError("no factor :$factor"))
    haskey(b.variable_index, variable) || throw(ArgumentError("no variable :$variable"))
    fid = b.factor_index[factor]
    vid = b.variable_index[variable]
    for e in b.edges
        if e.factor == fid && e.channel == channel
            throw(ArgumentError("channel :$channel of factor :$factor is already connected"))
        end
    end
    push!(b.edges, Edge(fid, channel, vid, direction, coupling))
    return length(b.edges)
end

"""
    build(b::GraphBuilder) -> FactorGraph

Freeze the builder into an immutable graph with adjacency indices.
"""
function build(b::GraphBuilder)
    eof = [Int[] for _ in b.factors]
    eov = [Int[] for _ in b.variables]
    for (i, e) in enumerate(b.edges)
        push!(eof[e.factor], i)
        push!(eov[e.variable], i)
    end
    return FactorGraph(
        copy(b.variables), copy(b.factors), copy(b.edges), eof, eov,
        copy(b.variable_index), copy(b.factor_index),
        Tuple(v.name for v in b.variables), Tuple(f.name for f in b.factors),
    )
end

# --- Queries ---------------------------------------------------------------

nvariables(g::FactorGraph) = length(g.variables)
nfactors(g::FactorGraph) = length(g.factors)
nedges(g::FactorGraph) = length(g.edges)

variable(g::FactorGraph, name::Symbol) = g.variables[g.variable_index[name]]
variable(g::FactorGraph, id::Int) = g.variables[id]
factornode(g::FactorGraph, name::Symbol) = g.factors[g.factor_index[name]]
factornode(g::FactorGraph, id::Int) = g.factors[id]

edges_of_factor(g::FactorGraph, fid::Int) = g.edges_of_factor[fid]
edges_of_variable(g::FactorGraph, vid::Int) = g.edges_of_variable[vid]

"""
    degree(g, ::Val{:variable}, vid) -> Int

The number of edges incident to a variable. This is the ``d_v`` of the Bethe counting
number ``1 - d_v``; see `free_energy.jl`.
"""
variable_degree(g::FactorGraph, vid::Int) = length(g.edges_of_variable[vid])
factor_degree(g::FactorGraph, fid::Int) = length(g.edges_of_factor[fid])

"""
    channels_of(g, fid) -> Tuple{Vararg{Symbol}}

The channel names of factor `fid` that are actually connected in this graph. A factor may
declare more channels than the graph uses; the unused ones are `Latent()` in every message.
"""
channels_of(g::FactorGraph, fid::Int) = Tuple(g.edges[e].channel for e in g.edges_of_factor[fid])

"""
    euler_characteristic(g) -> Int

``|F| + |V| - |E|``.

Equal to `1` exactly when the graph is a connected tree, and `1 - L` when it has `L`
independent loops. It is also **the sum of the Bethe counting numbers**
``\\sum_f 1 + \\sum_v (1 - d_v)``, since ``\\sum_v d_v = |E|`` — so this single integer says
how badly the free-energy bookkeeping over-counts. See `free_energy.md`.
"""
euler_characteristic(g::FactorGraph) = nfactors(g) + nvariables(g) - nedges(g)

"""
    isconnected(g) -> Bool

Whether the undirected bipartite graph is connected.
"""
function isconnected(g::FactorGraph)
    (nvariables(g) + nfactors(g)) == 0 && return true
    seen_v = falses(nvariables(g))
    seen_f = falses(nfactors(g))
    nvariables(g) == 0 && return nfactors(g) <= 1
    stack = [(:v, 1)]
    seen_v[1] = true
    while !isempty(stack)
        kind, id = pop!(stack)
        if kind === :v
            for e in g.edges_of_variable[id]
                f = g.edges[e].factor
                if !seen_f[f]
                    seen_f[f] = true
                    push!(stack, (:f, f))
                end
            end
        else
            for e in g.edges_of_factor[id]
                v = g.edges[e].variable
                if !seen_v[v]
                    seen_v[v] = true
                    push!(stack, (:v, v))
                end
            end
        end
    end
    return all(seen_v) && all(seen_f)
end

"""
    istree(g) -> Bool

Whether the **undirected** bipartite graph is a tree: connected and loop-free.

This is the property that decides whether message passing is **exact**. On a tree, two
sweeps (inward then outward) give the exact marginals and the exact free energy. Off a tree
you are running loopy BP and everything is an approximation — see `Loopy Message Passing.md`.

Do not confuse with [`isdag`](@ref), which is about *edge directions* and decides whether the
graph is Lux-compatible.
"""
istree(g::FactorGraph) = isconnected(g) && euler_characteristic(g) == 1

"""
    isdag(g) -> Bool

Whether the **directed** multigraph induced by the edge directions has no directed cycle.

An `Emitting` edge contributes the arc factor → variable, an `Absorbing` edge the arc
variable → factor, and a `Bidirectional` edge contributes **both** — so a single
bidirectional edge is already a 2-cycle and makes this `false`.

That is the correct behaviour, not an artefact: a bidirectional edge is exactly what
"implicit" means, and `isdag(g)` is precisely the predicate "this graph could have been
written in Lux".
"""
function isdag(g::FactorGraph)
    nv, nf = nvariables(g), nfactors(g)
    n = nv + nf
    vnode(i) = i
    fnode(i) = nv + i
    adj = [Int[] for _ in 1:n]
    for e in g.edges
        if emits(e.direction)
            push!(adj[fnode(e.factor)], vnode(e.variable))
        end
        if absorbs(e.direction)
            push!(adj[vnode(e.variable)], fnode(e.factor))
        end
    end
    state = zeros(UInt8, n)  # 0 unvisited, 1 on stack, 2 done
    function visit(u)
        state[u] == 1 && return false
        state[u] == 2 && return true
        state[u] = 1
        for w in adj[u]
            visit(w) || return false
        end
        state[u] = 2
        return true
    end
    for u in 1:n
        visit(u) || return false
    end
    return true
end

"""
    topological_order(g) -> Vector{Tuple{Symbol,Int}}

Nodes in topological order as `(:variable, id)` / `(:factor, id)` pairs. Errors unless
[`isdag`](@ref).

This is the order in which an ordinary forward pass runs, and it exists only for the
Lux-compatible case. Cyclic graphs get a [`FloodingSchedule`](@ref) or a
[`ResidualSchedule`](@ref) instead.
"""
function topological_order(g::FactorGraph)
    isdag(g) || throw(ArgumentError(
        "graph is not a DAG: it has a directed cycle (a Bidirectional edge is already one). \
         Use a flooding or residual schedule instead of a forward/backward pass."
    ))
    nv, nf = nvariables(g), nfactors(g)
    n = nv + nf
    adj = [Int[] for _ in 1:n]
    indeg = zeros(Int, n)
    for e in g.edges
        if emits(e.direction)
            push!(adj[nv + e.factor], e.variable);   indeg[e.variable] += 1
        end
        if absorbs(e.direction)
            push!(adj[e.variable], nv + e.factor);   indeg[nv + e.factor] += 1
        end
    end
    queue = [u for u in 1:n if indeg[u] == 0]
    out = Tuple{Symbol,Int}[]
    while !isempty(queue)
        u = popfirst!(queue)
        push!(out, u <= nv ? (:variable, u) : (:factor, u - nv))
        for w in adj[u]
            indeg[w] -= 1
            indeg[w] == 0 && push!(queue, w)
        end
    end
    return out
end

"""
    leaves(g) -> Vector{Tuple{Symbol,Int}}

Nodes of undirected degree 1 — where a tree sweep starts.
"""
function leaves(g::FactorGraph)
    out = Tuple{Symbol,Int}[]
    for v in g.variables
        variable_degree(g, v.id) == 1 && push!(out, (:variable, v.id))
    end
    for f in g.factors
        factor_degree(g, f.id) == 1 && push!(out, (:factor, f.id))
    end
    return out
end

"""
    learnable_factors(g) -> Vector{Int}

Factor node ids that contribute trainable parameters, per `LenticulumCore.islearnable`.
A graph of non-learnable factors is an inference graph, not a training graph.
"""
learnable_factors(g::FactorGraph) =
    [f.id for f in g.factors if LenticulumCore.islearnable(f.factor)]

function Base.show(io::IO, g::FactorGraph)
    print(io, "FactorGraph(", nvariables(g), " variables, ", nfactors(g), " factors, ",
        nedges(g), " edges; χ=", euler_characteristic(g),
        istree(g) ? ", tree" : ", loopy", isdag(g) ? ", DAG)" : ", cyclic)")
end
