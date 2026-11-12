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
Lux-style layer can all send emitting messages. For graph free energy, a factor with
[`uses_cavity`](@ref) (including `PriorFactor`) needs `Bidirectional()` instead, so it can
also receive the cavity message.
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
    AbstractMessageRule

How a factor node computes its outgoing messages: [`BeliefPropagation`](@ref), the default,
[`MeanField`](@ref), [`ExpectationPropagationRule`](@ref) or [`MixtureReduction`](@ref). Chosen per node with
`factor!(b, name, f; rule)`.
"""
abstract type AbstractMessageRule end

"""
    BeliefPropagation()

Sum-product messages: the factor is inverted against the cavity (variable → factor) messages
of its other channels. The default rule of every factor node.
"""
struct BeliefPropagation <: AbstractMessageRule end

"""
    MeanField()

Mean-field messages: the message to channel ``t`` is ``\\exp \\mathbb E_{\\prod_{i \\ne t} q_i}[\\log f]``, taken
over the *current marginals* ``q_i`` of the other channels' variables rather than their cavity
messages, computed by [`mean_field_message`](@ref). Mixing such nodes with
[`BeliefPropagation`](@ref) nodes gives the hybrid of Riegler et al. (2013): exact within
the BP part, factorised across the mean-field nodes. Iterate a sequential schedule to a fixed
point; one tree sweep is not one. The graph free energy is then the merged free energy of that
paper, eq. (17), with each mean-field node's [`mean_field_energy`](@ref) (`free_energy.md`).
"""
struct MeanField <: AbstractMessageRule end

"""
    ExpectationPropagationRule()

Expectation-propagation messages (Minka 2001): the belief-propagation message ``m_t`` is
computed as usual, and when it leaves the target's family (a mixture of Gaussians, say) the
*tilted* belief ``\\pi_t\\, m_t``, with ``\\pi_t`` the target's cavity (excluded marginal), is
projected onto the family by matching moments ([`ep_project`](@ref)) and the cavity is divided
out ([`ep_divide`](@ref)):

```math
m_t^{\\mathrm{EP}} \\;=\\; \\operatorname{proj}\\bigl[\\pi_t\\, m_t\\bigr] \\,/\\, \\pi_t .
```

A message already in its family is sent unchanged, so on a Gaussian graph the rule is belief
propagation. Gaussian division is subtraction of canonical parameters; the site may have an
indefinite precision and is then declared improper. An update whose cavity is not a proper
Gaussian, or whose tilted belief has a component that is not, is skipped: the edge keeps its
previous message. Lenticulum's `ExpectationPropagation` is a different thing: the inversion
method of a `LikelihoodFactor`. Iterate to a fixed point (with damping if
it oscillates); one tree sweep is not one. The graph free energy counts the node as a
belief-propagation node, which at a fixed point is the expectation-consistent Bethe free energy
(`passing.md`, "Expectation propagation").
"""
struct ExpectationPropagationRule <: AbstractMessageRule end

"""
    MixtureReduction(maxcomponents)

Mixture messages with a budget: the factor → variable message is the exact one, with every
mixture input expanded into its components ([`mixture_message`](@ref)) instead of collapsed to
its moments, and then reduced to at most `maxcomponents` components ([`reduce_message`](@ref);
for Gaussian mixtures Runnalls' merge). `MixtureReduction(1)` is assumed-density filtering on the outgoing
message; a large enough budget is exact belief propagation with mixtures. The graph free energy
is not defined for these nodes, and is refused.
"""
struct MixtureReduction <: AbstractMessageRule
    maxcomponents::Int
    function MixtureReduction(maxcomponents::Integer)
        maxcomponents >= 1 || throw(ArgumentError("MixtureReduction needs a budget of at least one component; got $maxcomponents"))
        return new(Int(maxcomponents))
    end
end

# Rules whose messages are computed from cavity (variable -> factor) messages, and whose nodes the
# Bethe free energy counts with their local free energy and their entropy (`free_energy.md`).
_cavity_rule(r::AbstractMessageRule) = r isa BeliefPropagation || r isa ExpectationPropagationRule

"""
    FactorNode(id, name, factor)

A node holding an `AbstractLenticulumFactor` — a parameterized statistical game — together with
its frozen parameter paths, its [`AbstractMessageRule`](@ref) and its **parameter key**
`params`: the entry of the graph's parameter and state trees it reads. The key is the node's
name unless `factor!(...; params)` ties it to other nodes ([`parameter_key`](@ref)).
"""
struct FactorNode{F,P,R<:AbstractMessageRule}
    id::Int
    name::Symbol
    factor::F
    freeze::P
    rule::R
    params::Symbol
end
FactorNode(id, name, factor) = FactorNode(id, name, factor, (), BeliefPropagation(), name)
FactorNode(id, name, factor, freeze) = FactorNode(id, name, factor, freeze, BeliefPropagation(), name)
FactorNode(id, name, factor, freeze, rule::AbstractMessageRule) = FactorNode(id, name, factor, freeze, rule, name)

"""
    parameter_key(node::FactorNode) -> Symbol

The entry of the graph's parameter tree `ps` (and state tree `st`) this node reads. Nodes with
the same key are **tied**: they share one parameter vector, and the graph gradient of that
entry is the sum of their contributions ([`plate!`](@ref), `graph.md`).
"""
parameter_key(f::FactorNode) = f.params
LenticulumCore.isfrozen(f::FactorNode) = LenticulumCore.isfrozen(f.factor) || f.freeze === :all || !isempty(LenticulumCore._freeze_paths(f.freeze))
LenticulumCore.islearnable(f::FactorNode) = LenticulumCore.islearnable(f.factor)

"""
    Edge(factor, channel, variable, direction, coupling)

Attaches channel `channel` of factor node `factor` to variable node `variable`.

`coupling` selects the boundary derivative in graph parameter gradients. `detach=true`
passes values in both directions and suppresses every derivative crossing this edge.
The AD rules support ForwardDiff and ChainRules reverse mode. Fixed-reference replay
also defines the cuts for central differences. Unknown coupling types throw.
"""
struct Edge{D<:EdgeDirection,C<:AbstractGradientCoupling}
    factor::Int
    channel::Symbol
    variable::Int
    direction::D
    coupling::C
    detach::Bool
    function Edge{D,C}(factor, channel, variable, direction, coupling, detach = false) where
            {D<:EdgeDirection,C<:AbstractGradientCoupling}
        LenticulumCore._check_gradient_coupling(coupling)
        return new{D,C}(factor, channel, variable, direction, coupling, detach)
    end
end
function Edge(factor, channel, variable, direction::D, coupling::C, detach = false) where
        {D<:EdgeDirection,C<:AbstractGradientCoupling}
    return Edge{D,C}(factor, channel, variable, direction, coupling, detach)
end

"""
    FactorGraph

A bipartite graph of variables and factors with named, directed channel attachments.

Build one with [`GraphBuilder`](@ref); do not construct directly, since the adjacency
indices must stay consistent with the edge list.
"""
struct FactorGraph
    variables::Vector{VariableNode}
    factors::Vector{FactorNode}
    edges::Vector{Edge}
    edges_of_factor::Vector{Vector{Int}}
    edges_of_variable::Vector{Vector{Int}}
    variable_index::Dict{Symbol,Int}
    factor_index::Dict{Symbol,Int}
    plates::Vector{Any}
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
    plates::Vector{Any}
    ports::Set{Tuple{Int,Symbol}}     # connected (factor, channel) pairs, for the duplicate check
end
GraphBuilder() = GraphBuilder(
    VariableNode[], FactorNode[], Edge[], Dict{Symbol,Int}(), Dict{Symbol,Int}(), Any[],
    Set{Tuple{Int,Symbol}}()
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
    factor!(b::GraphBuilder, name::Symbol, factor; freeze = (), rule = BeliefPropagation(),
            params = name) -> Int

Add a factor node wrapping an `AbstractLenticulumFactor`; returns its id.
`freeze` is `:all`, a parameter path such as `(:encoder,)`, or a tuple of paths.
Paths select complete subtrees and are checked against the actual parameter tree.
`rule` is how the node computes its messages: [`BeliefPropagation`](@ref), [`MeanField`](@ref),
[`ExpectationPropagationRule`](@ref) or [`MixtureReduction`](@ref).
`params` is the node's [`parameter_key`](@ref): nodes given the same key share one entry of `ps`
and `st` (their gradients add), and must have the same factor type and `freeze`; [`build`](@ref)
checks.
"""
function factor!(b::GraphBuilder, name::Symbol, factor; freeze = (), rule::AbstractMessageRule = BeliefPropagation(),
                 params::Symbol = name)
    haskey(b.factor_index, name) && throw(ArgumentError("duplicate factor :$name"))
    id = length(b.factors) + 1
    LenticulumCore._freeze_paths(freeze) # validate syntax; paths checked against ps at use
    push!(b.factors, FactorNode(id, name, factor, freeze, rule, params))
    b.factor_index[name] = id
    return id
end

"""
    connect!(b, factor::Symbol, channel::Symbol, variable::Symbol;
             direction = Bidirectional(), coupling = DiagonalCoupling(), detach = false)

Attach one channel of a factor to a variable.

Each (factor, channel) pair may be connected at most once: a channel is a single port, not a
bus. Fanning a variable out to several factors is done by connecting *those factors*, which
is the graph-level form of the copier of `Copiers Cups and Caps.md`.

All four built-in couplings are accepted. A detached edge overrides its coupling.
Non-diagonal parameter gradients require an explicit static replay schedule. Support
restrictions are checked when gradients are requested, with explanatory ArgumentErrors.
"""
function connect!(
    b::GraphBuilder, factor::Symbol, channel::Symbol, variable::Symbol;
    direction::EdgeDirection = Bidirectional(),
    coupling = DiagonalCoupling(),
    detach::Bool = false,
)
    LenticulumCore._check_gradient_coupling(coupling)
    haskey(b.factor_index, factor) || throw(ArgumentError("no factor :$factor"))
    haskey(b.variable_index, variable) || throw(ArgumentError("no variable :$variable"))
    fid = b.factor_index[factor]
    vid = b.variable_index[variable]
    (fid, channel) in b.ports &&
        throw(ArgumentError("channel :$channel of factor :$factor is already connected"))
    push!(b.edges, Edge(fid, channel, vid, direction, coupling, detach))
    push!(b.ports, (fid, channel))
    return length(b.edges)
end

"""
    build(b::GraphBuilder) -> FactorGraph

Freeze the builder into an immutable graph with adjacency indices.

Frozen factors are accepted. Structural [`validate`](@ref) checks remain optional;
parameter freeze paths are checked against the parameter tree when it is supplied.
Tied nodes (one [`parameter_key`](@ref)) must agree in factor type and `freeze`, and a key may
not be the name of a node that reads another key; otherwise this throws.
"""
function build(b::GraphBuilder)
    _check_ties(b.factors)
    eof = [Int[] for _ in b.factors]
    eov = [Int[] for _ in b.variables]
    for (i, e) in enumerate(b.edges)
        push!(eof[e.factor], i)
        push!(eov[e.variable], i)
    end
    return FactorGraph(
        copy(b.variables), copy(b.factors), copy(b.edges), eof, eov,
        copy(b.variable_index), copy(b.factor_index), copy(b.plates),
    )
end

function _check_ties(factors)
    first_of = Dict{Symbol,FactorNode}()
    for f in factors
        r = get!(first_of, f.params, f)
        r === f && continue
        typeof(r.factor) == typeof(f.factor) || throw(ArgumentError(
            "factors :$(r.name) and :$(f.name) share parameters :$(f.params) but have different types \
             ($(nameof(typeof(r.factor))) and $(nameof(typeof(f.factor))))"))
        r.freeze == f.freeze || throw(ArgumentError(
            "factors :$(r.name) and :$(f.name) share parameters :$(f.params) but freeze different paths"))
    end
    names = Dict(f.name => f for f in factors)
    for h in factors
        f = get(names, h.params, nothing)
        f === nothing || f.params === h.params || throw(ArgumentError(
            "factor :$(h.name) reads parameters :$(h.params), the name of factor :$(f.name), which reads \
             :$(f.params); tie them or rename one"))
    end
    return nothing
end

"""
    parameter_keys(g) -> Tuple{Vararg{Symbol}}

The distinct [`parameter_key`](@ref)s of the factor nodes, in order of first appearance: the
top-level keys of `LuxCore.initialparameters(rng, g)`. Without ties these are the factor names.
"""
parameter_keys(g::FactorGraph) = Tuple(unique(f.params for f in g.factors))

"""
    tied_factors(g, key::Symbol) -> Vector{Int}

The ids of the factor nodes that read parameter entry `key`.
"""
tied_factors(g::FactorGraph, key::Symbol) = [f.id for f in g.factors if f.params === key]

# The ids of the nodes reading each key, in one pass: `tied_factors` per key scans all factors,
# O(keys × factors) over the keys (`passing.md`, "The trees by position").
function _key_groups(g::FactorGraph)
    groups = Dict{Symbol,Vector{Int}}()
    for f in g.factors
        push!(get!(() -> Int[], groups, f.params), f.id)
    end
    return groups
end

# The first node reading `key`: it stands for the tied group in freeze and learnability checks.
function _key_node(g::FactorGraph, key::Symbol)
    i = findfirst(f -> f.params === key, g.factors)
    i === nothing && throw(ArgumentError("no factor reads parameters :$key"))
    return g.factors[i]
end

_active_indices(f::FactorNode, ps) = _cached_indices(ps,
    !LenticulumCore.islearnable(f.factor) || LenticulumCore.isfrozen(f.factor) ? :all : f.freeze)

# The active coordinates depend on the freeze and the shape of the parameter tree only, so they
# are cached by both: a frozen node no longer rebuilds the offset tree of every coordinate on
# each message (`graph.md`). Shapes are the leaves' sizes, cheap to compute.
const _INDEX_CACHE = Dict{Any,Vector{Int}}()
const _INDEX_LOCK = ReentrantLock()
_shape(x::NamedTuple) = map(_shape, x)
_shape(x::Tuple) = map(_shape, x)
_shape(x::AbstractArray) = size(x)
_shape(x) = typeof(x)
function _cached_indices(ps, freeze)
    key = (freeze, _shape(ps))
    lock(_INDEX_LOCK) do
        get!(() -> LenticulumCore._trainable_indices(ps, freeze), _INDEX_CACHE, key)
    end
end
# Vectors, not tuples: a tuple of every node makes compile time grow with the graph (plates).
_factor_nodes(g) = g.factors
_variable_nodes(g) = g.variables

function _factor_parameters(f::FactorNode, ps)
    isempty(ps) && return ps
    # nothing frozen: every coordinate is active, without building the index of all of them on
    # every message (review of 2026-11-04)
    LenticulumCore.islearnable(f.factor) && !LenticulumCore.isfrozen(f) && return ps
    indices = _active_indices(f, ps)
    length(indices) == LenticulumCore._flatlen(ps) && return ps
    θ = LenticulumCore._flatten_active(ps)
    return LenticulumCore._unflatten(_pin_coordinates(θ, indices), ps)
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
    variable_degree(g, vid) -> Int
    factor_degree(g, fid) -> Int

The number of edges incident to a variable or a factor. For a variable this is the ``d_v`` of the
Bethe counting number ``1 - d_v``; see `free_energy.jl`.
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

# One entry per parameter key, initialised by the first node that reads it (tied nodes share it).
function LuxCore.initialparameters(rng::AbstractRNG, g::FactorGraph)
    ks = parameter_keys(g)
    vs = Tuple(LuxCore.initialparameters(rng, _key_node(g, k).factor) for k in ks)
    return NamedTuple{ks}(vs)
end

function LuxCore.initialstates(rng::AbstractRNG, g::FactorGraph)
    ks = parameter_keys(g)
    vs = Tuple(LuxCore.initialstates(rng, _key_node(g, k).factor) for k in ks)
    return NamedTuple{ks}(vs)
end

LuxCore.parameterlength(g::FactorGraph) =
    sum(k -> LuxCore.parameterlength(_key_node(g, k).factor), parameter_keys(g); init = 0)

function Base.show(io::IO, g::FactorGraph)
    print(io, "FactorGraph(", nvariables(g), " variables, ", nfactors(g), " factors, ",
        nedges(g), " edges; χ=", euler_characteristic(g),
        istree(g) ? ", tree" : euler_characteristic(g) == _ncomponents(g) ? ", forest" : ", loopy",
        isdag(g) ? ", DAG)" : ", cyclic)")
end

# The number of connected components (each a tree exactly when χ equals this number).
function _ncomponents(g::FactorGraph)
    nv = nvariables(g)
    parent = collect(1:(nv + nfactors(g)))
    root(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    for e in g.edges
        a, b = root(e.variable), root(nv + e.factor)
        a == b || (parent[a] = b)
    end
    return count(i -> root(i) == i, eachindex(parent))
end

# `i in indices` per coordinate searched the index vector, quadratic in the parameter count on every
# message; a mask, built outside AD (`_active_mask`), is linear (2026-11-04).
_active_mask(n::Int, indices) = (m = falses(n); m[indices] .= true; m)
function _pin_coordinates(θ, indices)
    mask = _active_mask(length(θ), indices)
    return [mask[i] ? θ[i] : LenticulumCore.stop_gradient(θ[i]) for i in eachindex(θ)]
end

# Whether a state tree holds nothing: tied factors and the implicit graph adjoint require it.
_empty_graph_state(x::Union{Tuple,NamedTuple}) = all(_empty_graph_state, x)
_empty_graph_state(x) = false
