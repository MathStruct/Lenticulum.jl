# ---------------------------------------------------------------------------
# Schedules: WHICH message, in WHAT order.
#
# In Lux there is one order and it is implied by the wiring, so "scheduling" is invisible.
# On a general graph it is the whole problem. Two atomic operations:
#
#   VariableToFactor(e)  -- pool the OTHER messages at the variable (exclusion) and hand the
#                           result to the factor.  Legal iff the edge absorbs.
#   FactorToVariable(e)  -- resolve a polarity with e's channel Unobserved, assemble the
#                           lens, invert.  Legal iff the edge emits.
#
# See `schedules.md` and `Schedules.md`.
# ---------------------------------------------------------------------------

"""
    MessageTask(edge, kind)

One atomic message. `kind` is `:to_factor` or `:to_variable`.
"""
struct MessageTask
    edge::Int
    kind::Symbol
end

to_factor_task(e::Int) = MessageTask(e, :to_factor)
to_variable_task(e::Int) = MessageTask(e, :to_variable)

"""
    abstract type AbstractSchedule

A plan for message passing. Subtypes differ in whether they are *static* (a fixed task list
computed from the graph) or *dynamic* (recomputed from message residuals each step).
"""
abstract type AbstractSchedule end

"""
    SequentialSchedule(tasks)

A fixed list, executed in order, **in place**: each task sees the results of the ones before
it in the same sweep. Faster-propagating than flooding, and the basis of tree and
forward schedules.
"""
struct SequentialSchedule <: AbstractSchedule
    tasks::Vector{MessageTask}
end

"""
    FloodingSchedule(tasks)

All tasks computed from the *previous* sweep's messages and committed together
(double-buffered). The classical parallel BP update.

Slower to propagate information than a sequential sweep (one edge per sweep rather than a
whole path), but message-order-independent for stateless factors. Lux state still
threads in task order, so stateful factors can retain order dependence.
"""
struct FloodingSchedule <: AbstractSchedule
    tasks::Vector{MessageTask}
end

"""
    TreeSchedule(inward, outward, pruned)

The two-sweep schedule for a tree: every edge carries one message towards the root, then one
away from it.

> **On a tree with only `Bidirectional` edges this is exact.** After the outward sweep,
> `marginal(store, g, v)` is the true marginal for every `v`. This is the only schedule here
> with a correctness guarantee.

`pruned` counts the messages the *edge directions* forbade. A `Emitting` edge cannot carry a
variable → factor message and an `Absorbing` edge cannot carry a factor → variable one, so a
graph containing unidirectional edges has fewer than `2|E|` legal messages and the sweeps are
correspondingly shorter.

> [!warning] Pruning weakens the exactness guarantee, and deliberately so
> A `LossFactor` on `Absorbing` edges never tells its input variable anything: in *belief*
> terms a sink is genuinely uninformative about what it consumes. What a loss "tells" its
> input is a **cotangent**, not a belief, and cotangents travel by the
> `AbstractGradientCoupling` machinery in the parameter driver, with explicit static
> schedule replay for supported non-diagonal couplings (LP-011, L-SPEC-002). So `pruned > 0` does not mean
> the schedule is broken — it means part of the graph is explicit rather than implicit. It is
> surfaced as a field rather than swallowed so that a genuinely mis-wired graph (an implicit
> factor accidentally given a unidirectional edge) is visible.
"""
struct TreeSchedule <: AbstractSchedule
    inward::Vector{MessageTask}
    outward::Vector{MessageTask}
    pruned::Int
end
TreeSchedule(inward, outward) = TreeSchedule(inward, outward, 0)
tasks(s::TreeSchedule) = vcat(s.inward, s.outward)

"""
    ForwardBackwardSchedule(forward, backward)

For a graph that is a DAG under its edge directions: a topological belief sweep, then a
reverse sweep along whatever edges are `Bidirectional`.

> [!note] The backward list is empty for a strictly unidirectional DAG — and that is correct
> A Lux-style graph (all edges `Emitting`/`Absorbing`) has **no backward belief flow**. Its
> backward pass would carry *cotangents*, not beliefs. The per-edge
> parameter-driver couplings supply cotangent paths separately from these belief tasks
> (LP-011, L-SPEC-002). Seeing `isempty(sched.backward)` says
> "this graph is explicit; there is nothing to infer".
"""
struct ForwardBackwardSchedule <: AbstractSchedule
    forward::Vector{MessageTask}
    backward::Vector{MessageTask}
end
tasks(s::ForwardBackwardSchedule) = vcat(s.forward, s.backward)
tasks(s::SequentialSchedule) = s.tasks
tasks(s::FloodingSchedule) = s.tasks

# The indexed max-heap and the inputs it describes (`passing.jl`). A sweep can leave a
# few keys dirty at its last commit. Its snapshot contains message references, never the store.
mutable struct _TaskHeap{R<:Real}
    res::Vector{R}
    age::Vector{Int}
    heap::Vector{Int}
    pos::Vector{Int}
    nans::Int
end

struct _ResidualMemory
    graph::FactorGraph
    ps::Any
    st::Any
    damping::Real
    relative_tol::Union{Nothing,NTuple{2,Float64}}
    maxtasks::Int
    pins::Dict{Int,Any}
    to_variable::Vector{Any}
    to_factor::Vector{Any}
    heap::_TaskHeap
    dirty::Vector{Int}
end

"""
    ResidualSchedule(; maxtasks = 10_000, reuse = true)

Dynamic: repeatedly send whichever pending message would change the most (Elidan, McGraw &
Koller's *residual belief propagation*).

Converges on many loopy graphs where flooding oscillates, because it prioritises the parts
of the graph that have not settled. A `sweep!` commits up to `min(maxtasks, number of tasks)`
times the task of largest residual (the oldest message among equal residuals, then task order)
and re-evaluates only the tasks whose message reads the committed one (I4: a priority queue, as in Elidan,
McGraw & Koller). That is ``O(|\\text{tasks}| \\cdot \\text{degree})`` message evaluations per
sweep, and the same commits a scan of every task before each commit would make. Requires a
usable [`belief_distance`](@ref); pairs returning `Inf` are tried first.

With `reuse = true`, graphs whose factors all declare [`repeatable_messages`](@ref) keep the
queue per store across sweeps. The snapshot detects replacement of a message, the parameter or
state tree, and changes to pins, damping and relative tolerance. Use `reuse = false` if changing
factor data, parameters, state or stored beliefs in place between sweeps. Each sweep then starts
by evaluating every task's residual.
"""
struct ResidualSchedule <: AbstractSchedule
    maxtasks::Int
    reuse::Bool
    memory::WeakKeyDict{MessageStore,Union{Nothing,_ResidualMemory}}
    function ResidualSchedule(maxtasks::Int; reuse::Bool = true)
        # A sweep of zero tasks computes nothing and has residual 0, which would read as
        # convergence (LP-019 item 9).
        maxtasks >= 1 || throw(ArgumentError("ResidualSchedule needs maxtasks ≥ 1; got $maxtasks"))
        return new(maxtasks, reuse, WeakKeyDict{MessageStore,Union{Nothing,_ResidualMemory}}())
    end
end
ResidualSchedule(; maxtasks::Int = 10_000, reuse::Bool = true) = ResidualSchedule(maxtasks; reuse)

# --- Generators ------------------------------------------------------------

"""
    islegal(g, task) -> Bool

Whether the edge's direction permits this message. A `:to_factor` task needs an absorbing
edge; a `:to_variable` task needs an emitting one.

Schedules filter on this rather than throwing, because a unidirectional edge is a legitimate
modelling choice, not an error — see [`TreeSchedule`](@ref).
"""
islegal(g::FactorGraph, t::MessageTask) =
    t.kind === :to_factor ? can_absorb(g, t.edge) : can_emit(g, t.edge)

"""
    all_tasks(g) -> Vector{MessageTask}

Every message the edge directions permit: one `:to_factor` per absorbing edge, one
`:to_variable` per emitting edge. A bidirectional edge yields both.
"""
function all_tasks(g::FactorGraph)
    out = MessageTask[]
    for ei in 1:nedges(g)
        can_absorb(g, ei) && push!(out, to_factor_task(ei))
        can_emit(g, ei) && push!(out, to_variable_task(ei))
    end
    return out
end

flooding_schedule(g::FactorGraph) = FloodingSchedule(all_tasks(g))

function _resolve_tree_root(g::FactorGraph, root)
    if root === nothing
        lvs = leaves(g)
        isempty(lvs) || return first(lvs)
        return nvariables(g) > 0 ? (:variable, 1) : (:factor, 1)
    elseif root isa Tuple{Symbol,Int}
        return root
    elseif root isa Symbol
        haskey(g.variable_index, root) && return (:variable, g.variable_index[root])
        haskey(g.factor_index, root) && return (:factor, g.factor_index[root])
        throw(ArgumentError("no variable or factor named :$root in graph"))
    end
    throw(ArgumentError("unsupported root specification: $root"))
end

"""
    tree_schedule(g; root = nothing) -> TreeSchedule

Root the tree (at `root`, or at the first leaf found) and produce the inward/outward sweeps.

Errors unless [`istree`](@ref). The inward sweep is in reverse breadth-first order (children
strictly before parents) and the outward sweep in breadth-first order, which is what makes
each message computable exactly once from already-final inputs.
"""
function tree_schedule(g::FactorGraph; root = nothing)
    istree(g) || throw(ArgumentError(
        "tree_schedule requires a tree (connected, χ = 1); this graph has χ = \
         $(euler_characteristic(g))$(isconnected(g) ? "" : " and is disconnected"). \
         Use flooding_schedule or a ResidualSchedule."))
    return _tree_sweeps(g, _resolve_tree_root(g, root))
end

# Components and their BFS edges, independent of message legality.
function _forest_components(g, rootnode; edge_in = _ -> true)
    seen = falses(nvariables(g) + nfactors(g))
    roots = Tuple{Symbol,Int}[]
    forest = falses(nedges(g))
    nodes = vcat([rootnode], [(:variable, v) for v in 1:nvariables(g)],
                 [(:factor, f) for f in 1:nfactors(g)])
    nodeid(kind, id) = kind === :variable ? id : nvariables(g) + id
    for node in nodes
        seen[nodeid(node...)] && continue
        push!(roots, node)
        queue = [node]; seen[nodeid(node...)] = true
        i = 1
        while i <= length(queue)
            kind, id = queue[i]; i += 1
            edges = kind === :variable ? g.edges_of_variable[id] : g.edges_of_factor[id]
            for ei in edges
                edge_in(ei) || continue
                e = g.edges[ei]
                next = kind === :variable ? (:factor, e.factor) : (:variable, e.variable)
                seen[nodeid(next...)] && continue
                seen[nodeid(next...)] = true
                forest[ei] = true
                push!(queue, next)
            end
        end
    end
    return roots, forest
end

function _spanning_forests(g::FactorGraph; root = nothing, weights = nothing, ntrees = 1)
    (ntrees isa Int && ntrees > 0 || ntrees === :cover) ||
        throw(ArgumentError("ntrees must be a positive Int or :cover"))
    (weights === nothing || weights isa AbstractDict || applicable(weights, 1)) ||
        throw(ArgumentError("weights must be nothing, a dictionary keyed by factor name, or a callable on factor indices"))
    names = Dict(fi => name for (name, fi) in g.factor_index)
    fw = [weights === nothing ? 0 : weights isa AbstractDict ?
          (haskey(weights, names[fi]) ? weights[names[fi]] :
           throw(ArgumentError("missing weight for factor $(names[fi])"))) : weights(fi)
          for fi in 1:nfactors(g)]
    all(w -> w isa Real && isfinite(w), fw) ||
        throw(ArgumentError("factor weights must be finite real numbers"))
    ew = [fw[e.factor] for e in g.edges]
    forests = BitVector[]
    uses = zeros(Int, nedges(g))
    while isempty(forests) || (ntrees === :cover ? any(iszero, uses) : length(forests) < ntrees)
        if isempty(forests) && weights === nothing
            _, tree = _forest_components(g, _resolve_tree_root(g, root))
        else
            parent = collect(1:(nvariables(g) + nfactors(g)))
            function findroot(i)
                while parent[i] != i
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                end
                return i
            end
            tree = falses(nedges(g))
            edge_less(a, b) = uses[a] != uses[b] ? uses[a] < uses[b] :
                ew[a] != ew[b] ? ew[a] > ew[b] : a < b
            for ei in sortperm(1:nedges(g); lt = edge_less)
                e = g.edges[ei]
                a, b = findroot(e.variable), findroot(nvariables(g) + e.factor)
                a == b && continue
                parent[a] = b
                tree[ei] = true
            end
        end
        push!(forests, tree)
        uses .+= tree
    end
    return forests
end

"""
    spanning_tree_schedule(g; root = nothing, weights = nothing, ntrees = 1) -> SequentialSchedule

A static schedule for a **loopy** graph: the messages on the edges outside a spanning tree, then
the tree's inward and outward sweeps, all in place. One sweep carries information across the whole
tree, so on a graph with one loop a message goes once round the loop each way per sweep, where
flooding moves it one edge.

With `weights = nothing`, tree 1 is the breadth-first tree from `root` (resolved as in
[`tree_schedule`](@ref)). Otherwise `weights` is a callable on factor indices or a dictionary
keyed by factor name, with finite real values for every factor. Kruskal selects a maximum-weight
spanning forest; each edge inherits its factor's weight, with ties broken by lower edge index.
Each component is rooted at `root` when present, then its lowest-index variable or factor.

`ntrees` is a positive `Int` or `:cover`. Later forests prefer edges used fewer times, then
higher weight (zero without weights), then lower edge index. `:cover` stops once every edge
belongs to a forest. Tasks in no forest come first, followed by each forest's inward tasks of
all components and then its outward tasks. Repeated tasks are evaluated once per tree, so a
sweep with K trees costs approximately K tree sweeps. On a tree, one forest gives
`tree_schedule`'s task order. This is a static tree-based reparameterisation order
(Wainwright, Jaakkola & Willsky, 2003), without tree appearance weights or a modified objective.

There is no convergence guarantee on a loopy graph, and the fixed points are flooding's (a
schedule changes the map, not its fixed points). Measured on the resistive divider (one loop of
six edges; `bench/bp_schedules.jl`, `vault/Gaussian Graphs/Composing Optimisers.md` §4.2), it needs 0.17 to
0.18 of flooding's message evaluations to an error of ``10^{-10}``. For graphs with
[`MeanField`](@ref) nodes use [`hybrid_schedule`](@ref), which treats them separately.

Measured in `bench/bp_trees.jl`: on the 10 × 10 horizontal-strong grid, precision weights
with Anderson(3) used 147200 message evaluations against BFS with Anderson(3)'s 792120;
on the seeded random-noise 10 × 10 grid, the corresponding figures were 67160 and 196880.
Two precision-weighted trees with Anderson(3) used 191016 and 86412 evaluations respectively,
despite taking fewer sweeps than one tree in both settings.
"""
function spanning_tree_schedule(g::FactorGraph; root = nothing, weights = nothing, ntrees = 1)
    forests = _spanning_forests(g; root, weights, ntrees)
    tree_tasks = MessageTask[]
    for forest in forests
        roots, _ = _forest_components(g, _resolve_tree_root(g, root); edge_in = ei -> forest[ei])
        inward = MessageTask[]; outward = MessageTask[]
        for rootnode in roots
            ts = _tree_sweeps(g, rootnode; edge_in = ei -> forest[ei])
            append!(inward, ts.inward); append!(outward, ts.outward)
        end
        append!(tree_tasks, inward); append!(tree_tasks, outward)
    end
    intree = Set(tree_tasks)
    offtree = [t for t in all_tasks(g) if !(t in intree)]
    return SequentialSchedule(vcat(offtree, tree_tasks))
end

# The inward and outward sweeps of the component containing `rootnode`, which must be a tree
# (`tree_schedule` checks the whole graph; `plate_schedule` checks one component).
function _tree_sweeps(g::FactorGraph, rootnode; skip = _ -> false, edge_in = _ -> true)
    # breadth-first from the root, recording for each edge which endpoint is the child
    order = Tuple{Symbol,Int}[rootnode]
    seenv = falses(nvariables(g)); seenf = falses(nfactors(g))
    rootnode[1] === :variable ? (seenv[rootnode[2]] = true) : (seenf[rootnode[2]] = true)
    edge_child = Dict{Int,Symbol}()   # edge -> :variable or :factor (which side is the child)
    edge_order = Int[]
    i = 1
    while i <= length(order)
        kind, id = order[i]; i += 1
        if kind === :variable
            for ei in g.edges_of_variable[id]
                edge_in(ei) || continue
                f = g.edges[ei].factor
                (seenf[f] || skip(f)) && continue
                seenf[f] = true
                edge_child[ei] = :factor
                push!(edge_order, ei)
                push!(order, (:factor, f))
            end
        else
            for ei in g.edges_of_factor[id]
                edge_in(ei) || continue
                v = g.edges[ei].variable
                seenv[v] && continue
                seenv[v] = true
                edge_child[ei] = :variable
                push!(edge_order, ei)
                push!(order, (:variable, v))
            end
        end
    end
    inward = MessageTask[]
    outward = MessageTask[]
    pruned = 0
    for ei in reverse(edge_order)          # children first
        t = edge_child[ei] === :factor ? to_variable_task(ei) : to_factor_task(ei)
        islegal(g, t) ? push!(inward, t) : (pruned += 1)
    end
    for ei in edge_order                   # parents first
        t = edge_child[ei] === :factor ? to_factor_task(ei) : to_variable_task(ei)
        islegal(g, t) ? push!(outward, t) : (pruned += 1)
    end
    return TreeSchedule(inward, outward, pruned)
end

"""
    forward_backward_schedule(g) -> ForwardBackwardSchedule

Topological belief sweep, then the reverse sweep over bidirectional edges only.

Accepts a DAG or a tree. [`isdag`](@ref) excludes `Bidirectional` edges, so the DAG
branch's `backward` list is empty. On a tree that is not a DAG, wraps the inward and
outward lists from [`tree_schedule`](@ref). Other graphs raise `ArgumentError`.
"""
function forward_backward_schedule(g::FactorGraph)
    if !isdag(g)
        if istree(g)
            # on a tree the exact order is the tree schedule's: inward, then outward
            ts = tree_schedule(g)
            return ForwardBackwardSchedule(ts.inward, ts.outward)
        end
        throw(ArgumentError(
            "graph is neither a DAG nor a tree; use flooding_schedule or ResidualSchedule."))
    end
    order = topological_order(g)
    forward = MessageTask[]
    for (kind, id) in order
        kind === :factor || continue
        for ei in g.edges_of_factor[id]
            can_absorb(g, ei) && push!(forward, to_factor_task(ei))
        end
        for ei in g.edges_of_factor[id]
            can_emit(g, ei) && push!(forward, to_variable_task(ei))
        end
    end
    backward = MessageTask[]
    for (kind, id) in reverse(order)
        kind === :factor || continue
        for ei in g.edges_of_factor[id]
            g.edges[ei].direction isa Bidirectional || continue
            push!(backward, to_factor_task(ei))
            push!(backward, to_variable_task(ei))
        end
    end
    return ForwardBackwardSchedule(forward, backward)
end

# --- Hybrid graphs (Extending Mycelium §3) ------------------------------------

"""
    HybridSchedule(tasks)

A static schedule for a graph with [`MeanField`](@ref) nodes, built by [`hybrid_schedule`](@ref).
It sweeps like a [`SequentialSchedule`](@ref) of `tasks`; [`propagate!`](@ref) measures its
convergence on the variables' marginals by default (`criterion = :marginals`).
"""
struct HybridSchedule <: AbstractSchedule
    tasks::Vector{MessageTask}
end
tasks(s::HybridSchedule) = s.tasks

"""
    hybrid_schedule(g) -> HybridSchedule

The schedule of the hybrid message passing of Riegler et al. (2013) for a graph whose
belief-propagation part (the graph with its [`MeanField`](@ref) nodes removed) is a forest. A
sweep is:

1. every belief-propagation cluster (a component of that forest) swept inward and outward, as
   [`tree_schedule`](@ref) would, with the mean-field messages already at its variables held fixed;
2. each mean-field node in turn: its message to each variable it can emit to, then that
   variable's cluster swept again, so the next message reads the updated marginals.

Within a cluster the beliefs are then exact given the mean-field messages, and each mean-field
update is followed by the exact cluster posterior: block coordinate descent on the merged free
energy when the mean-field node's other variables are in other clusters. On a fresh store the
first pass of step 1 starts every cluster from its own factors (priors, data). Throws when a
cluster has a loop. Without mean-field nodes it is the tree schedule of each component.
"""
hybrid_schedule(g::FactorGraph) = HybridSchedule(_hybrid_tasks(g, 1:nvariables(g)))

# The hybrid sweep restricted to the variables `vars` (a union of components of the graph) and the
# mean-field nodes on them.
function _hybrid_tasks(g::FactorGraph, vars)
    mf(f) = g.factors[f].rule isa MeanField
    cluster_of = zeros(Int, nvariables(g))
    sweeps = Vector{Vector{MessageTask}}()
    for v in vars
        cluster_of[v] == 0 || continue
        nodes = _cluster(g, (:variable, v), mf)
        nvs = [id for (kind, id) in nodes if kind === :variable]
        nfs = [id for (kind, id) in nodes if kind === :factor]
        ne = sum(f -> factor_degree(g, f), nfs; init = 0)
        ne == length(nodes) - 1 || throw(ArgumentError(
            "the belief-propagation cluster of variable :$(g.variables[v].name) has a loop; \
             hybrid_schedule builds tree sweeps only (write a SequentialSchedule by hand)"))
        foreach(u -> cluster_of[u] = length(sweeps) + 1, nvs)
        deg(n) = n[1] === :variable ? count(ei -> !mf(g.edges[ei].factor), g.edges_of_variable[n[2]]) :
                                      factor_degree(g, n[2])
        root = nodes[something(findfirst(n -> deg(n) == 1, nodes), 1)]
        ts = _tree_sweeps(g, root; skip = mf)
        push!(sweeps, vcat(ts.inward, ts.outward))
    end
    out = reduce(vcat, sweeps; init = MessageTask[])
    for f in g.factors
        f.rule isa MeanField || continue
        all(ei -> cluster_of[g.edges[ei].variable] != 0, g.edges_of_factor[f.id]) || continue
        for ei in g.edges_of_factor[f.id]
            t = to_variable_task(ei)
            islegal(g, t) || continue
            push!(out, t)
            append!(out, sweeps[cluster_of[g.edges[ei].variable]])
        end
    end
    return out
end

# The nodes reachable from `node` without passing through a factor for which `skip` holds.
function _cluster(g::FactorGraph, node, skip)
    out = [node]; seen = Set([node]); i = 1
    while i <= length(out)
        kind, id = out[i]; i += 1
        nbrs = kind === :variable ?
            [(:factor, g.edges[e].factor) for e in g.edges_of_variable[id] if !skip(g.edges[e].factor)] :
            [(:variable, g.edges[e].variable) for e in g.edges_of_factor[id]]
        for m in nbrs
            m in seen && continue
            push!(seen, m); push!(out, m)
        end
    end
    return out
end

_default_criterion(::AbstractSchedule) = :messages
_default_criterion(::HybridSchedule) = :marginals
