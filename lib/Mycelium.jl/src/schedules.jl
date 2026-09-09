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
whole path), but order-independent, so the result does not depend on an arbitrary choice —
which matters when the graph has no natural order at all.
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
> `AbstractGradientCoupling` machinery, not by this scheduler. So `pruned > 0` does not mean
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
> backward pass carries *cotangents*, not beliefs, and cotangents are handled by the
> `AbstractGradientCoupling` on each edge together with the free-energy accumulation — not by
> this scheduler. Seeing `isempty(sched.backward)` is therefore a useful diagnostic: it says
> "this graph is explicit; there is nothing to infer".
"""
struct ForwardBackwardSchedule <: AbstractSchedule
    forward::Vector{MessageTask}
    backward::Vector{MessageTask}
end
tasks(s::ForwardBackwardSchedule) = vcat(s.forward, s.backward)
tasks(s::SequentialSchedule) = s.tasks
tasks(s::FloodingSchedule) = s.tasks

"""
    ResidualSchedule(; maxtasks)

Dynamic: repeatedly send whichever pending message would change the most (Elidan, McGraw &
Koller's *residual belief propagation*).

Converges on many loopy graphs where flooding oscillates, because it prioritises the parts
of the graph that have not settled. Requires a usable [`belief_distance`](@ref); with the
current belief types most pairs return `Inf`, so this degrades to "some order", which is
recorded honestly rather than hidden.
"""
struct ResidualSchedule <: AbstractSchedule
    maxtasks::Int
end
ResidualSchedule(; maxtasks::Int = 10_000) = ResidualSchedule(maxtasks)

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
    rootnode = root === nothing ? first(leaves(g)) : root
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
                f = g.edges[ei].factor
                seenf[f] && continue
                seenf[f] = true
                edge_child[ei] = :factor
                push!(edge_order, ei)
                push!(order, (:factor, f))
            end
        else
            for ei in g.edges_of_factor[id]
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

Errors unless [`isdag`](@ref) — which, note, excludes any graph containing a `Bidirectional`
edge. So in practice this schedule is for the *explicit* subset of graphs, and its
`backward` list is always empty. It is provided because that degenerate case is exactly
Lux.jl, and having it nameable makes the boundary between the two libraries explicit.
"""
function forward_backward_schedule(g::FactorGraph)
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
