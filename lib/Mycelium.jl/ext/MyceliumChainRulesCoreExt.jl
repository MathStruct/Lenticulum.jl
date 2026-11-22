module MyceliumChainRulesCoreExt
using Mycelium, LenticulumCore, ChainRulesCore
ChainRulesCore.@non_differentiable Mycelium._index_metadata(θ, indices)
ChainRulesCore.@non_differentiable Mycelium._active_indices(f, ps)
ChainRulesCore.@non_differentiable Mycelium._active_mask(n, indices)
ChainRulesCore.@non_differentiable LenticulumCore.isfrozen(f::Mycelium.FactorNode)
ChainRulesCore.@non_differentiable Mycelium._factor_nodes(g)
ChainRulesCore.@non_differentiable Mycelium._cluster_layout(c, i, F, b)
ChainRulesCore.@non_differentiable Mycelium._batch_layout(c, i, F, B)
ChainRulesCore.@non_differentiable Mycelium.as_cluster(f)
ChainRulesCore.@non_differentiable Mycelium._as_clusters(fs)
ChainRulesCore.@non_differentiable Mycelium.cluster_form(f::Mycelium.TableFactor, ps)    # a constant table
ChainRulesCore.@non_differentiable Mycelium._variable_nodes(g)
ChainRulesCore.@non_differentiable Mycelium.factornode(g, id)
ChainRulesCore.@non_differentiable Mycelium.parameter_keys(g)
ChainRulesCore.@non_differentiable Mycelium.tied_factors(g, key)
ChainRulesCore.@non_differentiable Mycelium._key_node(g, key)
ChainRulesCore.@non_differentiable Mycelium._check_tied_state(g, st)
ChainRulesCore.@non_differentiable Mycelium._energy_rows(s, g)
ChainRulesCore.@non_differentiable Mycelium._has_rows(s, g)
ChainRulesCore.@non_differentiable Mycelium._mark!(mask, ids)     # bookkeeping of the row paths (P20)
# A plate row's state write-back (P24): the primal writes it (with `push!` for a key without one, a
# mutation Zygote's reverse pass refuses); no tangent flows through it, as a row's state is empty or
# tied and stateless (`_group_trees`, `_check_tied_state`), as the row rule below assumes too.
ChainRulesCore.@non_differentiable Mycelium._group_writeback(st, fn, keys, stf0, stf)
# The variable rows' entropies read only the store's message blocks, never `ps`; a store built
# inside AD from message vectors has no block rows (`blocks.md`), so no gradient is lost (P20).
ChainRulesCore.@non_differentiable Mycelium._variable_terms_rows(s, g)
ChainRulesCore.@non_differentiable Mycelium._keypositions(tree)     # the trees by position (`passing.md`)
ChainRulesCore.@non_differentiable Mycelium._slot(t, name)
ChainRulesCore.@non_differentiable Mycelium._state_writes(g, results)
ChainRulesCore.@non_differentiable Mycelium._key_groups(g)

# Each node's parameter subtree in one pass (I16, `passing.md` "The backward pass by key"): the
# nodes' cotangents are added per key (tied nodes share one) and the tree's cotangent is built once,
# O(nodes + keys), where a pullback per read gave a cotangent with every key per node.
ChainRulesCore.rrule(::typeof(Mycelium._subtrees), ps::NamedTuple, nodes) = _by_key_rrule(ps, nodes, NamedTuple())
ChainRulesCore.rrule(::typeof(Mycelium._states), st::NamedTuple, nodes) = _by_key_rrule(st, nodes, nothing)
function ChainRulesCore.rrule(::typeof(Mycelium._states), st::NamedTuple, nodes, absent)
    y, back = _by_key_rrule(st, nodes, absent)
    return y, Δ -> (back(Δ)..., ChainRulesCore.NoTangent())
end
# I30, step 4: the same reads on a boxed tree. The rule calls `_by_key_rrule` on the unboxed tree
# behind a function barrier, so that the pullback closure's captures are inferred as `Any`, not as
# the tree's type, and the rule compiles once for all structures; the cotangent is reboxed as
# `Tangent{_Erased}(; tree = Δ)`.
function ChainRulesCore.rrule(::typeof(Mycelium._subtrees), ps::LenticulumCore._Erased, nodes)
    y, back = _erased_by_key_barrier(ps.tree, nodes, NamedTuple(), Mycelium._subtrees)
    return y, LenticulumCore._ErasedPullback(Δ -> (ChainRulesCore.NoTangent(), _erased_tangent(back(unthunk(Δ))[2]), ChainRulesCore.NoTangent()))
end
function ChainRulesCore.rrule(::typeof(Mycelium._states), st::LenticulumCore._Erased, nodes)
    y, back = _erased_by_key_barrier(st.tree, nodes, nothing, (t, n) -> Mycelium._states(t, n))
    return y, LenticulumCore._ErasedPullback(Δ -> (ChainRulesCore.NoTangent(), _erased_tangent(back(unthunk(Δ))[2]), ChainRulesCore.NoTangent()))
end
function ChainRulesCore.rrule(::typeof(Mycelium._states), st::LenticulumCore._Erased, nodes, absent)
    y, back = _erased_by_key_barrier(st.tree, nodes, absent, (t, n) -> Mycelium._states(t, n, absent))
    return y, LenticulumCore._ErasedPullback(Δ -> (ChainRulesCore.NoTangent(), _erased_tangent(back(unthunk(Δ))[2]), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()))
end
# A tree that is not a `NamedTuple` (`nothing` for a stateless graph) has no rule by key: `read` gives the
# primal's value (`_subtrees` or `_states`, as the caller is), and its cotangent is zero (GLM's review of I30:
# the fallback read `_subtrees` for `_states` too).
@noinline function _erased_by_key_barrier(@nospecialize(tree), nodes, absent, read)
    tree isa NamedTuple || return (read(tree, nodes), _ -> (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.NoTangent()))
    return _by_key_rrule(tree, nodes, absent)
end
_erased_tangent(Δ::ChainRulesCore.AbstractZero) = Δ
_erased_tangent(@nospecialize(Δ)) = ChainRulesCore.Tangent{LenticulumCore._Erased}(; tree = Δ)
# The reads by position, compiled for any `NamedTuple` (I17): specialised on a tree of one field per
# factor, `getfield(ps, i)` with a run-time `i` cost O(keys) per read, so the rule's forward pass was
# O(nodes × keys) (10.7 MB at 241 keys and 722 nodes, against 0.1 MB for the primal; `coupled_scaling.jl`).
# `@noinline` as well: a `@nospecialize` helper alone, or a read through `_IndexedTree`, was inlined
# into the specialised caller and changed nothing.
@noinline _read_slots(@nospecialize(ps::NamedTuple), slots, absent) = Any[i == 0 ? absent : getfield(ps, i) for i in slots]
function _by_key_rrule(ps::NamedTuple, nodes, absent)
    pos = Mycelium._keypositions(ps)
    slots = Int[get(pos, f.params, 0) for f in nodes]
    y = _read_slots(ps, slots, absent)
    function subtrees_pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        nt = ChainRulesCore.NoTangent()
        _iszero(Δ) && return (nt, ChainRulesCore.ZeroTangent(), nt)
        acc = Vector{Any}(nothing, length(ps))
        for (j, i) in enumerate(slots)
            i == 0 && continue
            acc[i] = _acc(acc[i], ChainRulesCore.unthunk(Δ[j]))
        end
        all(isnothing, acc) && return (nt, ChainRulesCore.ZeroTangent(), nt)
        vals = map(a -> a === nothing ? ChainRulesCore.ZeroTangent() : a, acc)
        back = NamedTuple{keys(ps)}(Tuple(vals))
        return (nt, ChainRulesCore.Tangent{typeof(ps),typeof(back)}(back), nt)
    end
    return y, subtrees_pullback
end
ChainRulesCore.@non_differentiable Mycelium._node_reads(g, nodes)
ChainRulesCore.@non_differentiable Mycelium._variable_reads_of(g, vs)
ChainRulesCore.@non_differentiable Mycelium._remaining_variables(g, done)
ChainRulesCore.@non_differentiable Mycelium._slice_slot(edges, ei)
ChainRulesCore.@non_differentiable Mycelium._slot_of(v, x)
ChainRulesCore.@non_differentiable Mycelium._has_pins(s)
ChainRulesCore.@non_differentiable Mycelium._no_pins(reads)

# A sum from the left (`_sum_onto`): every term and the start get the sum's cotangent.
function ChainRulesCore.rrule(::typeof(Mycelium._sum_onto), acc, terms)
    sum_pullback(Δ) = (ChainRulesCore.NoTangent(), ChainRulesCore.unthunk(Δ), _fill_like(terms, ChainRulesCore.unthunk(Δ)))
    return Mycelium._sum_onto(acc, terms), sum_pullback
end
_fill_like(terms, Δ) = _iszero(Δ) ? ChainRulesCore.ZeroTangent() : Any[Δ for _ in terms]

# The store's messages read once, per node a pair of slices (I16, `free_energy.jl`): each node's
# cotangent of the messages it read is added into the store's cotangent by edge, O(edges + reads),
# where a pullback per read gave a cotangent as long as the store. The pins are read outside.
ChainRulesCore.rrule(::typeof(Mycelium._local_messages), s::Mycelium.MessageStore, reads) =
    _local_messages_rrule(s, reads, (; pins = ChainRulesCore.NoTangent()))
function _local_messages_rrule(s, reads, extra)
    y = Mycelium._local_messages(s, reads)
    nt = ChainRulesCore.NoTangent()
    function local_messages_pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        _iszero(Δ) && return (nt, ChainRulesCore.ZeroTangent(), nt)
        dtv = Vector{Any}(nothing, length(s.to_variable)); dtf = Vector{Any}(nothing, length(s.to_factor))
        for (j, (tf, tv, _)) in enumerate(reads)
            dj = ChainRulesCore.unthunk(Δ[j])
            _iszero(dj) && continue
            _scatter!(dtv, tv, ChainRulesCore.unthunk(dj[1]))
            _scatter!(dtf, tf, ChainRulesCore.unthunk(dj[2]))
        end
        all(isnothing, dtv) && all(isnothing, dtf) && return (nt, ChainRulesCore.ZeroTangent(), nt)
        fill0(d) = all(isnothing, d) ? ChainRulesCore.ZeroTangent() : d
        back = (; to_variable = fill0(dtv), to_factor = fill0(dtf), iteration = nt, extra...)
        return (nt, ChainRulesCore.Tangent{typeof(s),typeof(back)}(back), nt)
    end
    return y, local_messages_pullback
end
_field(Δ, name) = _iszero(Δ) ? nothing : Δ isa NamedTuple ? get(Δ, name, nothing) : getproperty(Δ, name)
# A slice's cotangent: its `msgs`, one entry per edge of `edges`, added into `d` by edge.
function _scatter!(d, edges, Δslice)
    _iszero(Δslice) && return d
    dm = ChainRulesCore.unthunk(_field(ChainRulesCore.unthunk(Δslice), :msgs))
    _iszero(dm) && return d
    for (j, e) in enumerate(edges)
        d[e] = _acc(d[e], ChainRulesCore.unthunk(dm[j]))
    end
    return d
end

# The sum of two cotangents of one subtree or message, in whichever form the AD hands them over: a
# structural `Tangent` is read as its `NamedTuple`, and `NamedTuple`s that name different fields
# (Zygote leaves out fields it has no cotangent for) are added field by field over the union.
_iszero(x) = x === nothing || x isa ChainRulesCore.AbstractZero
function _acc(a, b)
    a, b = ChainRulesCore.unthunk(a), ChainRulesCore.unthunk(b)
    _iszero(b) && return a
    _iszero(a) && return b
    a, b = _structural(a), _structural(b)
    if a isa NamedTuple && b isa NamedTuple
        ks = (keys(a)..., filter(k -> !haskey(a, k), keys(b))...)
        return NamedTuple{ks}(map(k -> _acc(get(a, k, nothing), get(b, k, nothing)), ks))
    end
    a isa Tuple && b isa Tuple && return map(_acc, a, b)
    return a + b
end
_structural(x::ChainRulesCore.Tangent) = (y = ChainRulesCore.backing(x); y isa NamedTuple || y isa Tuple ? y : x)
_structural(x) = x

# A store built from message vectors inside AD (`_functional_infer`) wraps them in `EdgeMessages`
# without block rows; its gradient is that of the wrapped vector (`blocks.md`). Indexing such a
# store reads `msgs` directly, and `similar` of it is a plain `Vector`, so the tangent of the
# wrapper is either that vector or a structural tangent holding it.
function ChainRulesCore.rrule(::Type{Mycelium.EdgeMessages}, v::AbstractVector)
    m = Mycelium.EdgeMessages(v)
    pullback(Δ) = (ChainRulesCore.NoTangent(), _edge_tangent(ChainRulesCore.unthunk(Δ)))
    return m, pullback
end
_edge_tangent(Δ::AbstractArray) = Δ
_edge_tangent(Δ::ChainRulesCore.Tangent) = Δ.msgs
_edge_tangent(Δ::NamedTuple) = Δ.msgs
_edge_tangent(Δ) = Δ
ChainRulesCore.@non_differentiable Mycelium.resolve_polarity(g, fid, target, available)
ChainRulesCore.@non_differentiable Mycelium.check_legal(g, fid, p)
ChainRulesCore.@non_differentiable Mycelium.MessageStore(g::FactorGraph)
ChainRulesCore.@non_differentiable Mycelium._check_energy_edges(g)
function ChainRulesCore.rrule(::typeof(available_channels), s, g, fid; kwargs...)
    return available_channels(s,g,fid;kwargs...), _ -> (NoTangent(),NoTangent(),NoTangent(),NoTangent())
end

# A plate row's scalar free energy (P20). The primal is the batched `local_free_energies`; the factors'
# batched code mutates, so the reverse pass runs the caller's AD through the same energy taken replica
# by replica (`_columns_scalar`, the per-node `local_free_energy` Zygote already takes on a store
# without rows). Both give the same value (Lenticulum `test/plates.jl`), so this is the derivative of
# the free energy at fixed messages, as ForwardDiff computes it, not the envelope gradient of
# `free_energy_gradient` (LP-020). The columns are cut outside AD; no tangent reaches the messages,
# the factors or the state: the store's blocks are constants of the free energy in `ps` (a store
# built inside AD has no block rows), and a row's state is empty (`_group_trees`, `_check_tied_state`).
function ChainRulesCore.rrule(cfg::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
                              ::typeof(Mycelium._row_scalar_energy), factors, msgs, ps, st)
    y = Mycelium._row_scalar_energy(factors, msgs, ps, st)
    cols = [Mycelium._column(msgs, i) for i in eachindex(factors)]
    _, back = ChainRulesCore.rrule_via_ad(cfg, Mycelium._columns_scalar, factors, cols, ps, st)
    nt = ChainRulesCore.NoTangent()
    function row_pullback(Δ)
        dF = ChainRulesCore.unthunk(ChainRulesCore.unthunk(Δ)[1])
        dF isa ChainRulesCore.AbstractZero && return (nt, nt, nt, ChainRulesCore.ZeroTangent(), nt)
        return (nt, nt, nt, back(dF)[4], nt)
    end
    return y, row_pullback
end
end
