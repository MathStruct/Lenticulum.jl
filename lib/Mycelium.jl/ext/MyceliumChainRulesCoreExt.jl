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
