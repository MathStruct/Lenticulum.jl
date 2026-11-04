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
ChainRulesCore.@non_differentiable Mycelium.cluster_form(f::Mycelium.TableFactor, ps)    # a constant table
ChainRulesCore.@non_differentiable Mycelium._variable_nodes(g)
ChainRulesCore.@non_differentiable Mycelium.factornode(g, id)
ChainRulesCore.@non_differentiable Mycelium.parameter_keys(g)
ChainRulesCore.@non_differentiable Mycelium.tied_factors(g, key)
ChainRulesCore.@non_differentiable Mycelium._key_node(g, key)
ChainRulesCore.@non_differentiable Mycelium._check_tied_state(g, st)
ChainRulesCore.@non_differentiable Mycelium._frozen_beliefs(s, g, ps, st)
ChainRulesCore.@non_differentiable Mycelium._energy_rows(s, g)
ChainRulesCore.@non_differentiable Mycelium._has_rows(s, g)

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
ChainRulesCore.@non_differentiable Mycelium._ReplayStore(g::FactorGraph)
ChainRulesCore.@non_differentiable Mycelium._check_energy_edges(g)
ChainRulesCore.@non_differentiable Mycelium._check_unnormalised(g)
function ChainRulesCore.rrule(::typeof(available_channels), s, g, fid; kwargs...)
    return available_channels(s,g,fid;kwargs...), _ -> (NoTangent(),NoTangent(),NoTangent(),NoTangent())
end
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(infer!), g::FactorGraph, schedule, ps, st; kwargs...)
    chosen = MessageTask[]
    result = schedule isa ResidualSchedule ?
        infer!(g, schedule, ps, st; kwargs..., record=chosen) : infer!(g, schedule, ps, st; kwargs...)
    damping = get(kwargs, :damping, 0.0)
    ts = schedule isa ResidualSchedule ? chosen :
        Mycelium._realised_tasks(g, schedule, result[2].sweeps, ps, st; damping)
    _, back = rrule_via_ad(config, Mycelium._functional_infer, g, ts, ps, st, result[2], damping)
    function pullback(dy)
        d = back(dy)
        return NoTangent(), NoTangent(), NoTangent(), d[4], NoTangent()
    end
    return result, pullback
end
end
