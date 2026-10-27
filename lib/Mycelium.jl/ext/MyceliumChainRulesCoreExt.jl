module MyceliumChainRulesCoreExt
using Mycelium, LenticulumCore, ChainRulesCore
ChainRulesCore.@non_differentiable Mycelium._active_indices(f, ps)
ChainRulesCore.@non_differentiable Mycelium._factor_nodes(g)
ChainRulesCore.@non_differentiable Mycelium._variable_nodes(g)
ChainRulesCore.@non_differentiable Mycelium.factornode(g, id)
ChainRulesCore.@non_differentiable Mycelium.resolve_polarity(g, fid, target, available)
ChainRulesCore.@non_differentiable Mycelium.check_legal(g, fid, p)
ChainRulesCore.@non_differentiable Mycelium.MessageStore(g::FactorGraph)
ChainRulesCore.@non_differentiable Mycelium._ReplayStore(g::FactorGraph)
ChainRulesCore.@non_differentiable Mycelium._check_energy_edges(g)
function ChainRulesCore.rrule(::typeof(available_channels), s, g, fid; kwargs...)
    return available_channels(s,g,fid;kwargs...), _ -> (NoTangent(),NoTangent(),NoTangent(),NoTangent())
end
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(infer!), g::FactorGraph, schedule, ps, st; kwargs...)
    get(kwargs, :damping, 0.0) == 0 || throw(ArgumentError("reverse schedule differentiation currently requires damping=0"))
    result = infer!(g, schedule, ps, st; kwargs...)
    ts = Mycelium._static_tasks(schedule, result[2].sweeps)
    _, back = rrule_via_ad(config, Mycelium._functional_infer, g, ts, ps, st, result[2])
    function pullback(dy)
        d = back(dy)
        return NoTangent(), NoTangent(), NoTangent(), d[4], NoTangent()
    end
    return result, pullback
end
end
