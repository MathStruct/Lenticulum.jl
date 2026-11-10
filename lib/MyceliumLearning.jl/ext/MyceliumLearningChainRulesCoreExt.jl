module MyceliumLearningChainRulesCoreExt
using MyceliumLearning, Mycelium, LenticulumCore, ChainRulesCore
using Mycelium: FactorGraph, MessageTask, ResidualSchedule, infer!
ChainRulesCore.@non_differentiable MyceliumLearning._frozen_beliefs(s, g, ps, st)
ChainRulesCore.@non_differentiable MyceliumLearning._ReplayStore(g::FactorGraph)
ChainRulesCore.@non_differentiable MyceliumLearning._check_unnormalised(g)

# Reverse mode through `infer!`: the realised schedule is replayed functionally and that replay is
# differentiated (`gradient_couplings.jl`).
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(infer!), g::FactorGraph, schedule, ps, st; kwargs...)
    chosen = MessageTask[]
    result = schedule isa ResidualSchedule ?
        infer!(g, schedule, ps, st; kwargs..., record=chosen) : infer!(g, schedule, ps, st; kwargs...)
    damping = get(kwargs, :damping, 0.0)
    ts = schedule isa ResidualSchedule ? chosen :
        MyceliumLearning._realised_tasks(g, schedule, result[2].sweeps, ps, st; damping)
    _, back = rrule_via_ad(config, MyceliumLearning._functional_infer, g, ts, ps, st, result[2], damping)
    function pullback(dy)
        d = back(dy)
        return NoTangent(), NoTangent(), NoTangent(), d[4], NoTangent()
    end
    return result, pullback
end
end
