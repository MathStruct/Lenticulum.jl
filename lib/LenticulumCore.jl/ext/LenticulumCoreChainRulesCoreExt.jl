module LenticulumCoreChainRulesCoreExt
using LenticulumCore
using ChainRulesCore
ChainRulesCore.@non_differentiable LenticulumCore._trainable_indices(ps, freeze)
ChainRulesCore.@non_differentiable LenticulumCore.offset_tree(ps)
ChainRulesCore.@non_differentiable LenticulumCore._flatlen(ps)
function ChainRulesCore.rrule(::typeof(LenticulumCore._normalised_logweights), weights)
    y = LenticulumCore._normalised_logweights(weights)
    function back(dy)
        D = unthunk(dy)
        D isa AbstractZero && return NoTangent(), ZeroTangent()
        all(>(0),weights) || throw(ArgumentError(
            "normalised log-weight derivatives require strictly positive weights; support-boundary derivatives are undefined"))
        m = maximum(weights)
        return NoTangent(), D ./ weights .- (sum(D) / m) / sum(weights ./ m)
    end
    return y, back
end
ChainRulesCore.rrule(::typeof(stop_gradient), x) =
    (x, _ -> (NoTangent(), ZeroTangent()))
function ChainRulesCore.rrule(::typeof(belief_cov), b::GaussianBelief)
    C = belief_cov(b)
    function back(dC)
        dC = unthunk(dC)
        dC isa AbstractZero && return NoTangent(), ZeroTangent()
        return NoTangent(), Tangent{typeof(b)}(;η=ZeroTangent(),Λ=-(C' * dC * C'))
    end
    return C, back
end
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode},
                             ::typeof(LenticulumCore.straight_through), f, s, x)
    _, back = rrule_via_ad(config, s, x)
    return f(x), dy -> (NoTangent(), NoTangent(), NoTangent(), back(dy)[2])
end
end
