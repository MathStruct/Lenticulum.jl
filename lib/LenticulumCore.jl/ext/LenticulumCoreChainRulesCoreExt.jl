module LenticulumCoreChainRulesCoreExt
using LenticulumCore
using ChainRulesCore
ChainRulesCore.@non_differentiable LenticulumCore._trainable_indices(ps, freeze)
ChainRulesCore.@non_differentiable LenticulumCore.offset_tree(ps)
ChainRulesCore.@non_differentiable LenticulumCore._flatlen(ps)
ChainRulesCore.@non_differentiable LenticulumCore._nearly_symmetric(M)
function ChainRulesCore.rrule(::typeof(LenticulumCore._normalised_logweights), weights)
    y = LenticulumCore._normalised_logweights(weights)
    function back(dy)
        D = unthunk(dy)
        D isa AbstractZero && return NoTangent(), ZeroTangent()
        all(>(0),weights) || throw(ArgumentError(
            "normalised log-weight derivatives require strictly positive weights; support-boundary derivatives are undefined"))
        # ∂ log(wᵢ/Σw)/∂wⱼ = δᵢⱼ/wᵢ − 1/Σw, so the pullback is (Dⱼ − pⱼ ΣD)/wⱼ with p = w/Σw.
        # Form the cancelling numerator first: ΣD/Σw overflows for tiny w (LP-019 item 7).
        m = maximum(weights)
        p = (weights ./ m) ./ sum(weights ./ m)
        return NoTangent(), (D .- p .* sum(D)) ./ weights
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
