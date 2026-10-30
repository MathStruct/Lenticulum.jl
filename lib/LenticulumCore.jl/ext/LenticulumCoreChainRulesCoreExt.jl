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
        # Where pⱼ underflows (wⱼ ≪ max w) that loses 1/Σw, so use Dⱼ/wⱼ − ΣD/Σw there, which
        # is then safe because Σw ≥ max w is not small (LP-019 review).
        # The rule is linear in D: scale D by its largest entry so that ΣD cannot overflow,
        # and multiply back at the end (LP-019 third review).
        Dm = maximum(abs, D)
        iszero(Dm) && return NoTangent(), zero(D) ./ weights
        Ds = D ./ Dm
        m = maximum(weights)
        s = sum(weights ./ m)
        p = (weights ./ m) ./ s
        SD = sum(Ds)
        tiny = p .< floatmin(float(eltype(weights)))
        return NoTangent(), Dm .* ifelse.(tiny, Ds ./ weights .- (SD / m) / s, (Ds .- p .* SD) ./ weights)
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
        # Forward evaluation inverts the symmetric part of Λ, so its cotangent is symmetric too.
        G = -(C' * dC * C')
        return NoTangent(), Tangent{typeof(b)}(;η=ZeroTangent(),Λ=(G + G') / 2)
    end
    return C, back
end
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode},
                             ::typeof(LenticulumCore.straight_through), f, s, x)
    _, back = rrule_via_ad(config, s, x)
    return f(x), dy -> (NoTangent(), NoTangent(), NoTangent(), back(dy)[2])
end
end
