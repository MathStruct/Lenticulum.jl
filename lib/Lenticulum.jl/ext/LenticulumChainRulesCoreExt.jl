module LenticulumChainRulesCoreExt
using Lenticulum: Lenticulum as LT
using ChainRulesCore
using LinearAlgebra

# A mean-field node's product belief is built from the marginals, which the graph gradient holds
# fixed; only the parameters move.
ChainRulesCore.@non_differentiable LT._product_belief(names, dims, marginals)
ChainRulesCore.@non_differentiable LT._switch_product_belief(f, marginals)
# rank tests: structure, not values
ChainRulesCore.@non_differentiable LT._rank_deficient(A)
ChainRulesCore.@non_differentiable LT._partly_flat(b)
ChainRulesCore.@non_differentiable LT._as_clusters(fs)

function ChainRulesCore.rrule(::typeof(LT._prior_precision), Σ)
    P = LT._prior_precision(Σ)
    function back(dP)
        D = unthunk(dP)
        D isa AbstractZero && return NoTangent(), ZeroTangent()
        return NoTangent(), -(P' * D * P')
    end
    return P, back
end

function ChainRulesCore.rrule(::typeof(LT._clip_precision), P, cap)
    E = eigen(Symmetric(P))
    if maximum(E.values) < cap
        return P, dy -> (NoTangent(), unthunk(dy), NoTangent())
    end
    U = E.vectors
    result = U * Diagonal(min.(E.values, cap)) * U'
    K = LT._clip_frechet(E.values, cap)
    function pullback(dy)
        D = unthunk(dy)
        D isa AbstractZero && return (NoTangent(), D, NoTangent())
        # The primal reads the upper triangle through Symmetric(P).
        G = U * (K .* (U' * ((D + D') / 2) * U)) * U'
        dP = [i == j ? G[i, j] : i < j ? 2G[i, j] : zero(eltype(G))
              for i in axes(P, 1), j in axes(P, 2)]
        return NoTangent(), dP, NoTangent()
    end
    return result, pullback
end
end
