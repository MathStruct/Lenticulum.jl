module LenticulumChainRulesCoreExt
using Lenticulum: Lenticulum as LT
using ChainRulesCore
using LinearAlgebra

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
