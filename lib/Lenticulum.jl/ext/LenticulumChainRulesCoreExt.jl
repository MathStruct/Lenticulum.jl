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
# the fixed output coefficient is a constant, not a parameter; only its number type follows the parameters (P34)
ChainRulesCore.@non_differentiable LT._fixed_at(C, ps)
# the SVD of the flat path is of values; `_range_complement` carries the derivative (P4)
ChainRulesCore.@non_differentiable LT._range_svd(B)
ChainRulesCore.@non_differentiable LT._partly_flat(b)
# true inside a reverse pass (Zygote): the batched plate path of HardConstraintFactor (P6) takes
# its decompositions on values only and steps aside for the per-replica path, which carries them
ChainRulesCore.rrule(::typeof(LT._within_gradient)) = true, _ -> (NoTangent(),)

# A stored covariance in the call's working type (Q4): the further arguments only choose the type, so
# they get no tangent, and the covariance's is the cotangent projected back. Without this rule Zygote
# traced the type selection on every message and free energy: +16 % on MyceliumLearning's
# `kalman/zygote_gradient` benchmark (LP-091).
function ChainRulesCore.rrule(::typeof(LT._covariance_at), Q, inputs...)
    project = ProjectTo(Q)
    covariance_at_pullback(dy) = (NoTangent(), project(unthunk(dy)), map(_ -> ZeroTangent(), inputs)...)
    return LT._covariance_at(Q, inputs...), covariance_at_pullback
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
