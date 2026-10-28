module LenticulumCoreGPUArraysCoreChainRulesExt
using LenticulumCore: LenticulumCore as LC
using GPUArraysCore: AbstractGPUArray
using ChainRulesCore
using LinearAlgebra: Diagonal, diag

# Device diagonal moments must return a structural diagonal cotangent, consistent
# with the elementwise mean/logdet paths and direct diagonal field reads.
function ChainRulesCore.rrule(::typeof(LC.belief_cov),
        b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    C = LC.belief_cov(b)
    function back(dC)
        dC = unthunk(dC)
        dC isa AbstractZero && return NoTangent(), ZeroTangent()
        d = dC isa Tangent ? unthunk(dC.diag) : diag(dC)
        return NoTangent(), Tangent{typeof(b)}(;η=ZeroTangent(),
            Λ=Tangent{typeof(b.Λ)}(;diag=.-d ./ abs2.(b.Λ.diag)))
    end
    return C, back
end
end
