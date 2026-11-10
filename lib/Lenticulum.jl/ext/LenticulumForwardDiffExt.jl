module LenticulumForwardDiffExt
using Lenticulum: Lenticulum as LT
using ForwardDiff: ForwardDiff as FD
using LinearAlgebra

function LT._clip_precision(P::AbstractMatrix{FD.Dual{Tag,V,N}}, cap) where {Tag,V,N}
    primal = FD.value.(P)
    E = eigen(Symmetric(primal))
    maximum(E.values) < cap && return P
    U = E.vectors
    result = U * Diagonal(min.(E.values, cap)) * U'
    K = LT._clip_frechet(E.values, cap)
    tangents = ntuple(N) do k
        D = map(x -> FD.partials(x)[k], P)
        U * (K .* (U' * Symmetric(D) * U)) * U'
    end
    return map(eachindex(result)) do i
        FD.Dual{Tag}(result[i], FD.Partials(ntuple(k -> tangents[k][i], N)))
    end |> x -> reshape(x, size(P))
end
end
