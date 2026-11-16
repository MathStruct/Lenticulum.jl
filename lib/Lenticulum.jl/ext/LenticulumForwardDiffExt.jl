module LenticulumForwardDiffExt
using Lenticulum: Lenticulum as LT
using LenticulumCore: LenticulumCore as LC
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

# --- The flat path when the flat directions move (P25) ----------------------------------------
#
# Both methods give the derivative at constant rank, as `MyceliumForwardDiffFlatExt` does for a
# cluster's joint precision (P26): from the decomposition at the value, the bases of a subspace and
# of its complement move only into each other, to first order, and no eigenvector or singular
# vector is differentiated (repeated eigenvalues are no difficulty). The rank is the value path's,
# decided by the same rule on the values. A tangent that changes it to first order is refused: the
# flat-direction limit drops a different constant there, and the message has no derivative
# (gaussian.md, "Flat directions that move").

# Refuse a tangent with Lᵀ Dₖ R ≠ 0 (L, R the left and right null spaces at the value), which
# changes the rank to first order; `scale` is the size of the matrix's nonzero spectrum.
function _check_constant_rank(M::AbstractMatrix{FD.Dual{Tg,V,P}}, L, R, scale, what) where {Tg,V,P}
    (isempty(L) || isempty(R)) && return nothing
    for k in 1:P
        Dk = map(x -> FD.partials(x, k), M)
        tol = sqrt(eps(V)) * max(maximum(abs, Dk; init = zero(V)), scale)
        maximum(abs, L' * Dk * R; init = zero(V)) <= tol || throw(ArgumentError(
            "$what: a derivative that changes the number of flat directions has no flat-path \
             message (the limit has no derivative there) (P25)"))
    end
    return nothing
end

# The split of a flagged input (η, Λ) with Λ dual: mean Λ⁺η, covariance Λ⁺, flat along N. With
# U, λ, N the range, its eigenvalues and the null space at the value, and D the tangent of Λ:
#     U(ε) = U + ε N NᵀDU λ⁻¹,   N(ε) = N − ε U λ⁻¹ UᵀDN,   Λ⁺(ε) = U(ε) (UᵀΛU)⁻¹ U(ε)ᵀ,
# the last the first-order derivative of the pseudo-inverse at constant rank,
# −Λ⁺DΛ⁺ + NNᵀDΛ⁺² + Λ⁺²DNNᵀ.
function LT._moving_flat_split(η::AbstractVector, Λ::AbstractMatrix{FD.Dual{Tg,V,P}}) where {Tg,V<:LinearAlgebra.BlasFloat,P}
    Λ0 = FD.value.(Λ)
    LC._flat_split(FD.value.(η), Λ0)            # the value path's refusals: Λ ⪰ 0, η in range(Λ)
    E = eigen(LC._symmetric_part(Λ0))
    keep = E.values .> LC._eig_tol(E.values)
    U0, λ0, N0 = E.vectors[:, keep], E.values[keep], E.vectors[:, .!keep]
    _check_constant_rank(Λ, N0, N0, maximum(abs, λ0; init = zero(V)), "a flat input's precision")
    T = eltype(Λ)
    D = Λ .- Λ0                                  # the tangent, as duals of value zero
    U = U0 .+ N0 * ((N0' * D * U0) ./ transpose(λ0))
    N = Matrix{T}(N0 .- U0 * ((U0' * D * N0) ./ λ0))
    isempty(λ0) && return (zeros(T, length(η)), zeros(T, size(Λ)), N)
    K = U0' * Λ * U0
    S = U * (cholesky(Symmetric((K + K') / 2)) \ U')
    S = (S + S') / 2
    return (S * η, S, N)
end

# An orthonormal basis W of the complement of range(B), B of duals, and log pdet(BᵀB). With
# B = UᵣSVᵣᵀ at the value, W its complement and D the tangent:
#     W(ε) = W − ε (B⁺)ᵀDᵀW,   W(ε)W(ε)ᵀ = I − P(ε) to first order, P the projector on range(B);
#     d log pdet(BᵀB) = 2 tr(B⁺D).
function LT._range_complement(B::AbstractMatrix{FD.Dual{Tg,V,P}}) where {Tg,V<:LinearAlgebra.BlasFloat,P}
    B0 = FD.value.(B)
    F, r = LT._range_svd(B0)
    S = F.S[1:r]
    W0 = F.U[:, (r + 1):end]
    _check_constant_rank(B, W0, F.V[:, (r + 1):end], maximum(S; init = zero(V)), "the flat directions' coefficients")
    D = B .- B0
    Bp = F.V[:, 1:r] * (F.U[:, 1:r]' ./ S)      # B⁺ at the value
    W = W0 .- Bp' * (D' * W0)
    return W, 2 * sum(log, S; init = zero(V)) + 2 * sum(Bp' .* D; init = zero(eltype(B)))
end
end
