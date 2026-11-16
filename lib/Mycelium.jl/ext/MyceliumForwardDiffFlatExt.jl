# ForwardDiff through the diffuse limit of a cluster's joint precision (P26).
#
# A joint precision J with flat directions is split into its range and null space by an
# eigendecomposition, which has no rule for dual numbers. At constant rank the quantities the
# cluster uses (the pseudo-inverse, the pseudo-determinant, the range) are smooth in J, and their
# derivative depends only on J's tangent D, provided D is tangent to the matrices of that rank:
# NᵀDN = 0 (N the null space at the value). Then, with U, Λ the range and its eigenvalues at the
# value, a basis of the range and of the null space to first order are
#
#     U(ε) = U + ε N NᵀD U Λ⁻¹,      N(ε) = N − ε U Λ⁻¹ UᵀD N,
#
# (each moves only into the other, so both stay orthonormal to first order), and UᵀJU = Λ + ε UᵀDU
# on the range, its first-order term unaffected by the motion of U as J N = 0. No derivative of an
# eigenvector is taken, so repeated eigenvalues on the range are no difficulty. A tangent with
# NᵀDN ≠ 0 changes the rank, where the diffuse limit drops a different constant and has no
# derivative: refused (`cluster.md`, "Flat directions under a derivative").
module MyceliumForwardDiffFlatExt
using Mycelium, ForwardDiff, LinearAlgebra

function Mycelium._flat_factor(J::AbstractMatrix{ForwardDiff.Dual{Tg,V,P}}, C) where {Tg,V<:LinearAlgebra.BlasFloat,P}
    J0 = ForwardDiff.value.(J)
    F0 = Mycelium._flat_factor(J0, LinearAlgebra.cholesky(Symmetric(J0); check = false))
    F0 isa Mycelium._Flat || return C          # the primal's verdict: definite after all (C succeeded)
    U0, λ0, N0 = F0.U, F0.K.diag, F0.N
    T = eltype(J)
    for k in 1:P
        Dk = map(x -> ForwardDiff.partials(x, k), J)
        tol = sqrt(eps(V)) * max(maximum(abs, Dk; init = zero(V)), maximum(abs, λ0; init = zero(V)))
        maximum(abs, N0' * Dk * N0; init = zero(V)) <= tol || throw(ArgumentError(
            "cluster: a derivative that changes the number of flat directions of a joint precision (a parameter " *
            "that makes a flat direction determined): the diffuse limit has no derivative there (P26)"))
    end
    D = J .- J0                                # the tangent, as duals of value zero
    U = U0 .+ N0 * ((N0' * D * U0) ./ transpose(λ0))
    N = N0 .- U0 * ((U0' * D * N0) ./ λ0)
    isempty(λ0) && return Mycelium._Flat(Matrix{T}(U), Diagonal(zeros(T, 0)), Matrix{T}(N))
    K = U0' * J * U0
    return Mycelium._Flat(Matrix{T}(U), cholesky(Symmetric((K + K') / 2)), Matrix{T}(N))
end
end
