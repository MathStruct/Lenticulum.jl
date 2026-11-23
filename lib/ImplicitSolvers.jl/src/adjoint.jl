"""
    adjoint_solve(vjp, g::AbstractVector, alg; tol=Tolerance(), left=nothing, right=nothing)

Solve `J' * λ = g` with [`linsolve`](@ref). Pass the action `v -> J' * v`
or the adjoint matrix itself. The optional GMRES preconditioners apply to `J'`.
MINRES requires `J'` to be Hermitian.
Return `(λ, report)`.
"""
function adjoint_solve(vjp, g::AbstractVector, alg; tol::Tolerance=Tolerance(),
        left=nothing, right=nothing)
    return linsolve(vjp, g, alg; tol, left, right)
end

"""
    fixed_point_adjoint(vjp, g::AbstractVector, alg; tol=Tolerance(), left=nothing, right=nothing)

Solve `(I - J') * λ = g` with [`linsolve`](@ref). Pass the action `v -> J' * v`
or the adjoint matrix itself. The optional GMRES preconditioners apply to `I - J'`.
MINRES requires `I - J'` to be Hermitian.
Return `(λ, report)`.
"""
function fixed_point_adjoint(vjp, g::AbstractVector, alg; tol::Tolerance=Tolerance(),
        left=nothing, right=nothing)
    apply = _linear_operator(vjp, length(g))
    return linsolve(v -> v .- apply(v), g, alg; tol, left, right)
end
