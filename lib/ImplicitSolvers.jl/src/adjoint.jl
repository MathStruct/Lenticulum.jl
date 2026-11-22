"""
    adjoint_solve(vjp, g::AbstractVector, alg; tol=Tolerance())

Solve `J' * λ = g` with [`linsolve`](@ref). Pass the action `v -> J' * v`
or the transpose matrix itself. Return `(λ, report)`.
"""
function adjoint_solve(vjp, g::AbstractVector, alg; tol::Tolerance=Tolerance())
    return linsolve(vjp, g, alg; tol)
end

"""
    fixed_point_adjoint(vjp, g::AbstractVector, alg; tol=Tolerance())

Solve `(I - J') * λ = g` with [`linsolve`](@ref). Pass the action `v -> J' * v`
or the transpose matrix itself. Return `(λ, report)`.
"""
function fixed_point_adjoint(vjp, g::AbstractVector, alg; tol::Tolerance=Tolerance())
    apply = _linear_operator(vjp, length(g))
    return linsolve(v -> v .- apply(v), g, alg; tol)
end
