module MyceliumKrylovExt
using Mycelium, Krylov, LinearAlgebra

function Mycelium._graph_linear_solve(::Val{:krylov}, A::Mycelium._GraphAdjointOperator,
        rhs, ctx, tol, maxiter)
    T = typeof(real(zero(eltype(rhs))))
    λ, stats = Krylov.gmres(A, rhs; rtol=T(tol), atol=zero(T), itmax=maxiter,
        memory=min(length(rhs), 30))
    stats.solved || throw(ArgumentError("implicit graph GMRES did not converge: $(stats.status)"))
    return λ
end
end
