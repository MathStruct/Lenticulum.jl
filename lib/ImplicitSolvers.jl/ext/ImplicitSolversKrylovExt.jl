module ImplicitSolversKrylovExt

using LinearAlgebra
using ImplicitSolvers: ImplicitSolvers, KrylovJL, Tolerance, _bound, _krylov_result,
    _primal_norm, _primal_type, _report, _status, _strip_dual_level, _unchanged, linsolve
import Krylov

struct MulAction{T,F}
    n::Int
    product!::F
end
MulAction(::Type{T}, n, f::F) where {T,F} = MulAction{T,F}(n, f)
Base.size(A::MulAction) = (A.n, A.n)
Base.size(A::MulAction, d::Integer) = d <= 2 ? A.n : 1
Base.eltype(::MulAction{T}) where {T} = T
LinearAlgebra.mul!(y::AbstractVector, A::MulAction, x::AbstractVector) = A.product!(y, x)

_product(A::AbstractMatrix, apply, x) = mul!(similar(x), A, x)
_product(A, apply, x) = apply(x)
_mul!(y, A::AbstractMatrix, apply, x) = mul!(y, A, x)
_mul!(y, A, apply, x) = copyto!(y, apply(x))

# Krylov.jl is only called on a plain working type (dual systems are stripped first, P42), so a
# preconditioner that still carries a dual, one that A and b do not carry, is reduced to its primal
# values: the solution does not depend on it (LP-087 F1).
_primal_values(P::AbstractMatrix) = _primal_type(eltype(P)) === eltype(P) ? P : ImplicitSolvers.primal.(P)
_primal_values(P::UniformScaling) = UniformScaling(ImplicitSolvers.primal(P.λ))

_preconditioner(::Nothing, alg, b) = I
function _preconditioner(P::AbstractMatrix, alg, b)
    alg.ldiv && _check_ldiv(alg, eltype(P))
    return _primal_values(P)
end
# `I` and `c*I` are neither matrices nor callable; Krylov.jl takes them with `mul!` or `ldiv!` (LP-086 F2).
_preconditioner(P::UniformScaling, alg, b) = _primal_values(P)
function _preconditioner(P, alg, b)
    if alg.ldiv
        T = eltype(P)
        T <: Number && _check_ldiv(alg, T)
        return P
    end
    return MulAction(eltype(b), length(b), (y, x) -> y .= ImplicitSolvers.primal.(P(x)))
end

# P38 strips callable preconditioners when solving a tangent system.
_tangent_action(P::Union{AbstractMatrix,UniformScaling}) = v -> P * v
_tangent_action(P) = P

function _attempt(A, b, guess, alg, M, N; atol, rtol, itmax)
    kw = (; M, alg.ldiv, atol, rtol, itmax, history=true)
    if alg.method === :gmres
        return Krylov.gmres(A, b, guess...; kw..., N, alg.memory, restart=true)
    elseif alg.method === :minres
        # Disable the optional forward-error and condition-number exits.
        return Krylov.minres(A, b, guess...; kw..., etol=rtol, conlim=rtol)
    end
    # Krylov.jl's cg throws when r'Mr is negative or NaN. Its workspace keeps the residual, which
    # tells a nonfinite product (a :nonfinite report) from an indefinite preconditioner (LP-086 F1).
    workspace = Krylov.CgWorkspace(A, b)
    try
        Krylov.cg!(workspace, A, b, guess...; kw...)
    catch err
        err isa ErrorException && occursin("not symmetric positive definite", err.msg) || rethrow()
        finite = all(isfinite ∘ ImplicitSolvers.primal, workspace.r) &&
                 all(isfinite ∘ ImplicitSolvers.primal, workspace.z)
        finite && throw(ArgumentError(
            "KrylovJL(:cg): the preconditioner is not positive definite (Krylov.jl: $(err.msg))"))
        return nothing, workspace.stats
    end
    return workspace.x, workspace.stats
end

function _check_ldiv(alg, ::Type{T}) where {T}
    if alg.ldiv && _primal_type(T) !== T
        throw(ArgumentError("KrylovJL with ldiv=true does not support dual numbers"))
    end
end

_working_type(A::AbstractMatrix, b) = promote_type(eltype(A), float(eltype(b)))
_working_type(A, b) = float(eltype(b))

function _krylovjl_dual(A, apply, b, alg, tol, left, right, ::Type{T}, probes) where {T}
    left_action, right_action = _tangent_action(left), _tangent_action(right)
    op = A isa AbstractMatrix ? A : apply
    Avalue, bvalue, lvalue, rvalue = _strip_dual_level(op, b, left_action, right_action, T)
    xvalue, report = linsolve(Avalue, bvalue, alg; tol, left=lvalue, right=rvalue)
    x = T.(xvalue)
    nfe = report.nfe + probes
    if !report.converged
        return x, _report(report.reason, report.iterations, report.residual, nfe)
    end
    ax = _product(A, apply, x)
    r = b .- ax
    report = _report(report.reason, report.iterations, report.residual, nfe + 1)
    return _krylov_result(apply, b, x, ax, r, report, alg, tol, left_action, right_action)
end

function ImplicitSolvers._krylovjl_solve(A, apply, b::AbstractVector, alg::KrylovJL,
        tol::Tolerance, left, right)
    T = _working_type(A, b)
    _check_ldiv(alg, T)
    if A isa AbstractMatrix && _primal_type(T) !== T
        return _krylovjl_dual(A, apply, b, alg, tol, left, right, T, 0)
    end
    scale = _primal_norm(b)
    x = fill!(similar(b, T), zero(T))
    isfinite(scale) || return x, _report(:nonfinite, 0, scale, 0)

    ax = _product(A, apply, x)
    r = b .- ax
    # A callable may expose its element type only when applied.
    S = promote_type(T, eltype(r))
    _check_ldiv(alg, S)
    if _primal_type(S) !== S
        return _krylovjl_dual(A, apply, b, alg, tol, left, right, S, 1)
    end
    if S !== T
        x = fill!(similar(b, S), zero(S))
    end
    residual = _primal_norm(r)
    status = _status(residual, scale, tol)
    if status !== :continue
        return _krylov_result(apply, b, x, ax, r, _report(status, 0, residual, 1),
                              alg, tol, _tangent_action(left), _tangent_action(right))
    end
    iszero(alg.maxiters) && return x, _report(:maxiters, 0, residual, 1)

    rhs = eltype(b) === S ? b : S.(b)
    products = Ref(0)
    op = MulAction(S, length(b), (y, v) -> begin
        products[] += 1
        _mul!(y, A, apply, v)
    end)
    M = _preconditioner(left, alg, rhs)
    N = _preconditioner(right, alg, rhs)
    bound = _bound(tol, scale, typeof(residual))
    target = bound
    iterations, checks = 0, 1
    guess = ()
    while true
        candidate, stats = _attempt(op, rhs, guess, alg, M, N;
                                    atol=target, rtol=zero(target), itmax=alg.maxiters - iterations)
        iterations += stats.niter
        if candidate === nothing   # CG met a nonfinite product
            return x, _report(:nonfinite, iterations, residual, products[] + checks)
        end
        ax = _product(A, apply, candidate)
        r = b .- ax
        checks += 1
        nfe = products[] + checks
        next_residual = _primal_norm(r)
        status = _status(next_residual, scale, tol)
        if status !== :continue
            report = _report(status, iterations, next_residual, nfe)
            return _krylov_result(apply, b, candidate, ax, r, report,
                                  alg, tol, _tangent_action(left), _tangent_action(right))
        end
        iterations == alg.maxiters && return candidate, _report(:maxiters, iterations, next_residual, nfe)
        if next_residual >= residual || _unchanged(candidate, x)
            return x, _report(:stalled, iterations, residual, nfe)
        end

        # Transfer the remaining relative decrease to Krylov's residual estimate.
        target = min(target, (bound / next_residual) * last(stats.residuals))
        x, residual = candidate, next_residual
        guess = (x,)
    end
end

end
