abstract type AbstractLinearSolver end

"""
    DenseSolve(; spd=false)

Solve by LU, or by Cholesky when `spd=true` asserts a symmetric positive definite
matrix. Factorization uses host arrays. A callable operator is materialized by
applying it to the coordinate basis.

The report's `iterations` is `1` after a successful factorization and `0` when the
matrix has nonfinite entries or the factorization fails (the zero vector is then
returned). Callers tell a singular system from an unmet tolerance by it
(MyceliumLearning's graph adjoint does).
"""
struct DenseSolve <: AbstractLinearSolver
    spd::Bool
end
DenseSolve(; spd=false) = DenseSolve(spd)

"""
    GMRES(; restart=40, maxiters=200)

Restarted GMRES with double Gram–Schmidt and Givens rotations. `maxiters` bounds
the number of Arnoldi steps. The residual is checked with extra operator applications,
which are included in the report's `nfe`.
"""
struct GMRES <: AbstractLinearSolver
    restart::Int
    maxiters::Int

    function GMRES(; restart::Int=40, maxiters::Int=200)
        restart > 0 || throw(ArgumentError("restart must be positive"))
        maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
        return new(restart, maxiters)
    end
end

"""
    MINRES(; maxiters=200)

Paige–Saunders minimum residual iteration for a Hermitian operator (real symmetric
or complex Hermitian), including indefinite systems. `maxiters` bounds the number
of Lanczos steps. The solver assumes Hermitian symmetry without checking it.
For a non-Hermitian operator the recurrence can yield a wrong iterate. If its true
residual misses the tolerance, the solve reports `:stalled` or `:maxiters` for
finite arithmetic. Preconditioning is unsupported.
"""
struct MINRES <: AbstractLinearSolver
    maxiters::Int

    function MINRES(; maxiters::Int=200)
        maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
        return new(maxiters)
    end
end

"""
    KrylovJL(method::Symbol = :gmres; memory=40, maxiters=200, ldiv=false)

Use the Krylov.jl backend, which requires `using Krylov`. Choose `:gmres`,
`:minres` (Hermitian), or `:cg` (Hermitian positive definite). `memory` is the
restart length for GMRES. `maxiters` is the total primal iteration budget.

Pass approximate inverse actions or matrices as `left` and `right` to
[`linsolve`](@ref). With `ldiv=true`, pass objects supporting `ldiv!` instead.
MINRES and CG accept only a Hermitian positive definite left preconditioner.
"""
struct KrylovJL <: AbstractLinearSolver
    method::Symbol
    memory::Int
    maxiters::Int
    ldiv::Bool

    function KrylovJL(method::Symbol=:gmres; memory::Int=40, maxiters::Int=200, ldiv::Bool=false)
        method in (:gmres, :minres, :cg) || throw(ArgumentError("unknown KrylovJL method: $method"))
        memory > 0 || throw(ArgumentError("memory must be positive"))
        maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
        return new(method, memory, maxiters, ldiv)
    end
end

"""
    linsolve(A, b::AbstractVector, alg; tol=Tolerance(), left=nothing, right=nothing)

Return `(x, report)` for the square system `A*x = b`. Supply a square matrix
or a callable `v -> A*v` mapping vectors of length `length(b)` to vectors of the
same length. With GMRES, `left` and `right` can apply approximate inverses.
[`KrylovJL`](@ref) also accepts matrix or `ldiv!` preconditioners.
The report's residual and the initial convergence check use the true primal
residual norm and the primal norm of `b`.
DenseSolve and MINRES require both preconditioners to be `nothing`.

With ForwardDiff, Krylov solvers check the tangent residual after primal
convergence when the backend returns a dual-valued solution. Correction solves
contribute to the report's `iterations` and
`nfe`. If any correction fails to converge, the report gives the first failure status
while `residual` still describes the primal solve.
"""
function linsolve(A, b::AbstractVector, alg::AbstractLinearSolver; tol::Tolerance=Tolerance(),
        left=nothing, right=nothing)
    if left !== nothing || right !== nothing
        alg isa DenseSolve && throw(ArgumentError("DenseSolve does not support preconditioning"))
        alg isa MINRES && throw(ArgumentError(
            "preconditioned MINRES needs a symmetric positive definite preconditioner and is not implemented"))
        alg isa KrylovJL && alg.method !== :gmres && right !== nothing &&
            throw(ArgumentError("KrylovJL(:$(alg.method)) supports only a left preconditioner"))
    end
    n = length(b)
    apply = _linear_operator(A, n)
    if n == 0
        x = float.(b)
        return x, _report(:converged, 0, _primal_norm(x), 0)
    end
    return _linsolve(A, apply, b, alg, tol, left, right)
end

_linsolve(A, apply, b, alg::KrylovJL, tol, left, right) =
    _krylovjl_solve(A, apply, b, alg, tol, left, right)

_krylovjl_solve(A, apply, b, alg::KrylovJL, tol, left, right) =
    error("KrylovJL needs Krylov.jl: run `using Krylov`")

function _linear_operator(A::AbstractMatrix, n)
    size(A) == (n, n) || throw(DimensionMismatch("expected a matrix of size ($n, $n)"))
    return v -> A * v
end
_linear_operator(A, n) = v -> _vector_value(A(v), n)

_host_array(a::Array) = a
_host_array(a) = Array(a)

function _dense_matrix(A::AbstractMatrix, apply, b)
    return _host_array(A), 0
end

function _dense_matrix(A, apply, b)
    n = length(b)
    T = float(eltype(b))
    indices = 1:n
    M = stack(1:n) do j
        basis = similar(b, T)
        basis .= indices .== j
        return Array(apply(basis))
    end
    return M, n
end

function _linsolve(A, apply, b, alg::DenseSolve, tol, ::Nothing, ::Nothing)
    x0 = zero.(float.(b))
    scale = _primal_norm(b)
    isfinite(scale) || return x0, _report(:nonfinite, 0, scale, 0)
    M, nfe = _dense_matrix(A, apply, b)
    if !all(isfinite ∘ primal, M)
        r = _primal_norm(b .- apply(x0))
        return x0, _report(:nonfinite, 0, r, nfe + 1)
    end
    factor = alg.spd ? cholesky(Symmetric(M); check=false) : lu(M; check=false)
    if !issuccess(factor)
        r = _primal_norm(b .- apply(x0))
        status = _status(r, scale, tol)
        return x0, _report(status === :continue ? :stalled : status, 0, r, nfe + 1)
    end
    host_x = factor \ _host_array(b)
    x = copyto!(similar(b, eltype(host_x)), host_x)
    r = _primal_norm(b .- apply(x))
    status = _status(r, scale, tol)
    return x, _report(status === :continue ? :stalled : status, 1, r, nfe + 1)
end

_precondition(::Nothing, v) = v
_precondition(precond, v) = precond(v)

_tangent_refine(apply, b, x, ax, r, report, alg, tol, left, right) = (x, report)

function _krylov_result(apply, b, x, ax, r, report, alg, tol, left, right)
    report.converged || return x, report
    return _tangent_refine(apply, b, x, ax, r, report, alg, tol, left, right)
end

function _linsolve(A, apply, b, alg::GMRES, tol, left, right)
    x0 = zero.(float.(b))
    x = x0
    ax = apply(x)
    r = b .- ax
    operator = left === nothing && right === nothing ? apply :
        v -> _precondition(left, apply(_precondition(right, v)))
    scale = _primal_norm(b)
    total, nfe = 0, 1
    while true
        residual = _primal_norm(r)
        status = _status(residual, scale, tol)
        if status !== :continue
            return _krylov_result(apply, b, x, ax, r, _report(status, total, residual, nfe),
                                  alg, tol, left, right)
        end
        total == alg.maxiters && return x, _report(:maxiters, total, residual, nfe)
        k = min(alg.restart, length(b), alg.maxiters - total)
        # With right preconditioning the cycle works in correction coordinates.
        # Map that correction back once, then keep all iterates in the original system.
        base = right === nothing ? x : x0
        target_norm = left === nothing ? nothing : residual
        candidate, steps, exit =
            _gmres_cycle(operator, base, _precondition(left, r), k, tol, scale, target_norm)
        total += steps
        nfe += steps
        exit === :nonfinite && return x, _report(exit, total, residual, nfe)

        xnew = right === nothing ? candidate : x .+ right(candidate)
        ax = apply(xnew)
        r = b .- ax
        nfe += 1
        if exit === :stalled || _unchanged(xnew, x)
            residual = _primal_norm(r)
            status = _status(residual, scale, tol)
            report = _report(status === :continue ? :stalled : status, total, residual, nfe)
            return _krylov_result(apply, b, xnew, ax, r, report, alg, tol, left, right)
        end
        x = xnew
    end
end

# Only the small least-squares problem is on the host. The basis vectors, products
# and correction to x follow the array backend of the residual.
function _gmres_cycle(apply, x, r, k, tol, scale, true_norm)
    beta = norm(r)
    iszero(primal(beta)) && return x, 0, :stalled   # nothing to do; the caller reports the residual
    target = _bound(tol, scale, typeof(primal(beta)))
    if true_norm !== nothing
        isfinite(primal(beta)) || return x, 0, :nonfinite
        # Transfer the requested relative decrease to the preconditioned residual.
        # The result is only a cycle stopping estimate.
        target = (target / true_norm) * primal(beta)
    end
    q = r ./ beta
    V = [q]
    T = eltype(q)
    H = zeros(T, k + 1, k)
    c, s = zeros(T, k), zeros(T, k)
    g = zeros(T, k + 1)
    g[1] = beta

    for j in 1:k
        w = copy(apply(V[j]))
        before = _primal_norm(w)
        isfinite(before) || return x, j, :nonfinite
        for _ in 1:2, i in 1:j
            h = dot(V[i], w)
            H[i, j] += h
            w .-= h .* V[i]
        end
        after = _primal_norm(w)
        isfinite(after) || return x, j, :nonfinite
        breakdown = after <= eps(typeof(after)) * before
        next_norm = breakdown ? zero(T) : norm(w)
        H[j + 1, j] = next_norm

        for i in 1:(j - 1)
            a, b = H[i, j], H[i + 1, j]
            H[i, j], H[i + 1, j] = c[i] * a + s[i] * b, -conj(s[i]) * a + conj(c[i]) * b
        end
        a, b = H[j, j], H[j + 1, j]
        d = _gmres_hypot(a, b)
        isfinite(primal(d)) || return x, j, :nonfinite
        iszero(primal(d)) && return _gmres_update(x, V, H, g, j - 1), j, :stalled
        c[j], s[j] = conj(a) / d, conj(b) / d
        H[j, j], H[j + 1, j] = d, zero(T)
        g[j + 1] = -conj(s[j]) * g[j]
        g[j] = c[j] * g[j]

        if j == k || breakdown || abs(primal(g[j + 1])) <= target
            return _gmres_update(x, V, H, g, j), j, :continue
        end
        push!(V, w ./ next_norm)
    end
end

_gmres_hypot(a, b) = hypot(a, b)
_gmres_hypot(a::Complex, b::Complex) = hypot(abs(a), abs(b))

function _gmres_update(x, V, H, g, j)
    j == 0 && return x
    y = UpperTriangular(view(H, 1:j, 1:j)) \ view(g, 1:j)
    xnew = similar(x, promote_type(eltype(x), eltype(y)))
    xnew .= x
    for i in 1:j
        xnew .+= y[i] .* V[i]
    end
    return xnew
end

function _linsolve(A, apply, b, alg::MINRES, tol, ::Nothing, ::Nothing)
    x = zero.(float.(b))
    scale = _primal_norm(b)
    if iszero(scale)
        return _tangent_refine(apply, b, x, nothing, nothing, _report(:converged, 0, scale, 0),
                               alg, tol, nothing, nothing)
    end
    ax = apply(x)
    r = b .- ax
    residual = _primal_norm(r)
    total, nfe = 0, 1
    while true
        status = _status(residual, scale, tol)
        if status !== :continue
            return _krylov_result(apply, b, x, ax, r, _report(status, total, residual, nfe),
                                  alg, tol, nothing, nothing)
        end
        total == alg.maxiters && return x, _report(:maxiters, total, residual, nfe)

        xnew, steps, exit = _minres_cycle(apply, x, r, alg.maxiters - total, tol, scale)
        total += steps
        nfe += steps
        exit === :nonfinite && return x, _report(exit, total, residual, nfe)

        ax = apply(xnew)
        r = b .- ax
        nfe += 1
        residual = _primal_norm(r)
        if _unchanged(xnew, x)
            status = _status(residual, scale, tol)
            report = _report(status === :continue ? :stalled : status, total, residual, nfe)
            return _krylov_result(apply, b, xnew, ax, r, report, alg, tol, nothing, nothing)
        end
        x = xnew
    end
end

function _minres_cycle(apply, x, r, steps, tol, scale)
    ϕ = norm(r)
    target = _bound(tol, scale, typeof(primal(ϕ)))
    q = r ./ ϕ
    qprev = zero.(q)
    pprev = pprev2 = qprev
    β = zero(ϕ)
    c = cprev = one(ϕ)
    s = sprev = zero(ϕ)

    for j in 1:steps
        Aq = apply(q)
        isfinite(_primal_norm(Aq)) || return x, j, :nonfinite
        α = real(dot(q, Aq))
        w = Aq .- α .* q .- β .* qprev
        βnext = norm(w)
        if !isfinite(primal(βnext))
            # A scaled norm on duals can divide by a zero primal at breakdown.
            if _primal_type(eltype(w)) === eltype(w) || !iszero(_primal_norm(w))
                return x, j, :nonfinite
            end
            βnext = zero(βnext)
        end
        breakdown = iszero(primal(βnext))
        breakdown && (βnext = zero(βnext))

        # Apply the two previous rotations to the new Lanczos column.
        # ε and δ are the two superdiagonals of the triangular factor.
        ε = sprev * β
        δ = c * cprev * β + s * α
        γbar = c * α - s * cprev * β
        γ = hypot(γbar, βnext)
        isfinite(primal(γ)) || return x, j, :nonfinite
        iszero(primal(γ)) && return x, j, :continue
        cprev, sprev = c, s
        c, s = γbar / γ, βnext / γ

        p = (q .- δ .* pprev .- ε .* pprev2) ./ γ
        x = x .+ (c * ϕ) .* p
        ϕ = -s * ϕ
        if j == steps || breakdown || abs(primal(ϕ)) <= target
            return x, j, :continue
        end
        qprev, q = q, w ./ βnext
        pprev2, pprev = pprev, p
        β = βnext
    end
end
