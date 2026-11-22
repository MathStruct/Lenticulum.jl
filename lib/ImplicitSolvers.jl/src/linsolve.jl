abstract type AbstractLinearSolver end

"""
    DenseSolve(; spd=false)

Solve by LU, or by Cholesky when `spd=true` asserts a symmetric positive definite
matrix. Factorization uses host arrays. A callable operator is materialized by
applying it to the coordinate basis.
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
    linsolve(A, b::AbstractVector, alg; tol=Tolerance())

Return `(x, report)` for the real square system `A*x = b`. Supply a square matrix
or a callable `v -> A*v` mapping vectors of length `length(b)` to vectors of the
same length. The tolerance scale is the norm of `b`.
"""
function linsolve(A, b::AbstractVector, alg::AbstractLinearSolver; tol::Tolerance=Tolerance())
    n = length(b)
    apply = _linear_operator(A, n)
    if n == 0
        x = float.(b)
        return x, _report(:converged, 0, _primal_norm(x), 0)
    end
    return _linsolve(A, apply, b, alg, tol)
end

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

function _linsolve(A, apply, b, alg::DenseSolve, tol)
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

function _linsolve(A, apply, b, alg::GMRES, tol)
    x = zero.(float.(b))
    r = b .- apply(x)
    # The Givens rotations below are those of real arithmetic (GLM's review of W1, F2).
    eltype(r) <: Complex && throw(ArgumentError("GMRES solves real systems; got $(eltype(r))"))
    scale = _primal_norm(b)
    total, nfe = 0, 1
    while true
        residual = _primal_norm(r)
        status = _status(residual, scale, tol)
        status === :continue || return x, _report(status, total, residual, nfe)
        total == alg.maxiters && return x, _report(:maxiters, total, residual, nfe)
        k = min(alg.restart, length(b), alg.maxiters - total)
        xnew, steps, exit = _gmres_cycle(apply, x, r, k, tol, scale)
        total += steps
        nfe += steps
        exit === :nonfinite && return x, _report(exit, total, residual, nfe)

        r = b .- apply(xnew)
        nfe += 1
        if exit === :stalled || _unchanged(xnew, x)
            residual = _primal_norm(r)
            status = _status(residual, scale, tol)
            return xnew, _report(status === :continue ? :stalled : status, total, residual, nfe)
        end
        x = xnew
    end
end

# Only the small least-squares problem is on the host. The basis vectors, products
# and correction to x follow the array backend of the residual.
function _gmres_cycle(apply, x, r, k, tol, scale)
    beta = norm(r)
    iszero(primal(beta)) && return x, 0, :stalled   # nothing to do; the caller reports the residual
    target = _bound(tol, scale, typeof(primal(beta)))
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
            H[i, j], H[i + 1, j] = c[i] * a + s[i] * b, -s[i] * a + c[i] * b
        end
        a, b = H[j, j], H[j + 1, j]
        d = hypot(a, b)
        isfinite(primal(d)) || return x, j, :nonfinite
        iszero(primal(d)) && return _gmres_update(x, V, H, g, j - 1), j, :stalled
        c[j], s[j] = a / d, b / d
        H[j, j], H[j + 1, j] = d, zero(T)
        g[j + 1] = -s[j] * g[j]
        g[j] = c[j] * g[j]

        if j == k || breakdown || abs(primal(g[j + 1])) <= target
            return _gmres_update(x, V, H, g, j), j, :continue
        end
        push!(V, w ./ next_norm)
    end
end

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
