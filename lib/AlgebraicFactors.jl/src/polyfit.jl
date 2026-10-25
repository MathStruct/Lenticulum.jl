"""
    fit_nullspace(tab, X, k; rtol=1e-12)

Fit the bottom k right singular vectors of V (points are columns of N×n X;
V[i,:]=veronese(tab,X[:,i])). No centring, column scaling or normalisation.
Returns `(Θ, singular_values, numerical_rank, nullity, threshold)`. The m singular
values are descending, padded with structural zeros when n<m. `nullity` counts
values ≤ rtol*σmax, including exact zeros when σmax=0. Θ has orthonormal rows.
The rank decision is diagnostic; k remains the caller's choice.
"""
function fit_nullspace(tab::ExponentTable, X::AbstractMatrix{<:Real}, k::Integer; rtol::Real=1e-12)
    _polyfit_data(tab,X,k)
    isfinite(rtol) && rtol>=0 || throw(ArgumentError("rtol must be finite and nonnegative"))
    V = permutedims(reduce(hcat,[veronese(tab,float.(X[:,i])) for i in axes(X,2)]))
    m = size(V,2)
    F = LinearAlgebra.svd(V;full=size(V,1)<m)
    s = vcat(F.S,zeros(eltype(F.S),m-length(F.S)))
    threshold = rtol*first(s)
    nullity = count(t->t<=threshold,s)
    return (Θ=Matrix(F.Vt[end-k+1:end,:]),singular_values=s,
            numerical_rank=m-nullity,nullity,threshold)
end

function _polyfit_data(tab,X,k)
    size(X,1) == size(tab.exponents,1) || throw(DimensionMismatch("X must be N×n, with points in columns"))
    size(X,2)>0 && all(isfinite,X) || throw(ArgumentError("X must contain finite, nonempty data"))
    k isa Bool && throw(ArgumentError("k must be an integer dimension"))
    return Grassmann(k,size(tab.exponents,2))
end

# Cache polynomial evaluations and data-space pushed covariances, independent of Θ.
function _sampson_data(tab,X,Σ,λ)
    cov = _polynomial_covariance(Σ,size(X,1))
    return map(axes(X,2)) do i
        x = float.(X[:,i])
        v,D = veronese(tab,x),veronese_jacobian(tab,x)
        B = cov === nothing ? D*D' : D*cov*D'
        return v, B+λ*LinearAlgebra.I
    end
end

function _sampson_cost_gradient(Θ,data)
    G = zero(Θ)
    cost = zero(eltype(Θ))
    for (i,(v,B)) in enumerate(data)
        A = LinearAlgebra.Symmetric(Θ*B*Θ')
        ev = LinearAlgebra.eigvals(A)
        threshold = length(ev)*eps(eltype(ev))*maximum(ev)
        all(isfinite,ev) && minimum(ev)>threshold || throw(ArgumentError(
            "Sampson covariance at point $i is numerically singular; use positive λ damping"))
        u = LinearAlgebra.cholesky(A)\(Θ*v)
        cost += LinearAlgebra.dot(Θ*v,u)/2
        G += u*v' - (u*u')*Θ*B
    end
    return cost,G
end

"""
    fit_sampson(tab, X, k; Σ=I, Θ0=nothing, method=:riemannian,
                tol=1e-7, maxiter=10000, λ=0)

Minimise summed half squared Sampson distance on Grassmann(k,m). Initialisation
defaults to fit_nullspace; supplied Θ0 must have orthonormal rows. No data or
column scaling. `:riemannian` uses the analytic full objective gradient,
horizontal projection, QR retraction and Armijo backtracking (c=1e-4, halving,
at most 60 trials, initial step 1 then twice the last accepted step, capped at 1).
`:fns` (k=1 only) uses Chojnacki et al. (2000), §4.2, Fig. 1: the eigenvector of
Σ[vv′/a - r² B/a²] with eigenvalue closest to zero, with sign alignment.
FNS is not a monotone line-search method and has no convergence guarantee here.

Both stop only at horizontal Frobenius gradient norm ≤ tol. Return
`(Θ,cost,gradient_norm,converged,iterations)`; exhausted iterations or unsuccessful
line search return converged=false. Singular covariances throw with point index;
optional λ≥0 adds λΘΘ′ (gauge-compatible). There is no pseudoinverse or claim
of global optimality: the objective is non-convex. Dense real CPU data only.
"""
function fit_sampson(tab::ExponentTable,X::AbstractMatrix{<:Real},k::Integer;
                     Σ=LinearAlgebra.I,Θ0=nothing,method::Symbol=:riemannian,
                     tol::Real=1e-7,maxiter::Integer=10000,λ::Real=0)
    M = _polyfit_data(tab,X,k)
    method in (:riemannian,:fns) || throw(ArgumentError("unknown method $method"))
    method === :fns && k != 1 && throw(ArgumentError("FNS supports k=1 only"))
    isfinite(tol) && tol>0 || throw(ArgumentError("tol must be finite and positive"))
    maxiter>=0 && !(maxiter isa Bool) || throw(ArgumentError("maxiter must be nonnegative"))
    isfinite(λ) && λ>=0 || throw(ArgumentError("λ must be finite and nonnegative"))
    Θ = Θ0 === nothing ? fit_nullspace(tab,X,k).Θ : float.(copy(Θ0))
    Θ isa AbstractMatrix{<:Real} && ispoint(M,Θ) || throw(ArgumentError("Θ0 must be an orthonormal Grassmann representative"))
    data = _sampson_data(tab,X,Σ,λ)
    cost,G = _sampson_cost_gradient(Θ,data)
    grad = project_tangent(M,Θ,G)
    step = 1.0
    iterations = 0
    while LinearAlgebra.norm(grad)>tol && iterations<maxiter
        if method === :fns
            H = zeros(eltype(Θ),M.m,M.m)
            for (v,B) in data
                a = only(Θ*B*Θ')
                r = only(Θ*v)
                H += v*v'/a - (r/a)^2*B
            end
            F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(H))
            candidate = Matrix(transpose(F.vectors[:,argmin(abs.(F.values))]))
            LinearAlgebra.dot(candidate,Θ)<0 && (candidate = -candidate)
            cost,G = _sampson_cost_gradient(candidate,data)
        else
            accepted = false
            step = min(1.0,2step)
            for _ in 1:60
                candidate = retract(M,Θ,-step*grad)
                # A singular trial is an explicit failure, never silently regularised.
                trial,Gtrial = _sampson_cost_gradient(candidate,data)
                if trial <= cost-1e-4*step*sum(abs2,grad)
                    cost,G = trial,Gtrial
                    accepted = true
                    break
                end
                step /= 2
            end
            accepted || break
        end
        Θ = candidate
        grad = project_tangent(M,Θ,G)
        iterations += 1
    end
    gradient_norm = LinearAlgebra.norm(grad)
    return (;Θ,cost,gradient_norm,converged=gradient_norm<=tol,iterations)
end
