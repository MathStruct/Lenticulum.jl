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
    AbstractSampsonMethod

Solver interface for [`fit_sampson`](@ref). Implement
[`sampson_solve`](@ref) for a subtype to supply another optimiser.
"""
abstract type AbstractSampsonMethod end

"""
    RiemannianGradient(; c=1e-4, shrink=0.5, max_backtracks=60, initial_step=1.0)

Horizontal gradient descent with QR retraction and Armijo sufficient decrease
`c ∈ (0,1)`. Backtracking multiplies the step by `shrink ∈ (0,1)` at most
`max_backtracks ≥ 1` times. Each search starts at twice the last accepted step,
capped at finite positive `initial_step`. Defaults reproduce `:riemannian`.
"""
struct RiemannianGradient <: AbstractSampsonMethod
    c::Float64
    shrink::Float64
    max_backtracks::Int
    initial_step::Float64
    function RiemannianGradient(;c::Real=1e-4,shrink::Real=0.5,
                                max_backtracks=60,initial_step::Real=1.0)
        isfinite(c) && 0<c<1 || throw(ArgumentError("c must lie in (0,1)"))
        isfinite(shrink) && 0<shrink<1 || throw(ArgumentError("shrink must lie in (0,1)"))
        max_backtracks isa Integer && max_backtracks>=1 && !(max_backtracks isa Bool) || throw(ArgumentError("max_backtracks must be a positive integer"))
        isfinite(initial_step) && initial_step>0 || throw(ArgumentError("initial_step must be finite and positive"))
        new(c,shrink,max_backtracks,initial_step)
    end
end

"""
    FNS()

One-equation Fundamental Numerical Scheme of Chojnacki et al. (2000), §4.2,
Fig. 1. No tunable keywords. Uses the eigenvector with eigenvalue closest to
zero, with sign alignment; stops on the full Sampson gradient norm. Only k=1
is supported. This iteration need not decrease the cost monotonically.
"""
struct FNS <: AbstractSampsonMethod end

"""
    RiemannianTrustRegion(; Δ0=1.0, Δmax=100.0, ρ_accept=0.1,
                          tcg_maxiter=100, κ=0.1, θ=1.0)

Riemannian trust regions with Steihaug–Toint truncated conjugate gradients,
Frobenius metric on horizontal Grassmann lifts, and the existing QR retraction.
Uses analytic Riemannian Hessian-vector products (no finite-difference step).
See Absil, Mahony & Sepulchre (2008), Chapter 7, Algorithms 10–11.

Finite radii satisfy `0 < Δ0 ≤ Δmax`. Accept a step when actual/predicted cost
reduction exceeds `ρ_accept ∈ [0,1/4)`. Ratios below 1/4 quarter the radius;
ratios above 3/4 double it (up to Δmax) if CG reached the boundary. CG takes at
most `tcg_maxiter ≥ 1` iterations and stops when its residual is at most
`norm(grad)*min(κ,norm(grad)^θ)`, with finite `κ ∈ (0,1)` and `θ > 0`.
Negative curvature and boundary crossings truncate CG at the trust sphere.
"""
struct RiemannianTrustRegion <: AbstractSampsonMethod
    Δ0::Float64
    Δmax::Float64
    ρ_accept::Float64
    tcg_maxiter::Int
    κ::Float64
    θ::Float64
    function RiemannianTrustRegion(;Δ0::Real=1.0,Δmax::Real=100.0,ρ_accept::Real=0.1,
                                   tcg_maxiter=100,κ::Real=0.1,θ::Real=1.0)
        isfinite(Δ0) && isfinite(Δmax) && 0<Δ0<=Δmax || throw(ArgumentError("radii must satisfy 0 < Δ0 ≤ Δmax < Inf"))
        isfinite(ρ_accept) && 0<=ρ_accept<0.25 || throw(ArgumentError("ρ_accept must lie in [0,1/4)"))
        tcg_maxiter isa Integer && tcg_maxiter>=1 && !(tcg_maxiter isa Bool) || throw(ArgumentError("tcg_maxiter must be a positive integer"))
        isfinite(κ) && 0<κ<1 || throw(ArgumentError("κ must lie in (0,1)"))
        isfinite(θ) && θ>0 || throw(ArgumentError("θ must be finite and positive"))
        new(Δ0,Δmax,ρ_accept,tcg_maxiter,κ,θ)
    end
end

"""
    SampsonProblem

Problem supplied by `fit_sampson` to [`sampson_solve`](@ref). Public fields:
`manifold::Grassmann`, `cost_gradient(Θ)` returning `(cost, Euclidean_gradient)`,
and `hessian_vector(Θ, ξ)` returning the horizontal Riemannian Hessian action
for horizontal ξ. Use `project_tangent(problem.manifold, Θ, G)` and
`retract(problem.manifold, Θ, ξ)` for geometry. Data are cached once per fit;
solvers must not mutate the problem or the supplied initial Θ.
"""
struct SampsonProblem{F,H,D}
    manifold::Grassmann
    cost_gradient::F
    hessian_vector::H
    data::D
end

# Differentiate u=A⁻¹Θv, then G=uv′−uu′ΘB; include the Grassmann curvature term.
function _sampson_hessian(M,Θ,ξ,data)
    _,G = _sampson_cost_gradient(Θ,data) # same indexed SPD validation as the cost
    DG = zero(Θ)
    for (v,B) in data
        F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Θ*B*Θ'))
        u = F\(Θ*v)
        du = F\(ξ*v-(ξ*B*Θ'+Θ*B*ξ')*u)
        DG += du*v'-(du*u'+u*du')*Θ*B-(u*u')*ξ*B
    end
    normal = (G*Θ'+Θ*G')/2
    return project_tangent(M,Θ,DG-normal*ξ)
end

"""
    sampson_solve(method::AbstractSampsonMethod, problem::SampsonProblem, Θ0;
                  tol, maxiter)

Extension hook: implement this function for a user method subtype, accepting
positive gradient tolerance `tol` and nonnegative iteration budget `maxiter`.
`fit_sampson` validates the data and initial orthonormal representative first.
Use the problem's callable fields and Grassmann geometry (see [`SampsonProblem`](@ref)).
Return a named tuple containing `Θ`, `cost`, `gradient_norm`, `converged`,
`iterations`; extra fields are allowed. Θ must have orthonormal rows, cost is
the summed half squared Sampson objective there, gradient_norm is its horizontal
Frobenius norm, converged means gradient_norm ≤ tol, and iterations ≤ maxiter.
At maxiter=0 evaluate Θ0 without taking a step. Singular covariances throw;
solvers must not silently regularise them. No global optimum is promised.
"""
function sampson_solve end

"""
    fit_sampson(tab, X, k; Σ=I, Θ0=nothing, method=RiemannianTrustRegion(),
                tol=1e-7, maxiter=10000, λ=0)

Minimise summed half squared Sampson distance on Grassmann(k,m). Initialisation
uses `fit_nullspace`; supplied Θ0 must have orthonormal rows. No data or column
scaling. The default method is [`RiemannianTrustRegion`](@ref) (since 2026-10-26;
before, Armijo descent, now `RiemannianGradient()`). `method` accepts an [`AbstractSampsonMethod`](@ref) instance; symbols
`:riemannian`, `:fns`, `:trust_region` alias default [`RiemannianGradient`](@ref),
[`FNS`](@ref), [`RiemannianTrustRegion`](@ref) respectively. User methods implement
[`sampson_solve`](@ref). All built-in methods stop only at horizontal Frobenius
gradient norm ≤ tol. Return `(Θ,cost,gradient_norm,converged,iterations)`; RTR
also reports `hessian_products`. For RTR, iterations count attempted outer steps
(including rejected steps); for gradient descent and FNS, completed updates.
Singular current or trial covariances throw with point index. Explicit λ≥0 adds
λΘΘ′ to each pushed covariance. Dense real CPU data only; no global optimality.
"""
function fit_sampson(tab::ExponentTable,X::AbstractMatrix{<:Real},k::Integer;
                     Σ=LinearAlgebra.I,Θ0=nothing,method=RiemannianTrustRegion(),
                     tol::Real=1e-7,maxiter::Integer=10000,λ::Real=0)
    M = _polyfit_data(tab,X,k)
    if method isa Symbol
        method = method === :riemannian ? RiemannianGradient() :
                 method === :fns ? FNS() : method === :trust_region ? RiemannianTrustRegion() :
                 throw(ArgumentError("unknown method $method"))
    end
    method isa AbstractSampsonMethod || throw(ArgumentError("method must be an AbstractSampsonMethod or supported Symbol"))
    method isa FNS && k != 1 && throw(ArgumentError("FNS supports k=1 only"))
    isfinite(tol) && tol>0 || throw(ArgumentError("tol must be finite and positive"))
    maxiter>=0 && !(maxiter isa Bool) || throw(ArgumentError("maxiter must be nonnegative"))
    isfinite(λ) && λ>=0 || throw(ArgumentError("λ must be finite and nonnegative"))
    Θ = Θ0 === nothing ? fit_nullspace(tab,X,k).Θ : float.(copy(Θ0))
    Θ isa AbstractMatrix{<:Real} && ispoint(M,Θ) || throw(ArgumentError("Θ0 must be an orthonormal Grassmann representative"))
    data = _sampson_data(tab,X,Σ,λ)
    problem = SampsonProblem(M, Θ->_sampson_cost_gradient(Θ,data),
                             (Θ,ξ)->_sampson_hessian(M,Θ,ξ,data), data)
    return sampson_solve(method,problem,Θ;tol,maxiter)
end

function sampson_solve(method::Union{RiemannianGradient,FNS},problem::SampsonProblem,Θ;
                       tol,maxiter)
    M,data = problem.manifold,problem.data
    cost,G = problem.cost_gradient(Θ)
    grad = project_tangent(M,Θ,G)
    step = method isa RiemannianGradient ? method.initial_step : 1.0
    iterations = 0
    while LinearAlgebra.norm(grad)>tol && iterations<maxiter
        if method isa FNS
            H = zeros(eltype(Θ),M.m,M.m)
            for (v,B) in data
                a = only(Θ*B*Θ')
                r = only(Θ*v)
                H += v*v'/a - (r/a)^2*B
            end
            F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(H))
            candidate = Matrix(transpose(F.vectors[:,argmin(abs.(F.values))]))
            LinearAlgebra.dot(candidate,Θ)<0 && (candidate = -candidate)
            cost,G = problem.cost_gradient(candidate)
        else
            accepted = false
            step = min(method.initial_step,2step)
            for _ in 1:method.max_backtracks
                candidate = retract(M,Θ,-step*grad)
                trial,Gtrial = problem.cost_gradient(candidate)
                if trial <= cost-method.c*step*sum(abs2,grad)
                    cost,G = trial,Gtrial
                    accepted = true
                    break
                end
                step *= method.shrink
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

# Positive intersection of η+τd with the sphere. Stable quadratic-root evaluation.
function _trust_boundary(η,d,Δ)
    a,b = sum(abs2,d),LinearAlgebra.dot(η,d)
    c = sum(abs2,η)-Δ^2
    root = sqrt(max(zero(a),b*b-a*c))
    τ = b>=0 ? -c/(b+root) : (root-b)/a
    return η+τ*d
end

function _sampson_tcg(method,grad,H,Δ)
    η,r = zero(grad),copy(grad)
    d = -r
    rr = sum(abs2,r)
    target = sqrt(rr)*min(method.κ,sqrt(rr)^method.θ)
    for _ in 1:method.tcg_maxiter
        Hd = H(d)
        curvature = LinearAlgebra.dot(d,Hd)
        curvature<=0 && return _trust_boundary(η,d,Δ),true
        α = rr/curvature
        candidate = η+α*d
        LinearAlgebra.norm(candidate)>=Δ && return _trust_boundary(η,d,Δ),true
        η = candidate
        r += α*Hd
        rrnew = sum(abs2,r)
        sqrt(rrnew)<=target && return η,false
        d = -r+(rrnew/rr)*d
        rr = rrnew
    end
    return η,false
end

function sampson_solve(method::RiemannianTrustRegion,problem::SampsonProblem,Θ;tol,maxiter)
    M = problem.manifold
    cost,G = problem.cost_gradient(Θ)
    grad = project_tangent(M,Θ,G)
    Δ = method.Δ0
    iterations,hessian_products = 0,0
    while LinearAlgebra.norm(grad)>tol && iterations<maxiter
        H = ξ->begin
            hessian_products += 1
            problem.hessian_vector(Θ,ξ)
        end
        η,boundary = _sampson_tcg(method,grad,H,Δ)
        predicted = -LinearAlgebra.dot(grad,η)-LinearAlgebra.dot(η,H(η))/2
        predicted>0 && isfinite(predicted) || break
        candidate = retract(M,Θ,η)
        trial,Gtrial = problem.cost_gradient(candidate)
        ρ = (cost-trial)/predicted
        if ρ<0.25
            Δ /= 4
        elseif ρ>0.75 && boundary
            Δ = min(2Δ,method.Δmax)
        end
        if ρ>method.ρ_accept
            Θ,cost,G = candidate,trial,Gtrial
            grad = project_tangent(M,Θ,G)
        end
        iterations += 1
    end
    gradient_norm = LinearAlgebra.norm(grad)
    return (;Θ,cost,gradient_norm,converged=gradient_norm<=tol,iterations,hessian_products)
end
