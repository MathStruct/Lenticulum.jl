# Frozen pre-method-object solver from dev-danielb, for bit-identical regression tests.
function legacy_fit_sampson(tab::ExponentTable,X::AbstractMatrix{<:Real},k::Integer;
                     Σ=LinearAlgebra.I,Θ0=nothing,method::Symbol=:riemannian,
                     tol::Real=1e-7,maxiter::Integer=10000,λ::Real=0)
    M = AlgebraicFactors._polyfit_data(tab,X,k)
    method in (:riemannian,:fns) || throw(ArgumentError("unknown method $method"))
    method === :fns && k != 1 && throw(ArgumentError("FNS supports k=1 only"))
    isfinite(tol) && tol>0 || throw(ArgumentError("tol must be finite and positive"))
    maxiter>=0 && !(maxiter isa Bool) || throw(ArgumentError("maxiter must be nonnegative"))
    isfinite(λ) && λ>=0 || throw(ArgumentError("λ must be finite and nonnegative"))
    Θ = Θ0 === nothing ? fit_nullspace(tab,X,k).Θ : float.(copy(Θ0))
    Θ isa AbstractMatrix{<:Real} && ispoint(M,Θ) || throw(ArgumentError("Θ0 must be an orthonormal Grassmann representative"))
    data = AlgebraicFactors._sampson_data(tab,X,Σ,λ)
    cost,G = AlgebraicFactors._sampson_cost_gradient(Θ,data)
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
            cost,G = AlgebraicFactors._sampson_cost_gradient(candidate,data)
        else
            accepted = false
            step = min(1.0,2step)
            for _ in 1:60
                candidate = retract(M,Θ,-step*grad)
                # A singular trial is an explicit failure, never silently regularised.
                trial,Gtrial = AlgebraicFactors._sampson_cost_gradient(candidate,data)
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
