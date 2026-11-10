# All fitting and membership calculations use dense CPU Float64 arrays.
function _joint_vector(f, values, name)
    values isa AbstractArray{<:Real} &&
        (size(values) == f.alphabets ||
         (values isa AbstractVector && length(values) == size(f.A, 2))) ||
        throw(ArgumentError("$name must have size $(f.alphabets) or be a joint-state vector"))
    v = Float64.(vec(collect(values)))
    all(x -> isfinite(x) && x >= 0, v) ||
        throw(ArgumentError("$name must contain finite non-negative reals representable in Float64"))
    return v
end

function _probability_vector(f, p; positive = false)
    v = _joint_vector(f, p, "p")
    positive && !all(>(0), v) &&
        throw(ArgumentError("p must be strictly positive; use binomials at the boundary"))
    isapprox(sum(v), 1; atol = 1e-12, rtol = 1e-8) ||
        throw(ArgumentError("p must sum to one (atol=1e-12, rtol=1e-8)"))
    return v
end

# B = (I - 11'/m)A'. Q spans its image; R = V/S is its inverse on
# that image. R*β is orthogonal to ker(B), hence minimum norm in its gauge class.
function _fit_chart(A)
    B = Matrix{Float64}(transpose(A))
    all(isfinite, B) || throw(ArgumentError("A must be representable in Float64"))
    B .-= sum(B; dims = 1) ./ size(B, 1)
    F = LinearAlgebra.svd(B)
    cutoff = isempty(F.S) ? 0.0 : maximum(size(B)) * eps(Float64) * maximum(F.S)
    k = count(>(cutoff), F.S)
    return F.U[:, 1:k], F.V[:, 1:k] ./ transpose(F.S[1:k])
end

_moment_residual(A, p, t) = maximum(abs, A * p - t; init = 0.0)

function _newton_fit(Q, q, A, t, tol, maxiter)
    β = zeros(size(Q, 2))
    logp = _lognormalise(Q * β)
    p = exp.(logp)
    iterations = 0
    for _ in 1:maxiter
        _moment_residual(A, p, t) <= tol && break
        g = transpose(Q) * (q - p)
        μ = transpose(Q) * p
        C = Q .- transpose(μ)
        H = LinearAlgebra.Symmetric(transpose(C) * (p .* C))
        # The covariance is positive definite in exact arithmetic at finite β.
        # A spectral floor keeps a numerical near-boundary solve finite and ascending.
        E = LinearAlgebra.eigen(H)
        floor = max(eps(Float64) * maximum(E.values; init = 0.0), floatmin(Float64))
        step = E.vectors * ((transpose(E.vectors) * g) ./ max.(E.values, floor))
        slope = LinearAlgebra.dot(g, step)
        isfinite(slope) && slope > 0 || break
        ll = LinearAlgebra.dot(q, logp)
        accepted = false
        α = 1.0
        for _ in 1:60
            candidate = β + α * step
            logits = Q * candidate
            if all(isfinite, logits)
                lp = _lognormalise(logits)
                # Roundoff allowance matters once the likelihood gain is below eps,
                # even though moment matching can still improve substantially.
                if LinearAlgebra.dot(q, lp) >= ll + 1e-4 * α * slope - 8eps(Float64) * max(1, abs(ll))
                    β, logp = candidate, lp
                    p = exp.(logp)
                    accepted = true
                    break
                end
            end
            α *= 0.5
        end
        accepted || break
        iterations += 1
    end
    return β, iterations
end

function _logmass(logp, inds)
    peak = maximum(logp[inds])
    return peak + log(sum(x -> exp(x - peak), logp[inds]))
end

function _ipf_fit(Q, A, t, tol, maxiter)
    logp = fill(-log(size(A, 2)), size(A, 2))
    groups = [(findall(==(1), row), findall(==(0), row)) for row in eachrow(A)]
    iterations = 0
    for _ in 1:maxiter
        _moment_residual(A, exp.(logp), t) <= tol && break
        for (r, (yes, no)) in enumerate(groups)
            (isempty(yes) || isempty(no)) && continue
            tr = clamp(t[r], 0.0, 1.0) # Remove summation roundoff at binary endpoints.
            δ = tr == 0 ? -32.0 : tr == 1 ? 32.0 :
                log(tr) - log1p(-tr) + _logmass(logp, no) - _logmass(logp, yes)
            logp[yes] .+= δ
            logp = _lognormalise(logp)
        end
        iterations += 1
    end
    # Project out accumulated constant/gauge roundoff before recovering parameters.
    return transpose(Q) * (logp .- sum(logp) / length(logp)), iterations
end

"""
    fit_mle(f::ToricFactor, counts; method = :newton, tol = 1e-10, maxiter = 1000)

Fit the normalised model to finite non-negative real counts with positive total.
`counts` is an array of size `f.alphabets` or a vector in joint-state order (first
variable fastest). Calculations use dense CPU Float64 arrays and log probabilities.

Return `(η, p, converged, iterations, residual, existence)`. `p` is a vector in factor
order, reproduced by `ps = (η = η,)`. `η` is the minimum Euclidean norm representative
modulo `ker((I - 11'/m)A')`, computed by an SVD with relative cutoff
`max(size(A)...) * eps(Float64)`. `residual = norm(A*p - A*counts/sum(counts), Inf)`;
`converged` means exactly `residual ≤ tol`. `iterations` counts accepted Newton steps
or complete IPF sweeps; an exhausted budget or stalled line search can return false.

`:newton` uses an orthonormal basis of centred logits, their covariance Hessian,
and backtracking on the per-observation log-likelihood. Redundant rows and designs
without a constant row are supported. `:ipf` requires entries in `{0,1}` and cycles
over rows, exactly setting each binary feature's mean using its log odds. At target
0 or 1 the exact update is infinite; use a finite log-odds step of -32 or +32 per
sweep instead, approaching the endpoint while retaining finite parameters.

`existence` is separate from convergence: `:exists` certifies the ordinary MLE only
when every count is strictly positive (the target is then a positive convex
combination of all columns, hence in their relative interior). Otherwise it is
`:suspected_boundary` if the centred logit norm has grown from zero to at least 10
and `m*minimum(p) ≤ 1e-8`; it is `:unknown` otherwise. This heuristic is no proof of
non-existence. A boundary extended MLE can satisfy the moment tolerance while its
finite parameter approximations grow, so `converged` and `:suspected_boundary` can
occur together. No LP existence test or exact boundary parameter is returned.
"""
function fit_mle(f::ToricFactor, counts; method = :newton, tol = 1e-10, maxiter = 1000)
    method in (:newton, :ipf) || throw(ArgumentError("method must be :newton or :ipf"))
    tol isa Real && isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    maxiter isa Integer && !(maxiter isa Bool) && maxiter >= 0 ||
        throw(ArgumentError("maxiter must be a non-negative integer"))
    u = _joint_vector(f, counts, "counts")
    peak = maximum(u)
    peak > 0 || throw(ArgumentError("counts must have positive total"))
    # Scaling first also accepts finite counts whose unscaled total would overflow.
    q = u ./ peak
    q ./= sum(q)
    A = Matrix{Float64}(f.A)
    method == :ipf && !all(x -> x == 0 || x == 1, f.A) &&
        throw(ArgumentError("IPF requires a design with entries in {0,1}"))
    Q, R = _fit_chart(f.A)
    t = A * q
    β, iterations = method == :newton ? _newton_fit(Q, q, A, t, tol, maxiter) :
        _ipf_fit(Q, A, t, tol, maxiter)
    η = R * β
    # Report the probabilities of the returned parameter, including chart roundoff.
    logp = _lognormalise(_logits(f, (; η)))
    p = exp.(logp)
    residual = _moment_residual(A, p, t)
    centred = logp .- sum(logp) / length(logp)
    existence = all(>(0), u) ? :exists :
        LinearAlgebra.norm(centred) >= 10 && length(p) * minimum(p) <= 1e-8 ?
        :suspected_boundary : :unknown
    return (; η, p, converged = residual <= tol, iterations, residual, existence)
end

"""
    inmodel(f::ToricFactor, p; atol = 1e-8) -> Bool

Test positive model membership by least squares: `log(p)` must lie in the row space
of `[ones(1,m); A]`. `atol` bounds the Euclidean (2-norm) projection residual in log
probabilities, not a probability error. Uses the same SVD rank cutoff as `fit_mle`.
`p` must be a finite, strictly positive probability vector or an array of size
`f.alphabets`, in factor order; otherwise throw `ArgumentError`. Normalisation is
checked with `atol=1e-12, rtol=1e-8`, independently of the membership tolerance.
Boundary membership requires binomials; see [`binomial_residuals`](@ref).
"""
function inmodel(f::ToricFactor, p; atol = 1e-8)
    atol isa Real && isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("atol must be finite and non-negative"))
    v = _probability_vector(f, p; positive = true)
    Q, _ = _fit_chart(f.A)
    centred = log.(v)
    centred .-= sum(centred) / length(centred)
    return LinearAlgebra.norm(centred - Q * (transpose(Q) * centred)) <= atol
end

"""
    binomial_residuals(f::ToricFactor, p, moves) -> Vector{Float64}

Evaluate `prod(p.^u⁺) - prod(p.^u⁻)` for each supplied integer-vector move `u`.
`p` is a finite non-negative probability vector/array in factor order, including
boundary points. Every move must have length `prod(f.alphabets)`, `sum(u) == 0`,
and `A*u == 0`; these kernel checks use exact BigInt arithmetic. Invalid inputs
throw `ArgumentError`. Products use log space, treating empty products as 1 and
positive powers of zero as 0. A supplied collection need not be a Markov basis;
vanishing on it alone does not certify general boundary membership. No basis is computed.
"""
function binomial_residuals(f::ToricFactor, p, moves)
    v = _probability_vector(f, p)
    A = BigInt.(f.A)
    result = Float64[]
    for u in moves
        u isa AbstractVector{<:Integer} && length(u) == length(v) ||
            throw(ArgumentError("each move must be an integer joint-state vector"))
        b = BigInt.(u)
        iszero(sum(b)) && all(iszero, A * b) ||
            throw(ArgumentError("each move must lie in the integer kernel of [1'; A]"))
        products = map((1, -1)) do sign
            s = 0.0
            for i in eachindex(v)
                exponent = sign * b[i]
                exponent > 0 || continue
                v[i] == 0 && return 0.0
                # log(1) is exactly zero, even for exponents too large for Float64.
                v[i] == 1 || (s += Float64(exponent) * log(v[i]))
            end
            exp(s)
        end
        push!(result, products[1] - products[2])
    end
    return result
end
