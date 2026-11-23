"""
    NewtonDescent(; step=0.05, armijo=1e-4, min_damping=1e-8)

Use a Newton step with Armijo backtracking on the residual norm in stable regions.
Else subtract `step * residual`. If backtracking exhausts the damping budget,
take the first step at or below `min_damping`.
"""
struct NewtonDescent{S<:Real,A<:Real,D<:Real}
    step::S
    armijo::A
    min_damping::D
end

function NewtonDescent(; step=0.05, armijo=1e-4, min_damping=1e-8)
    0 < min_damping < 1 || throw(ArgumentError("min_damping must be between zero and one"))
    return NewtonDescent(step, armijo, min_damping)
end

"""
    TrustRegion(; radius=nothing, cg_maxiters=200, cg_rtol=1e-8, precision=nothing)

Minimize a supplied merit using quadratic models in a ball. Use an eigendecomposition
with a matrix Hessian and Steihaug CG with a Hessian action. The default initial radius
is `0.1 * sqrt(max(size(U0, 1), 1))`. The default arithmetic precision is the machine
epsilon of the primal iterate type. Pass `precision` when the merit is evaluated in
different arithmetic.
"""
struct TrustRegion{R<:Union{Nothing,Real},T<:Real,P<:Union{Nothing,Real}}
    radius::R
    cg_maxiters::Int
    cg_rtol::T
    precision::P
end

function TrustRegion(; radius=nothing, cg_maxiters=200, cg_rtol=1e-8, precision=nothing)
    precision === nothing || (isfinite(precision) && precision > 0) ||
        throw(ArgumentError("precision must be finite and positive"))
    cg_maxiters >= 0 || throw(ArgumentError("cg_maxiters must be nonnegative"))
    return TrustRegion(radius, Int(cg_maxiters), cg_rtol, precision)
end

"""
    solve_columns(problem, U0::AbstractMatrix, alg; tol::Tolerance, maxiters)
        -> (U, reports::Vector{SolverReport}, stable::Vector{Bool})

Solve independent root equations using [`NewtonDescent`](@ref) or [`TrustRegion`](@ref).
Supply `problem` as a named tuple of callbacks. The matrix `U` passed to a callback
contains just the columns to evaluate, in the order of their original indices in
`cols`. Callbacks should treat matrices passed to them as read-only.

- `residual(U, cols)` returns a residual matrix shaped like `U`.
- `scale(U, cols)` returns a vector of tolerance scales, one per column.
- For Newton descent, `system(U, cols)` returns `(flags, solves)`, a Boolean vector
  and a collection of functions `r -> J⁻¹*r`, both in column order.
- For the trust region, `merit(U, R, cols)` returns a vector of merit values.
  `model(U, R, cols)` returns a collection of `(g, H)` pairs, with gradient vector
  `g` and a symmetric matrix or a callable `v -> H*v`. `stable(U, cols)` returns
  a vector of flags for the reported points.

Optional `residual_norms(R)` supplies the column Euclidean norms when the caller
needs a particular reduction order. Newton descent also accepts `trial_residual`
with the same contract as `residual` for line searches that discard model state.
By default the driver uses primal column norms and the same residual callback at
all points. Set `round_steps = true` to store directions in the residual element
type and predicted decreases in the iterate element type before testing trials.
By default each direction and prediction retains its computed precision.
State, residuals and steps use the input array backend. Callback
collections of functions or models are host collections, as are the reports.
A trust-region radius below the arithmetic floor reports `:stalled` when the
root tolerance is unmet, including at a non-root Gauss–Newton minimum.
`nfe` counts all residual evaluations, including trials, for each column.
"""
function solve_columns(problem, U0::AbstractMatrix, alg::Union{NewtonDescent,TrustRegion};
                       tol::Tolerance, maxiters::Integer)
    maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
    U = float.(U0)
    B = size(U, 2)
    reports = Vector{SolverReport}(undef, B)
    stable = fill(false, B)
    B > 0 && _solve_columns!(U, reports, stable, problem, alg, tol, maxiters)
    return U, reports, stable
end

# Keep scalar metadata on the host, including reductions returned as device vectors.
_column_norms(R) = [_primal_norm(R[:, b]) for b in axes(R, 2)]
_column_norms(problem, R) = primal.(_host_array(get(problem, :residual_norms, _column_norms)(R)))

function _column_residual(eval, U, cols, nfe)
    R = eval(U, cols)
    R isa AbstractMatrix && size(R) == size(U) ||
        throw(DimensionMismatch("expected a residual matrix of size $(size(U))"))
    nfe[cols] .+= 1
    return R
end

function _column_status(nr, scale, tol, last, stalled=false)
    reason = _status(nr, scale, tol)
    reason !== :continue && return reason
    last && return :maxiters
    return stalled ? :stalled : :continue
end

function _record_columns!(reports, flags, cols, reasons, stable, nr, it, nfe)
    for (b, q) in enumerate(cols)
        reasons[b] === :continue && continue
        reports[q] = _report(reasons[b], it, nr[b], nfe[q])
        flags[q] = stable[b]
    end
end

function _solve_columns!(U, reports, flags, problem, alg::NewtonDescent, tol, maxiters)
    active = collect(axes(U, 2))
    nfe = zeros(Int, length(active))
    for it in 0:maxiters
        Ua = U[:, active]
        R = _column_residual(problem.residual, Ua, active, nfe)
        nr = _column_norms(problem, R)
        scale = primal.(_host_array(problem.scale(Ua, active)))
        reasons = [_column_status(nr[b], scale[b], tol, it == maxiters) for b in eachindex(active)]
        valid = findall(!=(:nonfinite), reasons)
        stable = fill(false, length(active))
        solves = if isempty(valid)
            ()
        else
            ss, actions = problem.system(Ua[:, valid], active[valid])
            stable[valid] .= _host_array(ss)
            actions
        end
        _record_columns!(reports, flags, active, reasons, stable, nr, it, nfe)
        keep = findall(==(:continue), reasons)
        isempty(keep) && break
        # `system` was evaluated on all finite columns, including those finishing now.
        actions = [solves[searchsortedfirst(valid, b)] for b in keep]
        active = active[keep]
        _newton_columns!(U, R[:, keep], nr[keep], active, stable[keep], actions, nfe, problem, alg)
    end
end

function _newton_columns!(U, R, nr, active, stable, solves, nfe, problem, alg)
    round_steps = get(problem, :round_steps, false)
    D = map(eachindex(active)) do b
        r = R[:, b]
        d = stable[b] ? -solves[b](r) : -alg.step .* r
        return round_steps ? (r .= d) : d
    end
    damping = ones(length(active))
    pending = findall(stable)
    trial_residual = get(problem, :trial_residual, problem.residual)
    while !isempty(pending)
        cols = active[pending]
        Ut = U[:, cols]
        for (b, a) in enumerate(pending)
            Ut[:, b] .+= damping[a] .* D[a]
        end
        nt = _column_norms(problem, _column_residual(trial_residual, Ut, cols, nfe))
        next = Int[]
        for (b, a) in enumerate(pending)
            nt[b] < (1 - alg.armijo * damping[a]) * nr[a] && continue
            damping[a] /= 2
            damping[a] > alg.min_damping && push!(next, a)
        end
        pending = next
    end
    for (b, q) in enumerate(active)
        U[:, q] .+= damping[b] .* D[b]
    end
end

function _solve_columns!(U, reports, flags, problem, alg::TrustRegion, tol, maxiters)
    active = collect(axes(U, 2))
    nfe = zeros(Int, length(active))
    T = _primal_type(eltype(U))
    radius = T(alg.radius === nothing ? 0.1 * sqrt(max(size(U, 1), 1)) : primal(alg.radius))
    Δ = fill(radius, length(active))
    εa = alg.precision === nothing ? eps(T) : primal(alg.precision)
    R = copy(_column_residual(problem.residual, U, active, nfe))
    # Preserve the merit's type and its initial batch of finite columns.
    valid = findall(isfinite, _column_norms(problem, R))
    values = isempty(valid) ? T[] : _host_array(problem.merit(U[:, valid], R[:, valid], active[valid]))
    f = similar(values, length(active))
    f[valid] .= values
    for it in 0:maxiters
        Ua, Ra = U[:, active], R[:, active]
        nr = _column_norms(problem, Ra)
        scale = primal.(_host_array(problem.scale(Ua, active)))
        reasons = [_column_status(nr[b], scale[b], tol, it == maxiters,
                        Δ[q] < 10εa * (1 + scale[b])) for (b, q) in enumerate(active)]
        ending = findall(r -> r !== :continue && r !== :nonfinite, reasons)
        stable = fill(false, length(active))
        if !isempty(ending)
            stable[ending] .= _host_array(problem.stable(Ua[:, ending], active[ending]))
        end
        _record_columns!(reports, flags, active, reasons, stable, nr, it, nfe)
        keep = findall(==(:continue), reasons)
        isempty(keep) && break
        active = active[keep]
        _trust_columns!(U, R, f, Δ, nr[keep], active, nfe, problem, alg, εa, T)
    end
end

function _trust_columns!(U, R, f, Δ, nr, active, nfe, problem, alg, εa, T)
    Ua, Ra = U[:, active], R[:, active]
    models = problem.model(Ua, Ra, active)
    round_steps = get(problem, :round_steps, false)
    steps = map(eachindex(active)) do b
        g, H = models[b]
        p = _tr_step(alg, g, H, Δ[active[b]])
        pred = -(dot(g, p) + dot(p, _apply_H(H, p)) / 2)
        if round_steps
            pred = convert(eltype(U), pred)
            if eltype(p) !== eltype(R)
                rounded = similar(p, eltype(R))
                rounded .= p
                p = rounded
            end
        end
        return (; p, pred)
    end
    Ut = copy(Ua)
    for (b, s) in enumerate(steps)
        Ut[:, b] .+= s.p
    end
    Rt = _column_residual(problem.residual, Ut, active, nfe)
    ft = _host_array(problem.merit(Ut, Rt, active))
    for (b, q) in enumerate(active)
        p, pred = steps[b]
        nt = _primal_norm(Rt[:, b])
        noisy = primal(pred) <= 100 * εa * max(abs(primal(f[q])), one(T))
        ratio = if !isfinite(nt)
            zero(T)
        elseif noisy
            nt < nr[b] ? one(T) : zero(T)
        else
            primal((f[q] - ft[b]) / pred)
        end
        np = _primal_norm(p)
        if ratio < 0.25
            Δ[q] = np / 4
        elseif ratio > 0.75 && np >= 0.99Δ[q]
            Δ[q] = 2Δ[q]
        end
        if ratio > 1e-4
            U[:, q] .= Ut[:, b]
            R[:, q] .= Rt[:, b]
            f[q] = ft[b]
        end
    end
end

_apply_H(H::AbstractMatrix, p) = H * p
_apply_H(H, p) = H(p)

# Use host factorization for the dense path, with bulk transfers for device vectors.
function _tr_step(::TrustRegion, g, H::AbstractMatrix, Δ)
    p = _exact_tr_step(_host_array(g), _host_hessian(H), Δ)
    g isa Array && return p
    return copyto!(similar(g, eltype(p)), p)
end

_host_hessian(H::Symmetric) = Symmetric(_host_array(parent(H)), Symbol(H.uplo))
_host_hessian(H::AbstractMatrix) = Symmetric(_host_array(H))

# Moré–Sorensen optimality conditions in an eigenbasis. In the hard case,
# augment the shifted solution along a lowest eigenvector.
function _exact_tr_step(g, H, Δ)
    E = eigen(H)
    λs, V = E.values, E.vectors
    gt = V' * g
    pof(λ) = -(V * (gt ./ (λs .+ λ)))
    λmin = first(λs)
    if λmin > 0
        p = pof(zero(λmin))
        norm(p) ≤ Δ && return p
    end
    lo = max(zero(λmin), -λmin)
    hi = lo + norm(g) / Δ + abs(lo) + one(λmin)
    while norm(pof(hi)) > Δ
        hi *= 2
    end
    if !(norm(pof(lo + sqrt(eps(λmin)) * (1 + abs(lo)))) > Δ)
        p = pof(lo + sqrt(eps(λmin)) * (1 + abs(lo)))
        v = V[:, 1]
        τ = _to_boundary(p, v, Δ)
        return p .+ τ .* v
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        norm(pof(mid)) > Δ ? (lo = mid) : (hi = mid)
    end
    return pof(hi)
end

# Steihaug's truncated CG stops on negative curvature, the boundary, or
# a sufficiently small linear residual.
function _tr_step(alg::TrustRegion, g, H, Δ)
    p = zero(g)
    r = copy(g)
    d = -r
    gn = _primal_norm(g)
    gn == 0 && return p
    for _ in 1:alg.cg_maxiters
        Hd = _apply_H(H, d)
        κ = dot(d, Hd)
        primal(κ) ≤ 0 && return p .+ _to_boundary(p, d, Δ) .* d
        α = dot(r, r) / κ
        pn = p .+ α .* d
        _primal_norm(pn) ≥ Δ && return p .+ _to_boundary(p, d, Δ) .* d
        rn = r .+ α .* Hd
        _primal_norm(rn) ≤ alg.cg_rtol * gn && return pn
        d = -rn .+ (dot(rn, rn) / dot(r, r)) .* d
        p, r = pn, rn
    end
    return p
end

# τ ≥ 0 with ‖p + τd‖ = Δ
function _to_boundary(p, d, Δ)
    a, b, c = dot(d, d), 2dot(p, d), dot(p, p) - Δ^2
    return (-b + sqrt(max(b^2 - 4a * c, zero(a)))) / (2a)
end
