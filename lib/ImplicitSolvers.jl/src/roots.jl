abstract type AbstractRootSolver end

"""
    Picard(; damping=1)

Fixed-point iteration `u ← u - damping * F(u)`. A contractive iteration converges
from initial guesses in its contraction region.
"""
struct Picard{T<:Real} <: AbstractRootSolver
    damping::T
end
Picard(; damping=1) = Picard(damping)

"""
    Broyden(; init_scale=1)

Good inverse Broyden with initial inverse `init_scale * I`. Stores a dense
approximate inverse Jacobian and updates it from successive secant pairs.
"""
struct Broyden{T<:Real} <: AbstractRootSolver
    init_scale::T
end
Broyden(; init_scale=1) = Broyden(init_scale)

"""
    LimitedBroyden(; memory=20, init_scale=1)

Good inverse Broyden with at most `memory` rank-one corrections to `init_scale * I`.
Drops the oldest correction when the history fills.
"""
struct LimitedBroyden{T<:Real} <: AbstractRootSolver
    memory::Int
    init_scale::T

    function LimitedBroyden(; memory::Int=20, init_scale::Real=1)
        memory > 0 || throw(ArgumentError("memory must be positive"))
        return new{typeof(init_scale)}(memory, init_scale)
    end
end

"""
    solve_root(F, u0::AbstractVector, alg; tol=Tolerance(), maxiters=100, history=nothing)

Return `(u, report)` for the root equation `F(u) = 0`. Supply an out-of-place residual
with the same length as `u0`. The report counts the initial evaluation and one
evaluation per completed step. With [`LimitedBroyden`](@ref), pass a `Ref{Any}()` as
`history` to receive `(; a, b, scale)` describing the final inverse approximation
`scale * I + sum(ai * bi')`.
"""
function solve_root(F, u0::AbstractVector, alg::AbstractRootSolver;
                    tol::Tolerance=Tolerance(), maxiters::Int=100, history=nothing)
    maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
    history === nothing || alg isa LimitedBroyden ||
        throw(ArgumentError("history is available with LimitedBroyden"))
    u = float.(u0)
    r = _vector_value(F(u), length(u))
    n = _primal_norm(r)
    status = _status(n, _primal_norm(u), tol)
    if status !== :continue || iszero(maxiters)
        history === nothing || (history[] = _root_state(alg, u, r))
        reason = status === :continue ? :maxiters : status
        return u, _report(reason, 0, n, 1)
    end

    state = _root_state(alg, u, r)
    history === nothing || (history[] = state)
    for k in 1:maxiters
        step = _root_step(state, alg, r)
        unew = u .+ step
        rnew = _vector_value(F(unew), length(u))
        n = _primal_norm(rnew)
        status = _status(n, _primal_norm(unew), tol)
        status === :nonfinite && return unew, _report(status, k, n, k + 1)
        if status === :continue && _unchanged(unew, u)
            return unew, _report(:stalled, k, n, k + 1)
        end
        _update_inverse!(state, alg, rnew, r, step)
        status === :converged && return unew, _report(status, k, n, k + 1)
        u, r = unew, rnew
    end
    return u, _report(:maxiters, maxiters, n, maxiters + 1)
end

function _vector_value(v, n)
    v isa AbstractVector && length(v) == n ||
        throw(DimensionMismatch("expected an output vector of length $n"))
    return v
end

function _root_coefficient(u, r, x)
    T = promote_type(_primal_type(eltype(u)), _primal_type(eltype(r)))
    scale = T(primal(x))
    isfinite(scale) ||
        throw(ArgumentError("root solver coefficients must be finite in the working precision"))
    return scale
end

_root_state(alg::Picard, u, r) = _root_coefficient(u, r, alg.damping)

function _root_state(alg::Broyden, u, r)
    scale = _root_coefficient(u, r, alg.init_scale)
    T = promote_type(typeof(scale), eltype(u), eltype(r))
    H = similar(u, T, length(u), length(u))
    indices = 1:length(u)
    H .= scale .* (indices .== indices')
    return H
end

function _root_state(alg::LimitedBroyden, u, r)
    scale = _root_coefficient(u, r, alg.init_scale)
    T = promote_type(typeof(scale), eltype(u), eltype(r))
    V = typeof(similar(u, T))
    return (; a=V[], b=V[], scale)
end

# H = scale * I + Σ aᵢ bᵢ'. Swapping the histories applies H'.
function _broyden_product(a, b, scale, v)
    out = scale .* v
    for (ai, bi) in zip(a, b)
        out .+= ai .* dot(bi, v)
    end
    return out
end

_root_step(scale, ::Picard, r) = (-scale) .* r
_root_step(H, ::Broyden, r) = -(H * r)
_root_step(h, ::LimitedBroyden, r) = -_broyden_product(h.a, h.b, h.scale, r)

# Normalizing the secant vectors makes the denominator guard relative to ‖step‖‖HΔr‖.
# Keeping the norms as primal scaling factors also avoids forming their product.
function _broyden_update(step, Hdr)
    nstep, nHdr = _primal_norm(step), _primal_norm(Hdr)
    (isfinite(nstep) && isfinite(nHdr) && nstep > 0 && nHdr > 0) || return nothing
    direction = step ./ nstep
    denom = dot(direction, Hdr ./ nHdr)
    value = abs(primal(denom))
    value > sqrt(eps(typeof(value))) || return nothing
    return ((step .- Hdr) ./ nHdr ./ denom, direction)
end

_update_inverse!(scale, ::Picard, rnew, r, step) = nothing

function _update_inverse!(H, ::Broyden, rnew, r, step)
    update = _broyden_update(step, H * (rnew .- r))
    update === nothing && return
    left, direction = update
    right = H' * direction
    H .+= left .* right'
    return nothing
end

function _update_inverse!(h, alg::LimitedBroyden, rnew, r, step)
    update = _broyden_update(step, _broyden_product(h.a, h.b, h.scale, rnew .- r))
    update === nothing && return
    left, direction = update
    right = _broyden_product(h.b, h.a, h.scale, direction)
    if length(h.a) == alg.memory
        popfirst!(h.a)
        popfirst!(h.b)
    end
    push!(h.a, left)
    push!(h.b, right)
    return nothing
end
