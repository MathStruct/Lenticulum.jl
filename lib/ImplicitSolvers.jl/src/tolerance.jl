"""
    primal(x)

Return the value used for solver decisions and diagnostics. This is the identity by
default. With ForwardDiff loaded, it recursively unwraps dual numbers.
"""
primal(x) = x

"""
    Tolerance(; atol=nothing, rtol=nothing)

Require the residual norm to be at most `atol + rtol * scale`. Each `nothing` uses
`max(1e-9, 30eps(T))` in the primal residual's precision `T`. Root solves use the
iterate norm as their scale and linear solves use the right-hand-side norm.
"""
struct Tolerance{A<:Union{Nothing,Real},R<:Union{Nothing,Real}}
    atol::A
    rtol::R

    function Tolerance(atol::A, rtol::R) where {A<:Union{Nothing,Real},R<:Union{Nothing,Real}}
        _check_tolerance(:atol, atol); _check_tolerance(:rtol, rtol)
        return new{A,R}(atol, rtol)
    end
end
Tolerance(; atol=nothing, rtol=nothing) = Tolerance(atol, rtol)

# A negative or NaN bound can never be met by a zero residual, so a solver would run on past an
# exact solution (GLM's review of W1, F1): refused.
_check_tolerance(name, ::Nothing) = nothing
function _check_tolerance(name, x::Real)
    v = primal(x)
    (isnan(v) || v < 0) && throw(ArgumentError("$name must be non-negative and not NaN; got $x"))
    return nothing
end

_primal_type(::Type{T}) where {T} = typeof(primal(float(zero(T))))
_primal_norm(x) = _primal_type(eltype(x)) === eltype(x) ? norm(x) : norm(primal.(x))

function _bound(tol, scale, ::Type{T}) where {T}
    default = max(T(1e-9), T(30) * eps(T))
    atol = tol.atol === nothing ? default : T(primal(tol.atol))
    rtol = tol.rtol === nothing ? default : T(primal(tol.rtol))
    return iszero(rtol) ? atol : atol + rtol * T(scale)
end

function _status(residual, scale, tol)
    isfinite(residual) && isfinite(scale) || return :nonfinite
    return residual <= _bound(tol, scale, typeof(residual)) ? :converged : :continue
end

_unchanged(x, y) = all(iszero, primal.(x .- y))
