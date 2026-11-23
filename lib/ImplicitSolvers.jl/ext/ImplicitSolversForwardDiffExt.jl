module ImplicitSolversForwardDiffExt

using ImplicitSolvers: ImplicitSolvers
using ForwardDiff: Dual, partials, value

ImplicitSolvers.primal(x::Dual) = ImplicitSolvers.primal(value(x))

const DualScalar = Union{Dual,Complex{<:Dual}}

# At Arnoldi breakdown the second entry is an exact zero, including its
# partials. Differentiating abs(0 + 0im) would contaminate the rotation with NaNs.
function ImplicitSolvers._gmres_hypot(a::Complex{D}, b::Complex{D}) where {D<:Dual}
    iszero(b) && return iszero(a) ? zero(real(a)) : abs(a)
    return hypot(abs(a), abs(b))
end

_dual_type(::Type{D}) where {D<:Dual} = D
_dual_type(::Type{Complex{D}}) where {D<:Dual} = D

# Match the tag being corrected. An operator that only captures an inner dual
# must retain it even when it is called on an input without the outer tag.
struct _Value{Tag} end
(strip::_Value)(x) = x
(strip::_Value{Tag})(x::Dual{Tag}) where {Tag} = value(x)
(strip::_Value)(x::Complex) = complex(strip(real(x)), strip(imag(x)))

struct _Partial{Tag}
    i::Int
end
(part::_Partial)(x) = zero(x)
(part::_Partial{Tag})(x::Dual{Tag}) where {Tag} = partials(x, part.i)
(part::_Partial)(x::Complex) = complex(part(real(x)), part(imag(x)))

struct _ReplaceValue end
(replace::_ReplaceValue)(x::Dual{Tag}, v) where {Tag} = Dual{Tag}(v, partials(x))
(replace::_ReplaceValue)(x::Complex, v) = complex(replace(real(x), real(v)), replace(imag(x), imag(v)))

struct _AddPartial
    i::Int
end
function (add::_AddPartial)(x::Dual{Tag,V,N}, dx) where {Tag,V,N}
    p = ntuple(j -> j == add.i ? partials(x, j) + dx : partials(x, j), Val(N))
    return Dual{Tag}(value(x), p)
end
(add::_AddPartial)(x::Complex, dx) = complex(add(real(x), real(dx)), add(imag(x), imag(dx)))

_strip_operator(f, strip) = v -> strip.(f(v))
_strip_operator(::Nothing, strip) = nothing

# The tangent system A ẋ = ḃ - Ȧx has residual ṙ = ḃ - partials(Ax), and its scale is that of its
# right-hand side, ‖A ẋ‖ at the solution. partials(Ax) = Ȧx + Aẋ alone is not a scale: with a plain b it
# equals -ṙ (LP-084, Mistral's F1). `gain` = ‖Ax‖/‖x‖ of the primal solve estimates ‖A ẋ‖ by gain·‖ẋ‖ without
# another product.
function _meets_tolerance(r, b, ax, x, gain, tol)
    P = ImplicitSolvers._primal_norm
    scale = max(P(b), P(ax), gain * P(x))
    return ImplicitSolvers._status(P(r), scale, tol) === :converged
end
function _meets_tolerance(r::AbstractVector{D}, b, ax, x, gain, tol) where {D<:DualScalar}
    return _nested_meets_tolerance(r, b, ax, x, gain, tol, _dual_type(D))
end
function _nested_meets_tolerance(r, b, ax, x, gain, tol, ::Type{Dual{Tag,V,N}}) where {Tag,V,N}
    strip = _Value{Tag}()
    _meets_tolerance(strip.(r), strip.(b), strip.(ax), strip.(x), gain, tol) || return false
    for i in 1:N
        part = _Partial{Tag}(i)
        _meets_tolerance(part.(r), part.(b), part.(ax), part.(x), gain, tol) || return false
    end
    return true
end

function _gain(x, ax)
    nx = ImplicitSolvers._primal_norm(x)
    return iszero(nx) ? zero(nx) : ImplicitSolvers._primal_norm(ax) / nx
end

function ImplicitSolvers._tangent_refine(apply, b, x::AbstractVector{D}, ax, r,
        report, alg, tol, left, right) where {D<:DualScalar}
    return _refine(apply, b, x, ax, r, report, alg, tol, left, right, _dual_type(D))
end

function _refine(apply, b, x, ax, r, report, alg, tol, left, right,
        ::Type{Dual{Tag,V,N}}) where {Tag,V,N}
    reason, steps, nfe = report.reason, report.iterations, report.nfe
    if r === nothing
        # MINRES returns immediately for a zero primal right-hand side.
        ax = apply(x)
        r = b .- ax
        nfe += 1
        report = ImplicitSolvers._report(reason, steps, report.residual, nfe)
    end

    strip = _Value{Tag}()
    op = _strip_operator(apply, strip)
    l = _strip_operator(left, strip)
    rr = _strip_operator(right, strip)
    if V <: Dual
        # First repair the inner solution. Changing it can change the mixed
        # residual through the outer derivative of the operator.
        v, inner = ImplicitSolvers._tangent_refine(op, strip.(b), strip.(x),
            strip.(ax), strip.(r), report, alg, tol, l, rr)
        if inner.nfe != nfe
            x = _ReplaceValue().(x, v)
            ax = apply(x)
            r = b .- ax
            reason, steps, nfe = inner.reason, inner.iterations, inner.nfe + 1
        end
    end

    gain = _gain(x, ax)
    for i in 1:N
        part = _Partial{Tag}(i)
        dr = part.(r)
        _meets_tolerance(dr, part.(b), part.(ax), part.(x), gain, tol) && continue
        dx, correction = ImplicitSolvers.linsolve(op, dr, alg; tol, left=l, right=rr)
        x = _AddPartial(i).(x, dx)
        steps += correction.iterations
        nfe += correction.nfe
        reason === :converged && (reason = correction.reason)
    end
    return x, ImplicitSolvers._report(reason, steps, report.residual, nfe)
end

end
