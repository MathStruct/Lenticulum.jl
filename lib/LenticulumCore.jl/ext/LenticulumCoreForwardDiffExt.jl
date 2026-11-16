module LenticulumCoreForwardDiffExt
using LenticulumCore
using ForwardDiff
LenticulumCore.stop_gradient(x::ForwardDiff.Dual) = ForwardDiff.value(x)
LenticulumCore.stop_gradient(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
LenticulumCore.stop_gradient(x::NamedTuple) = map(stop_gradient, x)
LenticulumCore.stop_gradient(x::Tuple) = map(stop_gradient, x)
LenticulumCore.stop_gradient(b::DiracBelief) = DiracBelief(stop_gradient(b.value))
LenticulumCore.stop_gradient(b::GaussianBelief) = GaussianBelief(stop_gradient(b.η), stop_gradient(b.Λ), b.improper)
_replace(x::ForwardDiff.Dual{T}, y) where {T} = ForwardDiff.Dual{T}(y, ForwardDiff.partials(x))
_replace(x::Number, y) = y
function LenticulumCore.straight_through(f, s, x::ForwardDiff.Dual)
    return _replace(s(x), f(ForwardDiff.value(x)))
end
function LenticulumCore.straight_through(f, s, x::AbstractArray{<:ForwardDiff.Dual})
    return _replace.(s(x), f(ForwardDiff.value.(x)))
end

# A tangent that changes a rank to first order (X = LᵀΔR ≠ 0, L and R the null spaces at the value)
# has no derivative in the extended Gaussian's decompositions (`extended_gaussian.jl`, P4). `scale`
# is the size of the nonzero spectrum; the tolerance is relative to it and to the tangent.
function LenticulumCore._check_constant_rank(X::AbstractArray{ForwardDiff.Dual{Tg,V,P}}, Δ, scale, what) where {Tg,V,P}
    isempty(X) && return nothing
    for k in 1:P
        Xk = map(x -> ForwardDiff.partials(x, k), X)
        Dk = map(x -> ForwardDiff.partials(x, k), Δ)
        tol = sqrt(eps(float(V))) * max(maximum(abs, Dk; init = zero(V)), scale)
        maximum(abs, Xk; init = zero(V)) <= tol || throw(ArgumentError(
            "$what: a derivative that changes its rank at first order has no derivative there \
             (the value jumps; extended_gaussian.md, \"Derivatives\")"))
    end
    return nothing
end
end
