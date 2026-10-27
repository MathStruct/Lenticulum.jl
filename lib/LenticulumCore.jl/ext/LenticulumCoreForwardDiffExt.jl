module LenticulumCoreForwardDiffExt
using LenticulumCore
using ForwardDiff
LenticulumCore.stop_gradient(x::ForwardDiff.Dual) = ForwardDiff.value(x)
LenticulumCore.stop_gradient(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
LenticulumCore.stop_gradient(x::NamedTuple) = map(stop_gradient, x)
LenticulumCore.stop_gradient(x::Tuple) = map(stop_gradient, x)
LenticulumCore.stop_gradient(b::DiracBelief) = DiracBelief(stop_gradient(b.value))
LenticulumCore.stop_gradient(b::GaussianBelief) = GaussianBelief(stop_gradient(b.η), stop_gradient(b.Λ))
_replace(x::ForwardDiff.Dual{T}, y) where {T} = ForwardDiff.Dual{T}(y, ForwardDiff.partials(x))
_replace(x::Number, y) = y
function LenticulumCore.straight_through(f, s, x::ForwardDiff.Dual)
    return _replace(s(x), f(ForwardDiff.value(x)))
end
function LenticulumCore.straight_through(f, s, x::AbstractArray{<:ForwardDiff.Dual})
    return _replace.(s(x), f(ForwardDiff.value.(x)))
end
end
