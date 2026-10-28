module LenticulumCoreGPUArraysCoreExt
using LenticulumCore: LenticulumCore as LC
using GPUArraysCore: AbstractGPUArray, AnyGPUArray
using LinearAlgebra: Diagonal

# Diagonal Gaussians need elementwise division/logs, not an unavailable device
# factorisation. Keep dense precisions explicit rather than silently copying them.
function LC._precision_factor(M::Diagonal{T,<:AbstractGPUArray}) where {T}
    all(x -> isfinite(x) && x > 0, M.diag) || throw(ArgumentError(
        "Gaussian device precision must have finite positive diagonal entries"))
    return M
end
LC._precision_solve(M::Diagonal{T,<:AbstractGPUArray}, η) where {T} = η ./ M.diag
LC._precision_logdet(M::Diagonal{T,<:AbstractGPUArray}) where {T} = sum(log, M.diag)
LC._precision_isproper(M::Diagonal{T,<:AbstractGPUArray}) where {T} =
    all(x -> isfinite(x) && x > 0, M.diag)
LC.isuninformative(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T} =
    all(iszero,b.Λ.diag) && all(iszero,b.η)
function LC.belief_cov(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    LC._precision_factor(b.Λ)
    return Diagonal(inv.(b.Λ.diag))
end
function LC.Gaussian(μ::AbstractGPUArray{T,1}, Σ::Diagonal{S,<:AbstractGPUArray}) where {T,S}
    LC._precision_factor(Σ)
    λ = inv.(Σ.diag)
    return LC.GaussianBelief(λ .* μ, Diagonal(λ))
end
function LC.belief_mean(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    LC._precision_factor(b.Λ)
    return b.η ./ b.Λ.diag
end
function LC.logpartition(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    LC._precision_factor(b.Λ)
    return sum(b.η .* (b.η ./ b.Λ.diag))/2 - sum(log, b.Λ.diag)/2 +
           T(LC.dimension(b))*log(T(2)*T(π))/2
end
LC._chol(M::Diagonal{T,<:AbstractGPUArray}, name) where {T} = LC._precision_factor(M)
LC._chol(M::AbstractGPUArray, name) = throw(ArgumentError(
    "Dense device Gaussian covariance factorisation is unsupported; use Diagonal device covariance"))
LC._precision_factor(M::AbstractGPUArray) = throw(ArgumentError(
    "Dense device Gaussian precision factorisation is unsupported; use Diagonal device precision"))
LC._precision_isproper(M::AbstractGPUArray) = throw(ArgumentError(
    "Dense device Gaussian precision factorisation is unsupported; use Diagonal device precision"))
LC._rebuild(θ::AbstractGPUArray, o::Int, x::Number) = throw(ArgumentError(
    "Device parameter vectors require array leaves; use a one-element device array for scalar parameters"))

# Empty tuple branches otherwise contribute host Float64[] vectors to vcat,
# selecting host concatenation and scalar-indexing the active device leaves.
function LC._flatten_active_tree(leaf::AnyGPUArray, x)
    parts = filter(!isempty, map(LC._flatten_active, values(x)))
    return isempty(parts) ? vec(copy(leaf)) : reduce(vcat, parts)
end

LC._jacobian(::Nothing, fun, u::AbstractGPUArray, args...; ε=1e-6) =
    throw(ArgumentError("Device finite-difference Jacobians are unsupported; pass ad=AutoForwardDiff() or AutoZygote()"))
LC._gradient(::Nothing, fun, u::AbstractGPUArray, args...; ε=1e-6) =
    throw(ArgumentError("Device finite-difference gradients are unsupported; pass ad=AutoForwardDiff() or AutoZygote()"))

end
