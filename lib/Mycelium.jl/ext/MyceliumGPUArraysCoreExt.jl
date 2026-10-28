module MyceliumGPUArraysCoreExt
using Mycelium: Mycelium as M
using LenticulumCore: stop_gradient, GaussianBelief, DiracBelief, isproper, belief_mean
using LinearAlgebra: Diagonal
using Random: randn
using LenticulumCore: dimension
using GPUArraysCore: AbstractGPUArray
# Upload host integer metadata so the indexing pullback uses the device
# gather/scatter rule instead of iterating an index vector on the host.
function M._index_metadata(θ::AbstractGPUArray, indices)
    ii = similar(θ, Int, length(indices))
    copyto!(ii, indices)
    return ii
end
M._gather_active(θ::AbstractGPUArray, indices) = θ[M._index_metadata(θ, indices)]
function M._mask_gradient(θ::AbstractGPUArray, indices)
    mask = [i in indices for i in eachindex(θ)]
    return ifelse.(M._index_metadata(θ, mask) .!= 0, θ, zero(eltype(θ)))
end
function M._pin_coordinates(θ::AbstractGPUArray, indices)
    mask = [i in indices for i in eachindex(θ)]
    return ifelse.(M._index_metadata(θ, mask) .!= 0, θ, stop_gradient(θ))
end
function M._replace_coordinates(θ::AbstractGPUArray, indices, active)
    positions = Dict(i => k for (k,i) in enumerate(indices))
    ii = [haskey(positions,i) ? length(θ)+positions[i] : i for i in eachindex(θ)]
    return M._gather_active(vcat(θ, active), ii)
end

# Device parameters retain their precision even with the default Float64 rule.
M.rule_put(r::M.GradientDescent, state, p::AbstractGPUArray, g) =
    (state, p .- convert(eltype(p), r.η) .* g)
M.rule_get(r::M.Nesterov, state, p::AbstractGPUArray) =
    p .+ convert(eltype(p), r.γ) .* state
for Rule in (:Momentum, :Nesterov)
    @eval function M.rule_put(r::M.$Rule, state, p::AbstractGPUArray, g)
        s = convert(eltype(p), r.γ) .* state .- convert(eltype(p), r.η) .* g
        return s, p .+ s
    end
end
function M._draw(rng, b::GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    isproper(b) || throw(ArgumentError("score sampling requires a proper Gaussian message"))
    μ = belief_mean(b)
    # Explicit host RNG context, one bulk transfer; no hardware RNG claim.
    noise = similar(μ)
    copyto!(noise, randn(rng, length(μ)))
    return μ .+ noise ./ sqrt.(b.Λ.diag)
end

function M.belief_distance(a::GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}},
        b::GaussianBelief{<:AbstractGPUArray,<:Diagonal{S,<:AbstractGPUArray}}) where {T,S}
    dimension(a)==dimension(b) || return Inf
    # Convergence diagnostics are metadata, not differentiated objectives.
    dη = stop_gradient(a.η .- b.η)
    dλ = stop_gradient(a.Λ.diag .- b.Λ.diag)
    return max(maximum(abs,dη;init=zero(eltype(dη))),
               maximum(abs,dλ;init=zero(eltype(dλ))))
end

function M.belief_logdensity(b::GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}, x::AbstractVector) where {T}
    isproper(b) || throw(ArgumentError("belief_logdensity requires a proper Gaussian message"))
    d = x .- belief_mean(b)
    return -T(dimension(b))*log(T(2)*T(π))/2 + sum(log,b.Λ.diag)/2 -
           sum(b.Λ.diag .* abs2.(d))/2
end

function M._damp(a::DiracBelief{<:AbstractGPUArray}, b::DiracBelief{<:AbstractGPUArray}, α::Real)
    w = convert(eltype(a.value), α)
    return DiracBelief(w .* a.value .+ (one(w) - w) .* b.value)
end
function M._damp(a::GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}},
        b::GaussianBelief{<:AbstractGPUArray,<:Diagonal{S,<:AbstractGPUArray}}, α::Real) where {T,S}
    w = convert(T, α)
    return GaussianBelief(w .* a.η .+ (one(w) - w) .* b.η,
        Diagonal(w .* a.Λ.diag .+ (one(w) - w) .* b.Λ.diag))
end
function M.variable_entropy(b::GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    isproper(b) || throw(ArgumentError("variable_entropy requires a proper Gaussian message"))
    return T(dimension(b))*(one(T)+log(T(2)*T(π)))/2 - sum(log,b.Λ.diag)/2
end
end
