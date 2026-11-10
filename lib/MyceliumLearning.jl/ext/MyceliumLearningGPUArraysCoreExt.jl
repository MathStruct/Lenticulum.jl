module MyceliumLearningGPUArraysCoreExt
using MyceliumLearning: MyceliumLearning as ML
using Mycelium: Mycelium as M
using LenticulumCore: GaussianBelief, isproper, belief_mean
using LinearAlgebra: Diagonal
using Random: randn
using GPUArraysCore: AbstractGPUArray

function ML._replace_coordinates(θ::AbstractGPUArray, indices, active)
    positions = Dict(i => k for (k,i) in enumerate(indices))
    ii = [haskey(positions,i) ? length(θ)+positions[i] : i for i in eachindex(θ)]
    return M._gather_active(vcat(θ, active), ii)
end

function ML._draw(rng, b::GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    isproper(b) || throw(ArgumentError("score sampling requires a proper Gaussian message"))
    μ = belief_mean(b)
    # Explicit host RNG context, one bulk transfer; no hardware RNG claim.
    noise = similar(μ)
    copyto!(noise, randn(rng, length(μ)))
    return μ .+ noise ./ sqrt.(b.Λ.diag)
end

end
