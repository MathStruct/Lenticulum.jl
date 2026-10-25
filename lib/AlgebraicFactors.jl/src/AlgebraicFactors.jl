"""
    AlgebraicFactors

Exact finite log-linear factors for Lenticulum and Mycelium. See [`ToricFactor`](@ref),
[`independence`](@ref), and [`hierarchical`](@ref). [`fit_mle`](@ref) fits moments;
[`inmodel`](@ref) and [`binomial_residuals`](@ref) check supplied probabilities.
[`Grassmann`](@ref) and [`Retracted`](@ref) provide Grassmannian parameter updates.
"""
module AlgebraicFactors

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium

export ToricFactor, independence, hierarchical
export fit_mle, inmodel, binomial_residuals
export Grassmann, project_tangent, retract, ispoint, Retracted

include("toric.jl")
include("fit.jl")
include("grassmann.jl")

end
