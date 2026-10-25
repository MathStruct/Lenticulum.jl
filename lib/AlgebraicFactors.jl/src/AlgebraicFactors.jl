"""
    AlgebraicFactors

Exact finite log-linear factors for Lenticulum and Mycelium. See [`ToricFactor`](@ref),
[`independence`](@ref), and [`hierarchical`](@ref). [`fit_mle`](@ref) fits moments;
[`inmodel`](@ref) and [`binomial_residuals`](@ref) check supplied probabilities.
"""
module AlgebraicFactors

using LinearAlgebra: LinearAlgebra
using Random: AbstractRNG
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium

export ToricFactor, independence, hierarchical
export fit_mle, inmodel, binomial_residuals

include("toric.jl")
include("fit.jl")

end
