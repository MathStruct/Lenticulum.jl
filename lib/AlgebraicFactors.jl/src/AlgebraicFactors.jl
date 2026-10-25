"""
    AlgebraicFactors

Exact finite log-linear factors for Lenticulum and Mycelium. See [`ToricFactor`](@ref),
[`independence`](@ref), and [`hierarchical`](@ref). Fitting and other algebraic families
are not implemented.
"""
module AlgebraicFactors

using LinearAlgebra: LinearAlgebra
using Random: AbstractRNG
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium

export ToricFactor, independence, hierarchical

include("toric.jl")

end
