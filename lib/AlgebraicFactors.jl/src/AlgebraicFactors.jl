"""
    AlgebraicFactors

Exact finite log-linear factors for Lenticulum and Mycelium. See [`ToricFactor`](@ref),
[`independence`](@ref), and [`hierarchical`](@ref). [`fit_mle`](@ref) fits moments;
[`inmodel`](@ref) and [`binomial_residuals`](@ref) check supplied probabilities.
[`Grassmann`](@ref) and [`Retracted`](@ref) provide Grassmannian parameter updates.
[`PluckerFactor`](@ref) transports beliefs through the bivector map [`plucker`](@ref).
[`ExponentTable`](@ref) and [`PolynomialFactor`](@ref) provide polynomial residuals
and Gaussian messages via [`PolynomialModel`](@ref), with optional multi-branch
inference through [`HomotopyBranches`](@ref);
[`fit_nullspace`](@ref) and [`fit_sampson`](@ref) fit complete data.
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

export plucker_pairs, plucker, plucker_jacobian, plucker_relations, plucker_residuals
export plucker_moments, PluckerFactor
export ExponentTable, veronese, veronese_jacobian, basis_change
export HomotopyBranches, polynomial_branches, branch_entropy_bounds
export PolynomialFactor, PolynomialModel, fit_nullspace, fit_sampson

include("plucker.jl")
include("toric.jl")
include("fit.jl")
include("grassmann.jl")
include("veronese.jl")
include("homotopy.jl")
include("polynomial.jl")
include("polyfit.jl")

end
