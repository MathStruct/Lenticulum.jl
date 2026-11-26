using GenericLinearAlgebra
using Test, AlgebraicFactors

include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
include("support/eltype_fixtures.jl")

@testset "AlgebraicFactors element-type fixtures" begin
    fixtures = AlgebraicEltypeFixtures.fixtures(; extended = true)
    EltypeMatrix.check_factor_coverage(fixtures, (AlgebraicFactors,))
    # Toric pointwise energy observes integer labels. Only η supplies a float type.
    parameter_only = EltypeMatrix.latent_cases(fixtures)
    push!(parameter_only, (:ToricFactor, :energy))
    EltypeMatrix.check_eltype_matrix(fixtures; parameter_only)
end
