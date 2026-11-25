using Test, AlgebraicFactors

include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
include("support/eltype_fixtures.jl")

@testset "AlgebraicFactors element-type fixtures" begin
    fixtures = AlgebraicEltypeFixtures.fixtures()
    EltypeMatrix.check_factor_coverage(fixtures, (AlgebraicFactors,))
    broken = Dict((:PolynomialFactor, op, :BigFloat) => "Residual rank checks require LAPACK"
                  for op in (:message_x_given_y, :message_y_given_x, :local_free_energy))
    EltypeMatrix.check_eltype_matrix(fixtures; broken)
end
