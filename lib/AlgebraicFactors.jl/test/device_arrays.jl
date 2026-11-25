using Test, AlgebraicFactors

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :AlgebraicEltypeFixtures) || include("support/eltype_fixtures.jl")
include("../../LenticulumCore.jl/test/support/device_matrix.jl")

@testset "AlgebraicFactors device fixtures" begin
    fixtures = AlgebraicEltypeFixtures.fixtures()
    EltypeMatrix.check_factor_coverage(fixtures, (AlgebraicFactors,))
    DeviceMatrix.check_device_matrix(fixtures)
end
