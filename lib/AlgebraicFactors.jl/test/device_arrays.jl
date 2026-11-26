using Test, AlgebraicFactors

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :AlgebraicEltypeFixtures) || include("support/eltype_fixtures.jl")
include("../../LenticulumCore.jl/test/support/device_matrix.jl")

# The free coordinates are integer indices into the joint, kept on the host.
DeviceMatrix.device_payload(b::AlgebraicFactors.PolynomialBelief) =
    DeviceMatrix.device_payload((b.z0, b.m, b.S, b.H, b.basis))

isdefined(DeviceMatrix, :device_block_payload) || include("../../Mycelium.jl/test/support/device_blocks.jl")

@testset "AlgebraicFactors device fixtures" begin
    fixtures = AlgebraicEltypeFixtures.fixtures(; extended = true, raw_blocks = true)
    EltypeMatrix.check_factor_coverage(fixtures, (AlgebraicFactors,))
    DeviceMatrix.check_device_matrix(fixtures; device_views = true)
end
