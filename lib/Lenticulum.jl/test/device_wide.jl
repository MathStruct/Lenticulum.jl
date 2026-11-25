# Q7's device matrix on wider fixtures (dimensions 2 and 3, several channels, diagonal and dense
# noise, M-steps of dimension 3 → 2); slow tier, about 3 minutes. LP-092.
using Test, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :DeviceMatrix) || include("../../LenticulumCore.jl/test/support/device_matrix.jl")
isdefined(@__MODULE__, :LenticulumWideFixtures) || include("support/wide_fixtures.jl")

@testset "Lenticulum device fixtures, wide" begin
    DeviceMatrix.check_device_matrix(vcat(LenticulumWideFixtures.fixtures(),
                                          LenticulumWideFixtures.mstep_fixtures()))
end
