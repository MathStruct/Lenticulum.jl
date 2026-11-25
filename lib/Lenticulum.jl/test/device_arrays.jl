using Test, Lenticulum, LenticulumCore, Mycelium

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :LenticulumEltypeFixtures) || include("support/eltype_fixtures.jl")
isdefined(@__MODULE__, :DeviceMatrix) || include("../../LenticulumCore.jl/test/support/device_matrix.jl")
include("support/m_step_fixtures.jl")

@testset "Lenticulum device fixtures" begin
    fixtures = LenticulumEltypeFixtures.fixtures()
    EltypeMatrix.check_factor_coverage(fixtures, (Lenticulum, LenticulumCore))
    flat = [(:RegimeFactor, Symbol(:message_, target, :_given_, given))
            for target in (:x, :y) for given in (:k, :latent)]
    DeviceMatrix.check_device_matrix(vcat(fixtures, MStepFixtures.fixtures()); flat)
end
