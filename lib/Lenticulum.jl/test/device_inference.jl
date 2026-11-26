using Test, Lenticulum, LenticulumCore, Mycelium

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :LenticulumEltypeFixtures) || include("support/eltype_fixtures.jl")
isdefined(@__MODULE__, :DeviceMatrix) || include("../../LenticulumCore.jl/test/support/device_matrix.jl")

# Class and assignment indices, offsets and channel names describe the cluster topology.
DeviceMatrix.device_payload(b::Union{Mycelium.FrozenClusterBelief,Mycelium.FrozenClusterBeliefs}) =
    DeviceMatrix.device_payload((b.q, b.M, b.Σ, b.H, b.points))

isdefined(DeviceMatrix, :device_block_payload) || include("../../Mycelium.jl/test/support/device_blocks.jl")

@testset "Lenticulum device inference fixtures" begin
    fixtures = LenticulumEltypeFixtures.fixtures(; extended = true, raw_blocks = true)
    flat = [(:RegimeFactor, op) for f in fixtures if f.name === :RegimeFactor
            for (op, _) in f.cases if any(s -> endswith(string(op), s),
                                         ("_x_given_k", "_x_given_latent", "_y_given_k", "_y_given_latent"))]
    DeviceMatrix.check_device_matrix(fixtures; flat, device_views = true)
end
