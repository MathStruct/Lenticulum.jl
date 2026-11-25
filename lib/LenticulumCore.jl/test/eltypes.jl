using Test, LenticulumCore

include("support/eltype_matrix.jl")
using .EltypeMatrix

@testset "Core factor fixture coverage" begin
    # Only Lenticulum implements the collapsed Gaussian pair and exact Gaussian
    # product components. Both containers are exercised by its eltypes.jl.
    check_factor_coverage((), (LenticulumCore,); exclusions = Dict(
        ComposedFactor => "the supported collapsed game is implemented in Lenticulum",
        TensorFactor => "the exact product components are implemented in Lenticulum",
    ))
end
