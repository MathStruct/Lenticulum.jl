using ImplicitSolvers, LinearAlgebra, Test, Random
using ForwardDiff, JLArrays
include(joinpath(@__DIR__, "..", "..", "LenticulumCore.jl", "test", "support", "strict_jlarrays.jl"))  # JLArrays refuses host arrays in kernels, as CUDA does (Q21)

@testset verbose=true "ImplicitSolvers" begin
    include("tolerance_report.jl")
    include("linsolve.jl")
    include("krylov_oracle.jl")
    include("roots.jl")
    include("descent.jl")
    include("adjoint_fd.jl")
    include("device.jl")
    include("krylovjl.jl")
    include("ad_matrix.jl")
    include("aqua.jl")
end
