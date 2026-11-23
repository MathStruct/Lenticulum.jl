using ImplicitSolvers, LinearAlgebra, Test, Random
using ForwardDiff, JLArrays

@testset verbose=true "ImplicitSolvers" begin
    include("tolerance_report.jl")
    include("linsolve.jl")
    include("krylov_oracle.jl")
    include("roots.jl")
    include("adjoint_fd.jl")
    include("device.jl")
    include("aqua.jl")
end
