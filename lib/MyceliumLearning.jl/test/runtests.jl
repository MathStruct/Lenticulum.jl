using MyceliumLearning
using Mycelium
using LenticulumCore
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, Polarity
using Test

include("conformance.jl")
include("gradient_couplings.jl")
include("graph_adjoint.jl")

# Device (JLArrays) coverage is slow; it runs with LENTICULUM_DEVICE_TESTS=true.
if get(ENV, "LENTICULUM_DEVICE_TESTS", "false") == "true"
    include("graph_adjoint_devices.jl")
else
    @info "Skipping device tests; set LENTICULUM_DEVICE_TESTS=true to run them."
end

include("aqua.jl")
