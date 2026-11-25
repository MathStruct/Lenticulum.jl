# The device tests of every package on a CUDA GPU (Q8).
#
#     julia --project=gpu -e 'using Pkg; Pkg.instantiate()'     # once; its Manifest is not tracked
#     julia --project=gpu gpu/runtests.jl                        # all packages
#     julia --project=gpu gpu/runtests.jl Lenticulum Mycelium   # some
#
# The same files run on JLArrays in each package's own suite. Here `Main.LENTICULUM_DEVICE_ARRAY`
# is CuArray, which each file reads into its `DeviceArray` (and `DeviceMatrix.device_array()`),
# so every device array of the tests is a CuArray, with scalar indexing off. JLArrays runs its
# kernels on the CPU and so accepts what CUDA refuses: a broadcast that mixes a host and a device
# array, a kernel that is not GPU-compilable. That is what this job catches. Each package's files
# are included into a module of their own, after the `using` lines its runtests.jl gives them.
# `.github/workflows/GPU.yml` runs this weekly on a self-hosted runner with a GPU.

using CUDA, Test

CUDA.functional() || error("gpu/runtests.jl needs a working CUDA device; see CUDA.versioninfo()")
const LENTICULUM_DEVICE_ARRAY = CuArray
ENV["LENTICULUM_DEVICE_TESTS"] = "true"     # Mycelium's device files check it
CUDA.allowscalar(false)

const LIB = normpath(joinpath(@__DIR__, "..", "lib"))

# package => (the `using` lines its files expect, the files under its test/). A file listed before
# the device files defines helpers they use (and runs its own CPU tests on the way).
const SUITES = [
    :ImplicitSolvers => (:(using ImplicitSolvers, LinearAlgebra, Test, Random, ForwardDiff, JLArrays, Krylov),
                         ["tolerance_report.jl", "krylov_oracle.jl", "device.jl", "krylovjl.jl"]),
    :LenticulumCore => (:(using LenticulumCore, LinearAlgebra, Test),
                        ["device_arrays.jl", "cuda_ext.jl"]),
    :Mycelium => (:(using Mycelium, LenticulumCore, LinearAlgebra, Test;
                    using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, Polarity),
                  ["lp019.jl", "device_improper.jl"]),
    :MyceliumLearning => (:(using MyceliumLearning, Mycelium, LenticulumCore, LinearAlgebra, Test;
                            using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, Polarity),
                          ["graph_adjoint.jl", "graph_adjoint_devices.jl"]),
    :Lenticulum => (:(using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, LuxCore, Random, Test;
                      using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent),
                    ["device_arrays.jl", "device_graphs.jl", "lp092.jl", "device_wide.jl"]),
    :AlgebraicFactors => (:(using AlgebraicFactors, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Test),
                          ["device_arrays.jl"]),
    :ImplicitLayers => (:(using ImplicitLayers, LenticulumCore, Mycelium, MyceliumLearning, LuxCore, LinearAlgebra, Random, Test;
                          using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Polarity),
                        ["device_arrays.jl"]),
]

const CHOSEN = isempty(ARGS) ? first.(SUITES) : Symbol.(ARGS)
for name in CHOSEN
    name in first.(SUITES) || error("unknown package $name; known: $(join(first.(SUITES), ", "))")
end

CUDA.versioninfo()
@testset verbose = true "Device tests on CUDA ($(CUDA.name(CUDA.device())))" begin
    for (name, (preamble, files)) in SUITES
        name in CHOSEN || continue
        @testset "$name" begin
            mod = Module(Symbol(name, :DeviceTests))
            Core.eval(mod, :(include(path) = Base.include($mod, path)))
            Core.eval(mod, preamble)
            for file in files
                Base.include(mod, joinpath(LIB, "$name.jl", "test", file))
            end
        end
    end
end
