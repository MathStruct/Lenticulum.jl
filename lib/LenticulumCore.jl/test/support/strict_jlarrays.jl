# Make JLArrays refuse a host Array inside a kernel, as CUDA does (Q21).
#
# JLArrays runs its kernels on the CPU, so a broadcast that mixes a host Array and a JLArray works
# there and fails on a GPU (LP-093, LP-095: host placeholders meeting device blocks). JLArrays 0.5
# converts each kernel argument with `adapt(JLArrays.Adaptor(), arg)`, and Adapt passes a host
# Array through unchanged; this method throws instead. CUDA refuses the same argument when it
# compiles the kernel. `gpu/jlarrays_vs_cuda.jl` lists what JLArrays still accepts that CUDA
# refuses (a host vector in a matrix-vector product, `dot` of a device and a host vector).
#
# Test-only type piracy, installed once per process by each package's runtests.jl. It touches
# kernel arguments only: `copyto!`, `adapt(JLArray, x)` and constructors from host arrays are
# unaffected.

using JLArrays: JLArrays

if pkgversion(JLArrays) >= v"0.5" &&
   which(JLArrays.Adapt.adapt_storage, Tuple{JLArrays.Adaptor,Array{Float64,1}}).module === JLArrays.Adapt
    @eval JLArrays.Adapt.adapt_storage(::JLArrays.Adaptor, x::Array) =
        throw(ArgumentError(string("a host ", typeof(x), " of size ", size(x), " reached a JLArray kernel; ",
                                   "CUDA refuses it too (Q21, lib/LenticulumCore.jl/test/support/strict_jlarrays.jl)")))
end
