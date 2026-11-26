# What JLArrays accepts that CUDA refuses (Q21).
#
#     julia --project=gpu gpu/jlarrays_vs_cuda.jl
#
# The package suites run their device files on JLArrays (CI has no GPU), and `gpu/runtests.jl` runs
# the same files on CUDA. A defect of the kind LP-093 and LP-095 met (a host array meeting a device
# array) shows on JLArrays only where JLArrays refuses the operation as CUDA does. Each probe below is
# one such operation, run on JLArrays and on CuArray with scalar indexing off; the table says which
# array type accepts it. A row where JLArrays accepts and CUDA refuses is a kind of defect that only
# the local GPU run catches. The result is recorded in `lib/LenticulumCore.jl/src/device_arrays.md`
# ("JLArrays against CUDA"). Pass `strict` to include the test hook first
# (`lib/LenticulumCore.jl/test/support/strict_jlarrays.jl`), as the package suites do.

using CUDA, JLArrays, LinearAlgebra
"strict" in ARGS && include(joinpath(@__DIR__, "..", "lib", "LenticulumCore.jl", "test", "support", "strict_jlarrays.jl"))

CUDA.functional() || error("gpu/jlarrays_vs_cuda.jl needs a working CUDA device")
CUDA.allowscalar(false)
JLArrays.allowscalar(false)

const n = 4
host_spd() = (A = randn(n, n); A * A' + n * I)

# name => function of the device constructor `D`; it returns nothing or throws.
const PROBES = [
    "matrix product, device * host" => D -> D(host_spd()) * host_spd(),
    "matrix product, host * device" => D -> host_spd() * D(host_spd()),
    "matrix-vector product, device * host vector" => D -> D(host_spd()) * randn(n),
    "mul! into a host matrix from device operands" => D -> mul!(zeros(n, n), D(host_spd()), D(host_spd())),
    "broadcast, device .+ host matrix" => D -> D(host_spd()) .+ host_spd(),
    "broadcast, device .* host vector" => D -> D(randn(n)) .* randn(n),
    "broadcast, device .+ host Bool matrix" => D -> D(host_spd()) .+ zeros(Bool, n, n),
    "copyto! device <- host" => D -> copyto!(D(zeros(n, n)), host_spd()),
    "dot(device, host)" => D -> dot(D(randn(n)), randn(n)),
    "isposdef(Symmetric(device))" => D -> isposdef(Symmetric(D(host_spd()))),
    "cholesky(Symmetric(device))" => D -> cholesky(Symmetric(D(host_spd()))),
    "device \\ device (LU)" => D -> D(host_spd()) \ D(randn(n)),
    "inv(device)" => D -> inv(D(host_spd())),
    "det(device)" => D -> det(D(host_spd())),
    "eigen(Symmetric(device))" => D -> eigen(Symmetric(D(host_spd()))),
    "tr(device)" => D -> tr(D(host_spd())),
    "diag(device)" => D -> diag(D(host_spd())),
    "device[1, 1] (scalar index)" => D -> D(host_spd())[1, 1],
    "Matrix(device) (copy to host)" => D -> Matrix(D(host_spd())),
    "view(device, 1:2, :) * device" => D -> view(D(host_spd()), 1:2, :) * D(host_spd()),
    "hcat(device, host)" => D -> hcat(D(host_spd()), host_spd()),
    "device Float64 .* Float32 device" => D -> D(host_spd()) .* D(Float32.(host_spd())),
]

outcome(f, D) = try
    f(D)
    "accepts"
catch err
    "refuses"
end

rows = [(name, outcome(f, JLArray), outcome(f, CuArray)) for (name, f) in PROBES]
w = maximum(length ∘ first, rows)
println(rpad("operation", w), "  JLArrays $(pkgversion(JLArrays))  CUDA $(pkgversion(CUDA))")
for (name, jl, cu) in rows
    flag = jl == "accepts" && cu == "refuses" ? "  <- only the GPU run catches it" :
           jl == "refuses" && cu == "accepts" ? "  <- JLArrays stricter" : ""
    println(rpad(name, w), "  ", rpad(jl, 15), "  ", cu, flag)
end
