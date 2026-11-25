# Dense SPD algebra of CUDA matrices through cuSOLVER and cuBLAS (Q8). The GPUArraysCore
# extension's generic kernels launch one or more kernels per column; for Float32 and Float64
# these replace them by `potrf` and `trsm`, 2–30× faster on an RTX 3080 from n = 2 to 1024
# (`gpu/device_benchmarks.md`). Other element types (dual numbers, integers) keep the generic path.
module LenticulumCoreCUDAExt

using LenticulumCore: LenticulumCore as LC
using CUDA: CuMatrix, CUSOLVER
using LinearAlgebra: LinearAlgebra, LowerTriangular, PosDefException, checksquare, triu!

const CuBlasReal = Union{Float32,Float64}

# The upper triangle is read, as the CPU path's `cholesky(Symmetric(A))` does. cuSOLVER reports
# a nonpositive pivot but not a NaN one; the generic kernel does (its test is `!(pivot > 0)`, as
# reference LAPACK's `dpotrf2`), and this keeps the two device paths alike. A NaN anywhere in the
# triangle reaches the diagonal of the factor, and the first NaN on it is the generic kernel's
# pivot. (OpenBLAS's `potrf`, the CPU path, does not flag NaN either: `info = 0`; LP-093.) One reduction and one synchronisation for that check (`findfirst` only on failure),
# about 13 µs of the 45 µs a 2 × 2 factorisation takes; the generic kernel takes 41 µs there, and
# cuSOLVER is faster from n = 8 on.
function LC._symmetric_cholesky(A::CuMatrix{T}; check = true) where {T<:CuBlasReal}
    n = checksquare(A)
    n == 0 && return LC.DeviceCholesky(similar(A, 0, 0), 0)
    U, status = CUSOLVER.potrf!('U', copy(A))
    info = Int(status)
    if info == 0
        d = view(vec(U), 1:(n + 1):(n * n))
        all(>(zero(T)), d) || (info = findfirst(x -> !(x > 0), d))
    end
    check && info != 0 && throw(PosDefException(info))
    return LC.DeviceCholesky(copy(transpose(triu!(U))), info)
end

LC._device_triangular_solve(L::CuMatrix{T}, B::CuMatrix{T}, transposed) where {T<:CuBlasReal} =
    transposed ? LowerTriangular(L)' \ B : LowerTriangular(L) \ B

end
