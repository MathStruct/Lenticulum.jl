# Storage hooks. The GPUArraysCore extension supplies the device methods;
# the CPU fallbacks retain the original materialisation and factorisations.
_device_leaf(x) = nothing
_device_leaf(x::Union{Tuple,NamedTuple,AbstractArray}) = _first_device(x)
_device_leaf(::AbstractArray{<:Number}) = nothing
_device_leaf(x::Diagonal) = _device_leaf(x.diag)
_device_leaf(x::Union{Symmetric,LinearAlgebra.Adjoint,LinearAlgebra.Transpose,SubArray}) =
    _device_leaf(parent(x))

# Tuples are walked by recursion, not a loop: the loop over a heterogeneous tuple is not inferred
# and dispatched per element on every call, which cost the CPU path up to 1.6× (LP-092). Unrolled,
# an all-host call folds to `nothing` at compile time.
_first_device(xs::NamedTuple) = _first_device(values(xs))
_first_device(::Tuple{}) = nothing
function _first_device(xs::Tuple)
    leaf = _device_leaf(first(xs))
    leaf === nothing ? _first_device(Base.tail(xs)) : leaf
end
function _first_device(xs)
    for x in xs
        leaf = _device_leaf(x)
        leaf === nothing || return leaf
    end
    return nothing
end

_on_device(::Nothing, x) = x

_matrix(x) = Matrix(x)
_matrix(::Type{T}, x) where {T} = Matrix{T}(x)
_vector(::Type{T}, x) where {T} = Vector{T}(x)
_eye(x, ::Type{T}, n) where {T} = _identity(_device_leaf(x), T, n)
_identity(::Nothing, ::Type{T}, n) where {T} = Matrix{T}(I, n, n)

# Only spectral decisions use this explicit host round trip. The callback
# may select subspaces on the host before returning their numeric payloads.
_host_spectral(f, A) = _spectral(_device_leaf(A), f, A)
_spectral(::Nothing, f, A) = f(A)

_symmetric_cholesky(A; check = true) = cholesky(Symmetric(A); check)
_symmetric_solve(A, b) = A \ b
_symmetric_logdet(A) = logdet(A)
_device_diag(A) = LinearAlgebra.diag(A)
_symmetric_inv(A) = inv(Symmetric(A))
_triangular_inverse(C) = inv(C.L)
_entry(A, i::Int) = A[i]
function _only_entry(A)
    _device_leaf(A) === nothing && return only(A)
    length(A) == 1 || throw(ArgumentError("expected one element"))
    return _entry(A, 1)
end
_dense_inv(A) = inv(A)
_eigen(A) = _eigen(_device_leaf(A), A)
_eigen(::Nothing, A) = eigen(A)
_eigvals(A) = _host_spectral(eigvals, A)

# The Cholesky factor of a device matrix: the dense lower factor `L` and LAPACK's `info` (0, or
# the first nonpositive pivot). The GPUArraysCore extension computes it column by column; the
# CUDA extension with cuSOLVER (Q8). Its solves go through `_device_triangular_solve`, which
# solves `L X = B`, or `L' X = B` when `transposed`, with the same two implementations.
struct DeviceCholesky{M}
    L::M
    info::Int
end
LinearAlgebra.issuccess(C::DeviceCholesky) = iszero(C.info)
Base.eltype(C::DeviceCholesky) = eltype(C.L)
Base.size(C::DeviceCholesky, d::Int) = size(C.L, d)
function _device_triangular_solve end

_cholesky_pivots(C) = LinearAlgebra.diag(C.U)
_lower_matrix(C) = Matrix(C.L)
