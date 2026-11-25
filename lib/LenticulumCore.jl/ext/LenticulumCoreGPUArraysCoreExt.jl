module LenticulumCoreGPUArraysCoreExt
using LenticulumCore: LenticulumCore as LC
using GPUArraysCore: AbstractGPUArray, AnyGPUArray
using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, PosDefException

# Diagonal Gaussians use elementwise division and logarithms.
function LC._precision_factor(M::Diagonal{T,<:AbstractGPUArray}) where {T}
    all(x -> isfinite(x) && x > 0, M.diag) || throw(ArgumentError(
        "Gaussian device precision must have finite positive diagonal entries"))
    return M
end
LC._precision_solve(M::Diagonal{T,<:AbstractGPUArray}, η) where {T} = η ./ M.diag
LC._precision_logdet(M::Diagonal{T,<:AbstractGPUArray}) where {T} = sum(log, M.diag)
LC._precision_isproper(M::Diagonal{T,<:AbstractGPUArray}) where {T} =
    all(x -> isfinite(x) && x > 0, M.diag)
LC.isuninformative(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T} =
    all(iszero,b.Λ.diag) && all(iszero,b.η)
# The device moment methods go through the same `_improper_check` as the CPU ones: a belief
# declared improper is a likelihood, and reading its moments or normaliser is an error there too,
# whatever the numerical precision looks like (GH-19).
function LC.belief_cov(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    LC._improper_check(b, "belief_cov")
    return Diagonal(inv.(b.Λ.diag))
end
function LC.Gaussian(μ::AbstractGPUArray{T,1}, Σ::Diagonal{S,<:AbstractGPUArray}) where {T,S}
    LC._precision_factor(Σ)
    λ = inv.(Σ.diag)
    return LC.GaussianBelief(λ .* μ, Diagonal(λ))
end
function LC.belief_mean(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    LC._improper_check(b, "belief_mean")
    return b.η ./ b.Λ.diag
end
function LC.logpartition(b::LC.GaussianBelief{<:AbstractGPUArray,<:Diagonal{T,<:AbstractGPUArray}}) where {T}
    LC._improper_check(b, "logpartition")
    return sum(b.η .* (b.η ./ b.Λ.diag))/2 - sum(log, b.Λ.diag)/2 +
           T(LC.dimension(b))*log(T(2)*T(π))/2
end
LC._chol(M::Diagonal{T,<:AbstractGPUArray}, name) where {T} = LC._precision_factor(M)
LC._device_leaf(x::AbstractGPUArray{<:Number}) = x
const DeviceSubMatrix = SubArray{T,2,P} where {T,P<:AbstractGPUArray}
LC._matrix(x::DeviceSubMatrix) = copy(x)
LC._matrix(::Type{T}, x::DeviceSubMatrix) where {T} = T.(x)
LC._chol(x::DeviceSubMatrix, name) = LC._chol(copy(x), name)
LC._precision_factor(x::DeviceSubMatrix) = LC._precision_factor(copy(x))
LC._precision_isproper(x::DeviceSubMatrix) = LC._precision_isproper(copy(x))
LC._matrix(x::AbstractGPUArray{T,2}) where {T} = x
LC._matrix(::Type{T}, x::AbstractGPUArray{S,2}) where {T,S} = LC._at_eltype(T, x)
LC._vector(::Type{T}, x::AbstractGPUArray{S,1}) where {T,S} = LC._at_eltype(T, x)
function LC._matrix(x::Symmetric{T,<:AbstractGPUArray}) where {T}
    A = parent(x)
    n = size(A, 1)
    row, col = 1:n, reshape(1:n, 1, n)
    triangle = x.uplo == 'U' ? (<=) : (>=)
    return ifelse.(triangle.(row, col), A, transpose(A))
end
LC._matrix(x::Diagonal{T,<:AbstractGPUArray}) where {T} = LC._eye(x.diag, T, length(x.diag)) .* x.diag
LC._matrix(x::LinearAlgebra.Adjoint{T,<:AbstractGPUArray}) where {T} = copy(x)
LC._matrix(x::LinearAlgebra.Transpose{T,<:AbstractGPUArray}) where {T} = copy(x)

# LP-092: a device matrix inside a LinearAlgebra wrapper is materialised on the device
# before it reaches the dense kernels; the CPU fallbacks (`Matrix(M)`, `cholesky`) would
# copy it to the host without a word, or index it elementwise.
const DeviceWrapped{T} = Union{Symmetric{T,<:AbstractGPUArray},
                               LinearAlgebra.Adjoint{T,<:AbstractGPUArray},
                               LinearAlgebra.Transpose{T,<:AbstractGPUArray}}
LC._symmetric_part(A::DeviceWrapped) = LC._symmetric_part(LC._matrix(A))
LC._nearly_symmetric(A::DeviceWrapped) = LC._nearly_symmetric(LC._matrix(A))
LC._chol(A::DeviceWrapped, name) = LC._chol(LC._matrix(A), name)
LC._precision_factor(A::DeviceWrapped) = LC._precision_factor(LC._matrix(A))
LC._precision_isproper(A::DeviceWrapped) = LC._precision_isproper(LC._matrix(A))
LC._symmetric_cholesky(A::Union{LinearAlgebra.Adjoint{T,<:AbstractGPUArray},
                                LinearAlgebra.Transpose{T,<:AbstractGPUArray},
                                Diagonal{T,<:AbstractGPUArray}}; check = true) where {T} =
    LC._symmetric_cholesky(LC._matrix(A); check)

function LC._on_device(prototype::AbstractGPUArray, x::AbstractArray)
    leaf = LC._device_leaf(x)
    if leaf !== nothing && Base.typename(typeof(leaf)) === Base.typename(typeof(prototype))
        return x
    end
    return copyto!(similar(prototype, eltype(x), size(x)), leaf === nothing ? Array(x) : x)
end
LC._on_device(prototype::AbstractGPUArray, x::Diagonal) =
    Diagonal(LC._on_device(prototype, x.diag))
LC._on_device(prototype::AbstractGPUArray, x::Union{Tuple,NamedTuple}) = map(v -> LC._on_device(prototype, v), x)
LC._on_device(prototype::AbstractGPUArray, x) = x

function LC._identity(prototype::AbstractGPUArray, ::Type{T}, n) where {T}
    A = similar(prototype, T, n, n)
    A .= ifelse.((1:n) .== reshape(1:n, 1, n), one(T), zero(T))
    return A
end

function LC._spectral(prototype::AbstractGPUArray, f, A)
    return LC._on_device(prototype, f(_host_matrix(A)))
end

_host_matrix(A) = Array(A)
_host_matrix(A::Symmetric) = Symmetric(Array(parent(A)), A.uplo == 'U' ? :U : :L)
# LP-092: `Array` of a wrapped device array copies elementwise (scalar indexing); copy
# the stored array in one transfer and rewrap it on the host.
_host_matrix(A::Diagonal) = Diagonal(Array(A.diag))
_host_matrix(A::Union{LinearAlgebra.Adjoint,LinearAlgebra.Transpose}) = Array(copy(A))
function LC._eigen(prototype::AbstractGPUArray, A)
    values, vectors = LC._spectral(prototype, A -> (E = LinearAlgebra.eigen(A); (E.values, E.vectors)), A)
    return (; values, vectors)
end

const DeviceCholesky = LC.DeviceCholesky
LC._cholesky_pivots(C::DeviceCholesky) = LC._device_diag(C.L)
LC._lower_matrix(C::DeviceCholesky) = C.L
LC._symmetric_cholesky(A::Symmetric{T,<:AbstractGPUArray}; check = true) where {T} =
    LC._symmetric_cholesky(LC._matrix(A); check)
LC._symmetric_solve(A::Symmetric{T,<:AbstractGPUArray}, B) where {T} =
    LC._symmetric_cholesky(A) \ B

function Base.show(io::IO, b::LC.GaussianBelief{<:AbstractGPUArray,<:AbstractGPUArray})
    print(io, "GaussianBelief(")
    show(io, b.η)
    print(io, ", ")
    show(io, b.Λ)
    print(io, ", ", b.improper, ")")
end

# Right-looking Cholesky, one kernel per column (the CUDA extension uses cuSOLVER's potrf for
# Float32 and Float64; `gpu/device_benchmarks.md`). The pivot reduction synchronises once per column
# to match LAPACK's first nonpositive pivot, including check=false callers.
function LC._symmetric_cholesky(A::AbstractGPUArray{T,2}; check = true) where {T}
    n = LinearAlgebra.checksquare(A)
    # Read the upper triangle, as the CPU path's `cholesky(Symmetric(A))` does (LP-092).
    L = float.(LC._matrix(Symmetric(A)))
    for j in 1:n
        pivot = sum(view(L, j:j, j:j))
        if !(pivot > 0)
            check && throw(PosDefException(j))
            return DeviceCholesky(L, j)
        end
        # The pivot is set to √pivot, not divided by it: Inf/√Inf is NaN where LAPACK gives Inf (Q8).
        view(L, j:j, j:j) .= sqrt(pivot)
        tail = view(L, (j + 1):n, j:j)
        tail ./= sqrt(pivot)
        view(L, (j + 1):n, (j + 1):n) .-= tail .* transpose(tail)
    end
    L .= ifelse.((1:n) .>= reshape(1:n, 1, n), L, zero(eltype(L)))
    return DeviceCholesky(L, 0)
end

# A row slice is a matrix even for one right hand side. Broadcasting the
# pivot keeps its value on the device throughout forward/back substitution.
# One kernel per column; the CUDA extension replaces it by cuBLAS's trsm.
function LC._device_triangular_solve(L::AbstractGPUArray, B, transposed)
    n = size(L, 1)
    X = copy(B)
    for j in (transposed ? (n:-1:1) : (1:n))
        row = view(X, j:j, :)
        row ./= view(L, j:j, j:j)
        tail = transposed ? (1:(j - 1)) : ((j + 1):n)
        col = transposed ? transpose(view(L, j:j, tail)) : view(L, tail, j:j)
        view(X, tail, :) .-= col .* row
    end
    return X
end

function Base.:\(C::DeviceCholesky, B::AbstractVecOrMat)
    LinearAlgebra.issuccess(C) || throw(PosDefException(C.info))
    size(B, 1) == size(C, 1) || throw(DimensionMismatch("incompatible Cholesky and right-hand-side dimensions"))
    T = promote_type(eltype(C), eltype(B))
    X = T.(LC._on_device(C.L, B))
    solved = LC._device_triangular_solve(C.L, LC._device_triangular_solve(C.L, reshape(X, size(X, 1), B isa AbstractVector ? 1 : size(B, 2)), false), true)
    return B isa AbstractVector ? vec(solved) : solved
end

function Base.inv(C::DeviceCholesky)
    X = C \ LC._eye(C.L, eltype(C), size(C.L, 1))
    return (X + X') / 2
end

LinearAlgebra.logdet(C::DeviceCholesky) = 2 * sum(log, LC._device_diag(C.L))
LC._device_diag(A::AbstractGPUArray{T,2}) where {T} =
    vec(A)[1:(size(A, 1) + 1):(1 + (min(size(A)...) - 1) * (size(A, 1) + 1))]
LC._nearly_symmetric(A::AbstractGPUArray{T,2}) where {T} =
    maximum(abs, A .- A'; init = zero(T)) <= sqrt(eps(Float32)) * maximum(abs, A; init = zero(T))
LC._check_symmetric(A::AbstractGPUArray{T,2}, name) where {T} =
    LC._nearly_symmetric(A) || throw(ArgumentError("$name is not symmetric"))
LC._symmetric_part(A::AbstractGPUArray{T,2}) where {T} = A .+ (A' .- A) ./ 2
function LC._precision_factor(A::AbstractGPUArray{T,2}) where {T}
    LC._check_symmetric(A, "Λ")
    return LC._symmetric_cholesky(LC._symmetric_part(A))
end
LC._precision_isproper(A::AbstractGPUArray{T,2}) where {T} =
    LC._nearly_symmetric(A) && LinearAlgebra.issuccess(LC._symmetric_cholesky(LC._symmetric_part(A); check = false))
function LC._chol(A::AbstractGPUArray{T,2}, name) where {T}
    LC._check_symmetric(A, name)
    C = LC._symmetric_cholesky(LC._symmetric_part(A); check = false)
    LinearAlgebra.issuccess(C) || throw(ArgumentError("$name is not positive definite"))
    return C
end
# Gather one entry via a one-element reduction for a host-side branch or
# mixture component. Each caller uses the scalar as control or metadata.
LC._entry(A::AbstractGPUArray, i::Int) = sum(view(vec(A), i:i))
LC._triangular_inverse(C::DeviceCholesky) =
    LC._device_triangular_solve(C.L, LC._eye(C.L, eltype(C), size(C, 1)), false)
LC._symmetric_inv(A::Diagonal{T,<:AbstractGPUArray}) where {T} = LC._matrix(Diagonal(inv.(A.diag)))
LC._symmetric_inv(A::AbstractGPUArray{T,2}) where {T} = inv(LC._symmetric_cholesky(A))
LC._dense_inv(A::AbstractGPUArray{T,2}) where {T} = LC._symmetric_inv(A)
LC._symmetric_solve(A::AbstractGPUArray{T,2}, B) where {T} =
    LC._symmetric_cholesky(A) \ B
LC._symmetric_logdet(A::AbstractGPUArray{T,2}) where {T} =
    LinearAlgebra.logdet(LC._symmetric_cholesky(A))

LC._rebuild(θ::AbstractGPUArray, o::Int, x::Number) = throw(ArgumentError(
    "Device parameter vectors require array leaves; use a one-element device array for scalar parameters"))

# Empty tuple branches otherwise contribute host Float64[] vectors to vcat,
# selecting host concatenation and scalar-indexing the active device leaves.
function LC._flatten_active_tree(leaf::AnyGPUArray, x)
    parts = filter(!isempty, map(LC._flatten_active, values(x)))
    return isempty(parts) ? vec(copy(leaf)) : reduce(vcat, parts)
end

LC._jacobian(::Nothing, fun, u::AbstractGPUArray, args...; ε=1e-6) =
    throw(ArgumentError("Device finite-difference Jacobians are unsupported; pass ad=AutoForwardDiff() or AutoZygote()"))
LC._gradient(::Nothing, fun, u::AbstractGPUArray, args...; ε=1e-6) =
    throw(ArgumentError("Device finite-difference gradients are unsupported; pass ad=AutoForwardDiff() or AutoZygote()"))

end
