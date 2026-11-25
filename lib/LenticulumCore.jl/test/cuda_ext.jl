# LenticulumCoreCUDAExt (Q8): cuSOLVER's Cholesky and cuBLAS's triangular solves against the
# generic device kernel (on JLArrays) and LAPACK, on generated inputs. Needs CUDA; run by
# `gpu/runtests.jl`, not by this package's suite.
using CUDA, JLArrays, LinearAlgebra, Random, Test, LenticulumCore
const LC = LenticulumCore

# What `issuccess` and `info` must be: the generic kernel's decision on the same matrix, which
# follows reference LAPACK's (the first pivot that is not positive, NaN included; OpenBLAS, the CPU
# path, reports no NaN pivot, LP-093).
generic(A) = LC._symmetric_cholesky(JLArray(A); check = false)

function matrices(rng, T, n)
    R = randn(rng, T, n, n)
    spd = R * R' + T(n) * I
    out = Any[spd, R + R']                                   # positive definite, indefinite
    if n > 0
        junk = copy(spd); junk[n, 1] = T(1e6)                 # the lower triangle is not read
        push!(out, junk)
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        for bad in (T(NaN), T(Inf))
            M = copy(spd); M[i, j] = M[j, i] = bad
            push!(out, M)
        end
        Z = copy(spd); Z[n, n] = 0; Z[n, :] .= 0; Z[:, n] .= 0      # singular, last pivot zero
        push!(out, Z)
    end
    # A NaN pivot before a negative one (Gemini's review of Q8, LP-093: not reproduced, the NaN
    # reaches the later pivots, so cuSOLVER does not stop at the negative one).
    n == 2 && push!(out, T[NaN 0; 0 -1])
    n == 3 && push!(out, T[1 NaN 0; NaN 1 0; 0 0 -1])
    return out
end

@testset "CUDA Cholesky against the generic kernel and LAPACK" begin
    CUDA.allowscalar(false)
    rng = Xoshiro(2026)
    for T in (Float32, Float64), n in (0, 1, 2, 3, 5, 8, 17, 64), A in matrices(rng, T, n)
        rtol = T === Float32 ? 2e-4 : 1e-10
        C = LC._symmetric_cholesky(CuArray(A); check = false)
        G = generic(A)
        @test C isa LC.DeviceCholesky && C.L isa CuArray{T,2}
        @test C.info == G.info
        if issuccess(G)
            ref = cholesky(Symmetric(A))
            @test Array(C.L) ≈ ref.L rtol=rtol atol=rtol
            @test Array(LC._lower_matrix(C)) ≈ Array(G.L) rtol=rtol atol=rtol
            B = randn(rng, T, n, 3)
            @test Array(C \ CuArray(B)) ≈ ref \ B rtol=10rtol atol=10rtol
            @test Array(C \ CuArray(B[:, 1])) ≈ ref \ B[:, 1] rtol=10rtol atol=10rtol
            n > 0 && @test logdet(C) ≈ logdet(ref) rtol=rtol
            @test Array(inv(C)) ≈ inv(ref) rtol=10rtol atol=10rtol
            @test Array(LC._triangular_inverse(C)) ≈ inv(ref.L) rtol=10rtol atol=10rtol
            @test LC._symmetric_cholesky(CuArray(A)) isa LC.DeviceCholesky
        else
            @test_throws PosDefException(G.info) LC._symmetric_cholesky(CuArray(A))
            @test_throws PosDefException C \ CuArray(ones(T, n))
        end
    end
    # A right-hand side of another element type takes the generic solve and promotes.
    A = [4.0f0 1; 1 3]
    C = LC._symmetric_cholesky(CuArray(A))
    @test Array(C \ CuArray([1.0, 2.0])) ≈ A \ [1.0, 2.0] rtol=1e-6
    @test eltype(C \ CuArray([1.0, 2.0])) === Float64
end
