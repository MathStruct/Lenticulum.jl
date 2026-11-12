using Lenticulum, LinearAlgebra, Random, Test

# I18: the rank test of a GaussianFactor's backward message is kept per weight array
# (`_rank_deficient`, gaussian.jl). It must agree with the SVD test on every call, also after the
# array is changed in place, and when the memo is full.
@testset "I18 rank memo agrees with the SVD test" begin
    LT = Lenticulum
    rng = Xoshiro(18)
    As = Matrix{Float64}[]
    for trial in 1:2000
        if isempty(As) || rand(rng) < 0.3
            m, n = rand(rng, 1:4), rand(rng, 1:4)
            A = randn(rng, m, n)
            rand(rng) < 0.3 && n > 1 && (A[:, end] = A[:, 1] * rand(rng))   # rank-deficient
            push!(As, A)
        end
        A = rand(rng, As)
        if rand(rng) < 0.2                      # a change in place: zero a column, or restore it
            j = rand(rng, 1:size(A, 2))
            A[:, j] = rand(rng) < 0.5 ? zeros(size(A, 1)) : randn(rng, size(A, 1))
        end
        @test LT._rank_deficient(A) == LT._rank_deficient_svd(A)
    end
    # The case the copy guards: full rank, then deficient in place, then full again.
    A = [1.0 0.0; 0.0 1.0]
    @test !LT._rank_deficient(A)
    A[:, 2] .= 0.0
    @test LT._rank_deficient(A)
    A[2, 2] = 1.0
    @test !LT._rank_deficient(A)
    # The memo is bounded: filling it past its size empties it, and answers stay right.
    for _ in 1:(LT._RANK_MEMO_MAX + 10)
        B = randn(rng, 2, 2)
        @test LT._rank_deficient(B) == LT._rank_deficient_svd(B)
    end
    @test length(LT._RANK_MEMO) <= LT._RANK_MEMO_MAX
    # The memo does not keep an array alive, and a new array that gets a collected one's id is
    # tested afresh (LP-026 F3).
    GC.gc(); empty!(LT._RANK_MEMO)
    let
        ids = UInt[]
        for _ in 1:200
            B = zeros(2, 2); B[1, 1] = 1.0              # rank-deficient
            @test LT._rank_deficient(B); push!(ids, objectid(B))
        end
        GC.gc(true); GC.gc(true)
        @test count(id -> LT._RANK_MEMO[id][1].value !== nothing, ids) < 200
        # Id reuse, planted: an entry under B's id that belongs to another array, and one whose
        # array was collected, both with the wrong answer.
        B = [1.0 0.0; 0.0 1.0]; other = copy(B)
        LT._RANK_MEMO[objectid(B)] = (WeakRef(other), copy(B), true)
        @test !LT._rank_deficient(B)
        C = [1.0 0.0; 0.0 1.0]
        LT._RANK_MEMO[objectid(C)] = (WeakRef(nothing), copy(C), true)
        @test !LT._rank_deficient(C)
    end
    # Other element types bypass the memo and agree with the SVD test.
    for A in (Float32[1 2; 2 4], [1 0; 0 1], Symmetric([1.0 2.0; 2.0 4.0]))
        @test LT._rank_deficient(A) == LT._rank_deficient_svd(A)
    end
    # Concurrent callers on shared and private arrays.
    shared = [randn(rng, 3, 3) for _ in 1:8]; shared[2][:, 3] = shared[2][:, 1]
    ok = Threads.Atomic{Int}(0)
    Threads.@threads for k in 1:4000
        local C = isodd(k) ? shared[mod1(k, 8)] : randn(Xoshiro(k), 2, 3)   # not the outer `A`
        LT._rank_deficient(C) == LT._rank_deficient_svd(C) && Threads.atomic_add!(ok, 1)
    end
    @test ok[] == 4000
end
