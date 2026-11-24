# julia --project=bench bench/krylov_dual.jl
#
# Compare KrylovJL's separate tangent solves with the in-house dual arithmetic.
using ForwardDiff, ImplicitSolvers, Krylov, LinearAlgebra, Printf, Random

const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
const SIZES = SMOKE ? (20,) : (100, 1000)
const PARTIALS = SMOKE ? (1, 2) : (1, 4, 8)
const REPEATS = SMOKE ? 1 : 20
const TOL = Tolerance(atol=1e-10, rtol=1e-10)

function dual_rhs(rhs, ::Val{N}) where {N}
    return [ForwardDiff.Dual{:krylov_dual}(rhs[j, 1], ntuple(i -> rhs[j, i + 1], Val(N)))
            for j in axes(rhs, 1)]
end

function measure(A, rhs, N, algorithms)
    b = dual_rhs(rhs, Val(N))
    reference = A \ rhs
    for (name, alg) in algorithms
        solve() = linsolve(A, b, alg; tol=TOL)
        x, report = solve()
        @assert report.converged
        @assert norm(ForwardDiff.value.(x) - reference[:, 1]) <= 1e-8 * norm(reference[:, 1])
        for i in 1:N
            @assert norm(ForwardDiff.partials.(x, i) - reference[:, i + 1]) <=
                    1e-8 * norm(reference[:, i + 1])
        end
        bytes = @allocated solve()
        time = minimum(@elapsed(solve()) for _ in 1:REPEATS)
        @printf("%6d  %3d  %-17s %6d %12d %10.3f\n", size(A, 1), N, name, report.nfe, bytes, 1e3time)
        flush(stdout)
    end
end

function main()
    println("Julia ", VERSION, ", ", Threads.nthreads(), " Julia thread(s), ", BLAS.get_num_threads(),
            " BLAS thread(s); Krylov ", pkgversion(Krylov), "; atol = rtol = ", TOL.atol)
    @printf("%6s  %3s  %-17s %6s %12s %10s\n", "n", "N", "solver", "nfe", "bytes", "time/ms")
    for n in SIZES
        rng = Xoshiro(4242)
        W = randn(rng, n, n) / (4sqrt(n))
        general = Matrix(I + W)
        spd = Matrix(I + W' * W)
        rhs = randn(rng, n, maximum(PARTIALS) + 1)
        for N in PARTIALS
            directions = rhs[:, 1:(N + 1)]
            measure(general, directions, N, (("GMRES", GMRES()), ("KrylovJL gmres", KrylovJL(:gmres))))
            measure(spd, directions, N, (("MINRES", MINRES()), ("KrylovJL minres", KrylovJL(:minres)),
                                        ("KrylovJL cg", KrylovJL(:cg))))
        end
    end
end

main()
