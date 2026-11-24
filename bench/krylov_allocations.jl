# julia --project=bench bench/krylov_allocations.jl
#
# Allocations and time of one GMRES solve on the same well-conditioned dense system.
# Compare the in-house GMRES, the optional KrylovJL backend, and Krylov.jl's gmres!
# with a preallocated workspace. See krylov_allocations.md for the measurements.
@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

using ImplicitSolvers, Krylov, LinearAlgebra, Random, Printf

const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
const SIZES = SMOKE ? (50,) : (200, 2000)
const RESTART, MAXITERS, TOL = 40, 400, 1e-10

function system(n; seed = 1)
    rng = Xoshiro(seed)
    A = Matrix{Float64}(I, n, n) .+ randn(rng, n, n) ./ (4sqrt(n))
    return A, randn(rng, n)
end

shared(op, b) = linsolve(op, b, GMRES(restart = RESTART, maxiters = MAXITERS);
                         tol = Tolerance(atol = TOL, rtol = TOL))
backend(A, b) = linsolve(A, b, KrylovJL(:gmres; memory = RESTART, maxiters = MAXITERS);
                        tol = Tolerance(atol = TOL, rtol = TOL))
krylov!(ws, A, b) = gmres!(ws, A, b; atol = TOL, rtol = TOL, restart = true, itmax = MAXITERS)

function measure(f)
    f()                                   # compile
    bytes = @allocated f()
    time = minimum(@elapsed(f()) for _ in 1:(SMOKE ? 1 : 20))
    return bytes, time
end

println("Julia ", VERSION, ", ", Threads.nthreads(), " Julia thread(s), ", BLAS.get_num_threads(),
        " BLAS thread(s); Krylov ", pkgversion(Krylov), "; restart $RESTART, atol = rtol = $TOL")
@printf("%6s  %-28s %6s %12s %10s\n", "n", "solver", "iters", "bytes", "time/ms")
for n in SIZES
    A, b = system(n)
    xref = A \ b
    for (name, op) in (("shared, matrix", A), ("shared, closure v -> A*v", v -> A * v))
        x, report = shared(op, b)
        @assert report.converged && norm(x - xref) <= 1e-8 * norm(xref)
        bytes, time = measure(() -> shared(op, b))
        @printf("%6d  %-28s %6d %12d %10.3f\n", n, name, report.iterations, bytes, 1e3time)
    end
    x, report = backend(A, b)
    @assert report.converged && norm(x - xref) <= 1e-8 * norm(xref)
    bytes, time = measure(() -> backend(A, b))
    @printf("%6d  %-28s %6d %12d %10.3f\n", n, "KrylovJL, matrix", report.iterations, bytes, 1e3time)
    ws = GmresWorkspace(A, b; memory = RESTART)
    krylov!(ws, A, b)
    @assert ws.stats.solved && norm(ws.x - xref) <= 1e-8 * norm(xref)
    bytes, time = measure(() -> krylov!(ws, A, b))
    @printf("%6d  %-28s %6d %12d %10.3f\n", n, "Krylov.jl gmres!, workspace", ws.stats.niter, bytes, 1e3time)
    @printf("%6s  one vector is %d bytes\n", "", sizeof(b))
end
