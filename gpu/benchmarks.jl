# Device benchmarks on a CUDA GPU (Q8): what the device paths cost, and from which size a GPU pays.
#
#     julia --project=gpu gpu/benchmarks.jl            # full sizes, about 10 min on an RTX 3080
#     LENTICULUM_SMOKE=true julia --project=gpu gpu/benchmarks.jl
#
# Four parts, each a Markdown table on stdout (and in ~/.cache/lenticulum/gpu/benchmarks-<date>.md):
#  1. dense SPD kernels: the GPUArraysCore extension's column-by-column Cholesky and triangular solve
#     against cuSOLVER/cuBLAS (the CUDA extension) and LAPACK on the CPU;
#  2. synchronisation and transfer: one `_entry` read (the 57 call sites of Q7), one `dot`, copies;
#  3. inference on a Kalman chain (dimension d, K steps), `infer!` on the CPU and on the device;
#  4. linear solves (ImplicitSolvers' `DenseSolve`, `GMRES`, `KrylovJL`) on the CPU and the device.
# Times are medians of BenchmarkTools samples, each ending in `CUDA.@sync`. Recorded in
# `gpu/device_benchmarks.md`.

using CUDA, BenchmarkTools, LinearAlgebra, Printf, Random, Dates, Statistics
using Adapt, GPUArraysCore
using LenticulumCore, Mycelium, Lenticulum, ImplicitSolvers, Krylov
const LC = LenticulumCore

CUDA.functional() || error("gpu/benchmarks.jl needs a working CUDA device")
CUDA.allowscalar(false)
const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
BenchmarkTools.DEFAULT_PARAMETERS.seconds = SMOKE ? 0.05 : 1.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = SMOKE ? 3 : 200

const OUT = IOBuffer()
emit(s) = (println(s); println(OUT, s))
µs(t) = @sprintf("%.1f", t / 1e3)            # BenchmarkTools times are in ns
med(b) = median(b).time
ratio(a, b) = @sprintf("%.2f", a / b)

emit("# Device benchmarks, $(Dates.format(now(), "yyyy-mm-dd HH:MM"))\n")
emit("GPU: $(CUDA.name(CUDA.device())); CUDA runtime $(CUDA.runtime_version()); " *
     "CPU: $(Sys.cpu_info()[1].model), $(Threads.nthreads()) Julia threads, BLAS threads $(BLAS.get_num_threads()); " *
     "Julia $(VERSION); smoke = $SMOKE\n")

# 1. Dense SPD kernels -------------------------------------------------------------------------
generic_cholesky(A::CuMatrix{T}) where {T} = invoke(LC._symmetric_cholesky, Tuple{AbstractGPUArray{T,2}}, A)
generic_solve(C, B::CuMatrix{T}) where {T} =
    invoke(LC._device_triangular_solve, Tuple{AbstractGPUArray,Any,Any}, C.L,
           invoke(LC._device_triangular_solve, Tuple{AbstractGPUArray,Any,Any}, C.L, B, false), true)
cusolver_solve(C, B) = LC._device_triangular_solve(C.L, LC._device_triangular_solve(C.L, B, false), true)

emit("## 1. Cholesky factorisation and solve (4 right-hand sides), µs\n")
emit("| T | n | generic factor | cuSOLVER factor | CPU factor | generic solve | cuBLAS solve | CPU solve | factor generic/cuSOLVER |")
emit("|---|---|---|---|---|---|---|---|---|")
for T in (Float32, Float64), n in (SMOKE ? (2, 32) : (2, 8, 32, 128, 512, 2048))
    rng = Xoshiro(n)
    A = randn(rng, T, n, n); A = A * A' + n * I
    B = randn(rng, T, n, 4)
    dA, dB = CuArray(A), CuArray(B)
    C = LC._symmetric_cholesky(dA)
    tg = med(@benchmark CUDA.@sync generic_cholesky($dA))
    tc = med(@benchmark CUDA.@sync LC._symmetric_cholesky($dA))
    th = med(@benchmark cholesky(Symmetric($A)))
    sg = med(@benchmark CUDA.@sync generic_solve($C, $dB))
    sc = med(@benchmark CUDA.@sync cusolver_solve($C, $dB))
    Ch = cholesky(Symmetric(A))
    sh = med(@benchmark $Ch \ $B)
    emit("| $T | $n | $(µs(tg)) | $(µs(tc)) | $(µs(th)) | $(µs(sg)) | $(µs(sc)) | $(µs(sh)) | $(ratio(tg, tc)) |")
end

# 2. Synchronisation and transfer --------------------------------------------------------------
emit("\n## 2. Synchronisation and transfer, µs\n")
emit("| operation | n | time |")
emit("|---|---|---|")
for n in (SMOKE ? (16,) : (16, 4096, 1 << 20))
    x = CUDA.rand(Float64, n); h = rand(n); y = CUDA.rand(Float64, n)
    emit("| `_entry` (one entry to the host) | $n | $(µs(med(@benchmark LC._entry($x, 1)))) |")
    emit("| `dot` (reduction to the host) | $n | $(µs(med(@benchmark dot($x, $y)))) |")
    emit("| broadcast `x .+ y` (no synchronisation until `@sync`) | $n | $(µs(med(@benchmark CUDA.@sync $x .+ $y))) |")
    emit("| host → device copy | $n | $(µs(med(@benchmark CUDA.@sync copyto!($x, $h)))) |")
    emit("| device → host copy | $n | $(µs(med(@benchmark copyto!($h, $x)))) |")
end

# 3. Inference on a Kalman chain ---------------------------------------------------------------
function kalman_chain(d, K, ::Type{T} = Float64; rng = Xoshiro(d + K)) where {T}
    b = GraphBuilder()
    Id = Matrix{T}(I, d, d)
    A = T.(Matrix(qr(randn(rng, d, d)).Q) * 0.95)
    variable!(b, :x1, d)
    factor!(b, :prior, GaussianPrior(:x1, zeros(T, d), Id))
    connect!(b, :prior, :x1, :x1)
    ps = Pair{Symbol,Any}[]
    for k in 1:K
        xk, yk, obs = Symbol(:x, k), Symbol(:y, k), Symbol(:obs, k)
        variable!(b, yk, d)
        factor!(b, obs, GaussianFactor(d => d; noise = T(0.2) * Id))
        connect!(b, obs, :x, xk); connect!(b, obs, :y, yk)
        factor!(b, Symbol(:data, k), DataFactor(yk, T.(randn(rng, d))))
        connect!(b, Symbol(:data, k), yk, yk; direction = Emitting())
        push!(ps, obs => (A = Id, b = zeros(T, d)))
        k == K && break
        xn, trans = Symbol(:x, k + 1), Symbol(:trans, k)
        variable!(b, xn, d)
        factor!(b, trans, GaussianFactor(d => d; noise = T(0.1) * Id))
        connect!(b, trans, :x, xk); connect!(b, trans, :y, xn)
        push!(ps, trans => (A = A, b = zeros(T, d)))
    end
    return build(b), NamedTuple(ps)
end

emit("\n## 3. Inference on a Kalman chain (`infer!` with `tree_schedule`), ms\n")
emit("| d | K | CPU Float64 | CUDA Float64 | CUDA Float32 | CUDA64/CPU | max rel. difference of the means (Float64) |")
emit("|---|---|---|---|---|---|---|")
for (d, K) in (SMOKE ? ((2, 4),) : ((2, 10), (16, 10), (64, 10), (256, 10)))
    g, ps = kalman_chain(d, K)
    sched = tree_schedule(g)
    cpu = first(infer!(g, sched, ps, (;)))
    dg, dp = adapt(CuArray, (g, ps))
    gpu = first(infer!(dg, tree_schedule(dg), dp, (;)))
    diff = maximum(n -> (a = belief_mean(cpu[n]); norm(Array(belief_mean(gpu[n])) - a) / max(norm(a), eps())),
                   [n for n in keys(cpu) if cpu[n] isa GaussianBelief])
    g32, p32 = adapt(CuArray, kalman_chain(d, K, Float32))
    tc = med(@benchmark infer!($g, $sched, $ps, (;)))
    td = med(@benchmark CUDA.@sync infer!($dg, tree_schedule($dg), $dp, (;)))
    t32 = try
        first(infer!(g32, tree_schedule(g32), p32, (;)))
        @sprintf("%.2f", med(@benchmark CUDA.@sync infer!($g32, tree_schedule($g32), $p32, (;))) / 1e6)
    catch err
        "fails: $(first(sprint(showerror, err), 80))"
    end
    emit("| $d | $K | $(@sprintf("%.2f", tc / 1e6)) | $(@sprintf("%.2f", td / 1e6)) | $t32 | $(ratio(td, tc)) | $(@sprintf("%.1e", diff)) |")
end

# 4. Linear solves ------------------------------------------------------------------------------
emit("\n## 4. Linear solves `linsolve(A, b, alg)`, ms (A = I + R/(3√n), nonsymmetric, well conditioned)\n")
emit("| T | n | algorithm | CPU | CUDA | CUDA/CPU | iterations (CUDA) |")
emit("|---|---|---|---|---|---|---|")
for T in (Float32, Float64), n in (SMOKE ? (64,) : (256, 1024, 4096))
    rng = Xoshiro(n)
    A = Matrix{T}(I, n, n) + randn(rng, T, n, n) / T(3sqrt(n)); b = randn(rng, T, n)
    dA, db = CuArray(A), CuArray(b)
    for (name, alg) in (("DenseSolve", DenseSolve()), ("GMRES", GMRES()), ("KrylovJL(:gmres)", KrylovJL(:gmres)))
        name == "DenseSolve" && T === Float32 && n > 1024 && continue
        x, rep = linsolve(dA, db, alg)
        tc = med(@benchmark linsolve($A, $b, $alg))
        td = med(@benchmark CUDA.@sync linsolve($dA, $db, $alg))
        it = hasproperty(rep, :iterations) ? rep.iterations : "—"
        emit("| $T | $n | $name | $(@sprintf("%.3f", tc / 1e6)) | $(@sprintf("%.3f", td / 1e6)) | $(ratio(td, tc)) | $it |")
    end
end

dir = joinpath(homedir(), ".cache", "lenticulum", "gpu"); mkpath(dir)
file = joinpath(dir, "benchmarks-$(Dates.format(now(), "yyyy-mm-dd-HHMM"))$(SMOKE ? "-smoke" : "").md")
write(file, String(take!(OUT)))
println("\nwritten to $file")
