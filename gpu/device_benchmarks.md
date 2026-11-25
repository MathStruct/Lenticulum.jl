# Device benchmarks (Q8)

> What the device paths cost on a CUDA GPU, from `gpu/benchmarks.jl`, run once on 2026-11-25: NVIDIA
> GeForce RTX 3080 (CUDA runtime 13.4), AMD Ryzen 7 7800X3D, Julia 1.13.1, one Julia thread, 4 BLAS
> threads. Medians of BenchmarkTools samples, each ending in `CUDA.@sync`. One machine, one run: the
> ratios are what to read, not the last digit. The full output is in
> `~/.cache/lenticulum/gpu/benchmarks-2026-11-25-2020.md` on that machine.

> Sources: code: `gpu/benchmarks.jl`, `lib/LenticulumCore.jl/ext/LenticulumCoreCUDAExt.jl`,
> `lib/LenticulumCore.jl/ext/LenticulumCoreGPUArraysCoreExt.jl`, `lib/ImplicitSolvers.jl/src/linsolve.jl`

## What it shows

1. **The factors gain nothing from a GPU at the sizes they have.** Inference on a Kalman chain of 10 steps
   is 160× slower on the device at dimension 2, 46× at 16, 9.5× at 64 and 2.9× at 256 (Float64; Float32
   1.08× at 256). Each message is a handful of small kernels and a few synchronisations, about 20 µs
   each, against a CPU message of a few µs. A GPU pays only when many factors' messages are batched into
   one kernel, which no code does yet (plates batch on the CPU), or when single factors are hundreds wide.
   The device path is for keeping data where a neural factor or a diffusion model already has it, not
   for speed. The device means agree with the CPU's to 6e-16.
2. **cuSOLVER is the right factorisation on CUDA.** The generic kernel (one kernel and one pivot read
   per column) is 3–36× slower than cuSOLVER's `potrf` from n = 8 on, and its triangular solves 2–29×
   slower than cuBLAS's `trsm`. At n = 2 the generic kernel is faster (38 against 48 µs), because the
   CUDA extension spends one reduction (about 13 µs) on checking the pivots for NaN, which cuSOLVER does
   not report and the generic kernel does. Only from n ≈ 512 (Float32) or 2048 (Float64; the RTX 3080 has little
   FP64 throughput) does the GPU beat LAPACK on the CPU.
3. **A synchronisation costs 5–15 µs.** One `_entry` read (a one-element reduction copied to the host) is
   5 µs, a `dot` 15 µs, a broadcast kernel 7 µs. So Q7's 57 `_entry` call sites cost 5 µs per read: small
   next to the 20–50 µs of a factorisation, and none of the measured paths is dominated by them. Rewriting
   them as reductions (Q20) is not worth it for speed.
4. **GMRES on the device is bound by its synchronisations.** ImplicitSolvers' GMRES reads two `dot`s per
   basis vector per step to the host (`H` and `g` live on the host): 0.28 ms per iteration at n = 256
   in Float32, 0.49 ms in Float64; Krylov.jl's GMRES 0.16 and 0.26 ms. The device is slower than the CPU
   up to n = 1024 (9–83×) and
   faster from n = 4096 (Float64: 0.77× in-house, 0.54× Krylov.jl). `DenseSolve` factorises on the host
   by design (`_dense_matrix`, `_host_array` in `linsolve.jl`), so its device time is the CPU's plus the
   copies.

## Tables

Cholesky factorisation and solve with 4 right-hand sides, µs:

| T | n | generic factor | cuSOLVER factor | CPU factor | generic solve | cuBLAS solve | CPU solve |
|---|---|---|---|---|---|---|---|
| Float32 | 2 | 37.7 | 48.0 | 0.1 | 21.0 | 20.9 | 0.2 |
| Float32 | 8 | 159.9 | 48.8 | 0.2 | 67.4 | 21.1 | 0.3 |
| Float32 | 32 | 656.9 | 57.0 | 1.6 | 254.4 | 22.5 | 1.4 |
| Float32 | 128 | 2674.4 | 120.5 | 38.3 | 1009.4 | 43.5 | 9.5 |
| Float32 | 512 | 11287.6 | 401.0 | 620.5 | 4217.8 | 144.8 | 80.8 |
| Float32 | 2048 | 80813.7 | 2268.7 | 10750.9 | 17975.9 | 635.3 | 1856.7 |
| Float64 | 2 | 37.6 | 49.7 | 0.1 | 22.8 | 24.2 | 0.2 |
| Float64 | 8 | 159.9 | 53.6 | 0.2 | 74.7 | 31.2 | 0.4 |
| Float64 | 32 | 651.2 | 74.0 | 1.6 | 321.9 | 66.9 | 1.6 |
| Float64 | 128 | 2659.9 | 273.1 | 56.8 | 1050.2 | 258.0 | 10.5 |
| Float64 | 512 | 11603.4 | 1144.0 | 564.0 | 4453.2 | 1212.4 | 136.0 |
| Float64 | 2048 | 109443.5 | 16513.0 | 18211.1 | 19129.8 | 1966.5 | 1859.4 |

Synchronisation and transfer, µs (n = 16 / 4096 / 1,048,576 Float64):

| operation | 16 | 4096 | 1,048,576 |
|---|---|---|---|
| `_entry` (one entry to the host) | 5.1 | 5.2 | 5.1 |
| `dot` (reduction to the host) | 15.5 | 15.5 | 42.2 |
| broadcast `x .+ y` | 7.3 | 7.3 | 44.8 |
| host → device copy | 4.4 | 8.2 | 361.6 |
| device → host copy | 5.2 | 7.7 | 402.4 |

Inference on a Kalman chain, `infer!` with `tree_schedule`, 10 steps, ms:

| d | CPU Float64 | CUDA Float64 | CUDA Float32 | CUDA64 / CPU |
|---|---|---|---|---|
| 2 | 0.13 | 20.80 | 18.11 | 160 |
| 16 | 0.46 | 21.49 | 18.71 | 46 |
| 64 | 4.84 | 46.11 | 22.86 | 9.5 |
| 256 | 72.69 | 209.12 | 78.21 | 2.9 |

Linear solves `linsolve(A, b, alg)`, $A = I + R/(3\sqrt n)$, ms (12 iterations in Float32, 19 in Float64):

| T | n | algorithm | CPU | CUDA | CUDA / CPU |
|---|---|---|---|---|---|
| Float32 | 256 | GMRES | 0.047 | 3.332 | 70.9 |
| Float32 | 256 | KrylovJL(:gmres) | 0.032 | 1.875 | 59.0 |
| Float32 | 1024 | GMRES | 0.240 | 3.466 | 14.5 |
| Float32 | 1024 | KrylovJL(:gmres) | 0.219 | 1.981 | 9.0 |
| Float32 | 4096 | GMRES | 2.613 | 4.665 | 1.79 |
| Float32 | 4096 | KrylovJL(:gmres) | 2.558 | 3.213 | 1.26 |
| Float64 | 256 | GMRES | 0.113 | 9.346 | 82.7 |
| Float64 | 1024 | GMRES | 0.701 | 10.363 | 14.8 |
| Float64 | 4096 | GMRES | 18.667 | 14.315 | 0.77 |
| Float64 | 4096 | KrylovJL(:gmres) | 17.556 | 9.407 | 0.54 |
| Float64 | 4096 | DenseSolve | 205.1 | 215.4 | 1.05 |

## Not measured

Kernel counts per message (by profiler), memory use, AD through device paths (forward and reverse
cost), VariationalDiffusion's device paths (its device tests are inline in its `runtests.jl` and run
on JLArrays only; `examples/gpu_energy.jl` measures one energy network), batched factors, a second
GPU or a second run (the noise of these numbers is unknown).
