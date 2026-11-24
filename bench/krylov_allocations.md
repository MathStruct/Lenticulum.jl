# GMRES allocations and time

The benchmark solves the same dense Float64 system through three entry points, with
`A = I + randn(n,n)/(4sqrt(n))`, restart length 40, and
`atol = rtol = 1e-10`. The program checks each solution against a direct solve.
The timing is the minimum of 20 warm calls. The allocation count is for one
warm call. The preallocated workspace is constructed outside the measurement.

Measured on Julia 1.13.1 and Krylov.jl 0.10.12, using eight Julia threads and four
BLAS threads. The source was the P41 working tree based on commit
`402a4d024db8897b944ff923e582339ecd2796bb`.

| Size | Solver | Iterations | Bytes | Time (ms) |
| ---: | --- | ---: | ---: | ---: |
| 200 | In-house GMRES, matrix | 17 | 112,008 | 0.074 |
| 200 | In-house GMRES, callable | 17 | 112,184 | 0.074 |
| 200 | KrylovJL, matrix | 17 | 89,256 | 0.066 |
| 200 | Krylov.jl gmres!, preallocated workspace | 17 | 0 | 0.057 |
| 2,000 | In-house GMRES, matrix | 17 | 947,672 | 2.120 |
| 2,000 | In-house GMRES, callable | 17 | 947,864 | 2.107 |
| 2,000 | KrylovJL, matrix | 17 | 780,856 | 2.023 |
| 2,000 | Krylov.jl gmres!, preallocated workspace | 17 | 0 | 1.817 |

`KrylovJL` includes workspace allocation and the true-residual checks in the
measurement. The preallocated `gmres!` row measures a solve with the workspace
already available and uses Krylov.jl's own convergence test. The older
comparison is in `vault/Design/Shared Solver Layer.md`, under “Why not Krylov.jl”.

The full-size run is `julia --project=bench bench/krylov_allocations.jl`. With
`LENTICULUM_SMOKE=true`, the script uses size 50 and one timed call. The P41
measurements cover warm CPU solves on this fixture.
