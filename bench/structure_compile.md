# First gradient on a new structure

`structure_compile.jl` uses the position-velocity Kalman chain, `ExactCoupling()` on every edge
and the tree schedule. A five-step chain warms the shared code. Each measured size has a new
parameter and state tree type. The first call includes compilation. The warm number is the best of
three subsequent calls after garbage collection.

Run one mode per process. The modes are `zygote` and `fd` for the unrolled free-energy gradient,
`implicit` for the dense implicit adjoint with Zygote, and `infer` for Zygote through `infer!`.
`LENTICULUM_SMOKE=true` runs all four modes on two- and three-step chains after a one-step warmup.

## Zygote through `infer!` (2026-11-22)

`infer` differentiates the squared norm of the final state marginal mean with respect to the
parameter tree. The public `infer!` rule replays the realised schedule through `_functional_infer`,
which retains per-factor pinning.

```sh
MODE=infer TS="10 20 40" julia --project=bench bench/structure_compile.jl
```

| factors | first (s) | best warm (s) |
|---|---:|---:|
| 31 | 2.04 | 0.0105 |
| 61 | 2.62 | 0.0201 |
| 121 | 4.53 | 0.0385 |

The first call grows 2.2× from 31 to 121 factors. The pre-I30 Zygote replay rose 5.1× over the
same sizes (2.72, 5.55 and 13.75 s). These measurements use different losses and show the
first-call trends for the two paths. See `lib/MyceliumLearning.jl/src/gradient_couplings.md` for
the replay and implicit-adjoint results. The run used Julia 1.13.1 and `JULIA_NUM_THREADS=auto`.
Its log is `scratch/i30open/infer.log`.
