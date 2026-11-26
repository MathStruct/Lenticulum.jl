#problem

> **Status: closed (2026-11-26).** Q20 (device arrays on every message path) and what its checks found.
> Codex ran Q7's device matrix over Q19's operations (`inference_cases`) and added views of device arrays,
> graph tests beyond belief propagation and tests of the LP-092 claims, and fixed what failed there on
> JLArrays (see the Q20 section of [[LP-092 Findings of the reviews of Q7|LP-092]]). Its sandbox could not
> reach the GPU. The session's CUDA run then failed 64 checks at six host paths that JLArrays 0.4 accepts:
> host placeholders and host copies meeting device arrays in a BLAS product or a broadcast. Mistral's review,
> run on JLArrays 0.5 (which refuses BLAS on a device array, as CUDA does), found the same plate and
> mean-field paths and one more, the learned-noise `GaussianPrior` constructor. All seven fixed; the
> package test environments still pinned JLArrays 0.3/0.4, so the CPU suites did not catch this kind (row Q21,
> done: [[LP-096 Findings of the review of Q21|LP-096]]).
> Gemini's quota was used up: no second review. Related: [[Backlog Q Quality and Engineering]] (Q7, Q8, Q20,
> Q21), [[LP-093 Findings of the hardware GPU run and the review of Q8|LP-093]].

> Sources: Codex (`gpt-6.2-sol`, report in `../Lenticulum.jl-codex/scratch/q20/REPORT.md`); the session's
> CUDA runs (`gpu/runtests.jl` on an RTX 3080, logs `scratch/q20/gpu-cuda*.log` there); Mistral Large
> (`vibe --agent review`, \$0, 100 turns, turn limit, many shell calls refused in headless mode; answer
> file and logs in `scratch/q20r/` of `../Lenticulum.jl-mistral`, JLArrays 0.5.0 in `scratch/q20r/env`).
> code: `lib/Lenticulum.jl/src/hard_constraint.jl`, `lib/Lenticulum.jl/src/switching.jl`,
> `lib/Lenticulum.jl/src/constraint.jl`, `lib/Lenticulum.jl/src/gaussian.jl`, `lib/Mycelium.jl/src/gaussian.jl`,
> `lib/Lenticulum.jl/test/device_inference.jl`, `lib/Lenticulum.jl/test/device_graphs.jl`,
> `lib/Lenticulum.jl/test/device_paths.jl`.

## Found on CUDA (session)

First run: 4734 passed, 64 errored, 1 broken (LenticulumCore, Mycelium, Lenticulum, AlgebraicFactors).
Each error traced to its first source frame:

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | `_hard_block_parts` copies a block to the host (`Matrix(b.X)`, `Matrix(b.η)`, `zeros(...)`), and `_hard_shared_messages` passes `Matrix(At)` and the host `Bool` placeholders of an absent channel to the shared kernel: `gemm!` with a device and a host matrix (`hard_constraint.jl` 356, 405, 410; 44 errors, plate messages of `HardConstraintFactor`) | right | the block's storage kept (`_matrix`, `_zeros_like`); the parts moved to the device of `Q` before the kernel; `_eye` for the kernel's identity | `device_inference.jl` on CUDA, `plate_message_*` |
| 2 | the switching plate message to the outcome takes host placeholders from `_block_parts` of an absent channel (`switching.jl` 620; 9 errors) | right | `_parts_on_device` | same, `plate_message_*_k_given_*` |
| 3 | `_switch_mixture_block` broadcasts a host `zeros(Bool, K, n)` outcome block with device weights (`switching.jl` 779; 6 errors) | right | moved to the device of `Q` | same, `plate_message_*_given_latent` |
| 4 | `belief_distance` of a device extended Gaussian and the host extended form of a `TrivialBelief` (`Mycelium gaussian.jl` 104; the mean-field graph) | right | both moved to a common device | `device_graphs.jl` "Mean field" on CUDA |
| 5 | second run (after 1–4: 3701 passed, 1 errored), unchanged after 6: the mean-field product belief copies covariances to the host (`constraint.jl` `_product_belief`, `Matrix(belief_cov(b))`), then `_frozen_residual` multiplies device and host | right | `_matrix` | same |
| 6 | the lift of a hard input in a frozen Gaussian belief built on the host (`Matrix{T}(I, …)`, `Matrix(l[2])`, `gaussian.jl` 927) | found by reading while tracing 5 (fixed first, did not clear it); not shown by a test | `_eye`, the device of the mean | none beyond the suites; recorded as untested |
| 7 | `_noisy_twin` (mean-field energy of a hard constraint) checks `isposdef` on the host (`hard_constraint.jl` 476) | right (JLArrays 0.5, `device_graphs.jl`, 1 error) | `_precision_isproper` on a device | `device_graphs.jl` on JLArrays 0.5 |

These cannot be shown on JLArrays 0.4, which runs a BLAS product of a host and a JLArray and a mixed
broadcast on the CPU; the regression tests are the CUDA run and a JLArrays 0.5 run of the same files.

## Mistral's review

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| M1 | `GaussianPrior(:x, μ, Q; learn_noise = :full)` with device `μ, Q` calls `isposdef(Symmetric(Q))`, LAPACK on a device array (`gaussian.jl` 700) | right (JLArrays 0.5: "Illegal conversion of a JLArray to a Ptr") | `_precision_isproper` on a device | `device_paths.jl`, direct constructors, on JLArrays 0.5: errored before, passes after |
| M2 | `device_inference.jl` errors in `_hard_shared_kernel` (host `At` times device) | right; the same as finding 1 | finding 1 | same |
| M3 | `device_graphs.jl` "Mean field" errors in the free energy (host times device) | right; the same as findings 5 and 7 | findings 5, 7 | same |

Not checked by Mistral (turn limit): CPU bitwise equality, type inference, the dense-Cholesky
`@test_broken`. The session ran the first (below); Codex's report has `@inferred` spot checks on four paths
and `test/lp092.jl`'s inference tests pass.

## After the fixes

- CUDA, `julia --project=gpu gpu/runtests.jl Lenticulum Mycelium AlgebraicFactors LenticulumCore
  MyceliumLearning`: 5377 passed, 1 broken (the dense Gaussian score replay: reverse AD reaches the device
  Cholesky), 15 min 47 s.
- JLArrays 0.5 (`scratch/q20r/env`): `device_inference.jl` 2520 passed, `device_graphs.jl` 102 passed,
  `device_paths.jl` 57 passed, 1 broken.
- Float64 recorder of Q19 (`scratch/q19/probe.jl`): 462/462 operations bitwise unchanged.
- Suites, run in the Codex worktree: Lenticulum with `LENTICULUM_SLOW_TESTS=true`, LenticulumCore,
  Mycelium and MyceliumLearning with `LENTICULUM_DEVICE_TESTS=true`, AlgebraicFactors: all passed.

Not run: the benchmark comparison against `dev-danielb` (Codex's four CPU spot checks only); ImplicitLayers
and VariationalDiffusion suites (not touched); a second review (Gemini's quota used up, 2026-11-26).
