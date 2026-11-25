#problem

> **Status: closed (2026-11-25).** Q8 (the device tests on CUDA, the device benchmarks, cuSOLVER for the
> device Cholesky) and what it found. The first run of the device tests on a GPU (an RTX 3080) found two
> defects that every JLArray test had passed, both a host array broadcast against a device array, which
> JLArrays accepts because its kernels run on the CPU. Comparing the new cuSOLVER Cholesky with the generic
> kernel on generated inputs found a third in the generic kernel. All three are fixed, each with the test
> that shows it. Of the two reviews, Mistral's showed that a claim of the change about LAPACK was wrong for
> the CPU path (corrected in the comments and notes), and Gemini's one finding was not reproduced. Related:
> [[LP-092 Findings of the reviews of Q7|LP-092]], [[Backlog Q Quality and Engineering]] (Q8, Q20),
> [[Shared Solver Layer]].

> Sources: the session's GPU runs (`gpu/runtests.jl`, logs in the session's scratch directory); Mistral Large
> (`vibe --agent review`, \$0, 100 turns, turn limit reached after three probes, 23 commands refused in the
> headless run; its answer file and probes in `scratch/q8r/` of `../Lenticulum.jl-mistral`); Gemini Hydrogen
> (`agy --mode plan`, `gemini-4-hydrogen-high`, read-only, one finding). code:
> `lib/LenticulumCore.jl/ext/LenticulumCoreCUDAExt.jl`, `lib/LenticulumCore.jl/ext/LenticulumCoreGPUArraysCoreExt.jl`,
> `lib/Lenticulum.jl/src/scale_shift.jl`, `lib/Lenticulum.jl/src/nonlinear.jl`,
> `lib/LenticulumCore.jl/test/cuda_ext.jl`, `lib/LenticulumCore.jl/test/device_arrays.jl`, `gpu/`.

## Found by running on the GPU

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | `ScaleShiftFactor`'s message given no information built `extended(nothing, …)` on the host (`zeros`, `Matrix(I)`) and broadcast it against device parameters: "cannot use a CPU array in a CUDA kernel" | yes | `_no_information_like(s, d)` allocates like `s` | the device matrix on CUDA (`ScaleShiftFactor`, `message_*_given_latent`) |
| 2 | `NonlinearFactor`'s rank cleaning of a difference Jacobian broadcast host scales (`_error_scales` returns host vectors) against the device rounding bounds `R`, and the host projector against the device `J` | yes | the rank decision runs on the host (`R` copied there; it carries values only), the projector and `Jv` go back to `J`'s array type; CPU results unchanged (`_on_device(nothing, x) === x`) | `lp092.jl` "default difference Jacobian" on CUDA |
| 3 | The generic device Cholesky computed the diagonal entry of an infinite pivot as Inf/√Inf = NaN; LAPACK and cuSOLVER give Inf | yes (reproduced on JLArrays with the old code: `[2 0.5; 0.5 Inf]` gave `NaN` for `L[2,2]`) | the pivot is set to √pivot | `device_arrays.jl` "Q8: generic device Cholesky agrees with LAPACK" (JLArrays, every suite run); `cuda_ext.jl` |

JLArrays cannot show 1 and 2: its broadcast of a JLArray with a `Vector` runs on the CPU and succeeds. So
device paths added later (Q20) should run on CUDA as well (`gpu/runtests.jl`), not only JLArrays.

## Mistral's review

| # | finding | right? | what was done |
|---|---|---|---|
| M1 | The CUDA extension's contract holds on 184 generated cases (SPD, indefinite, NaN and Inf, junk in the lower triangle, n = 0 to 64, both types): `info` equals the generic kernel's, the factor, solves, `inv`, `logdet`, `_triangular_inverse` agree with LAPACK, results stay on the device | — (no defect) | — |
| M2 | OpenBLAS's `potrf`, the CPU path, does not flag NaN: `cholesky(Symmetric([NaN;;]); check = false).info == 0`. The change's comments said cuSOLVER differs from LAPACK there; it differs from the generic kernel (and reference LAPACK's `dpotrf2`), not from the LAPACK the CPU runs | yes (checked: `[NaN;;]`, `[1 NaN; NaN 1]` give `info = 0`, `issuccess` true on the CPU) | comments and notes corrected. The device paths keep flagging NaN, so a NaN precision fails a device factorisation and passes a CPU one; that difference predates Q8 (Q7's generic kernel) and is left as it is, recorded in `device_arrays.md` |

Its third probe (the `ScaleShiftFactor` path on CUDA) did not finish; the session's GPU run covers it (finding 1).

## Gemini's review

| # | finding | right? | what was done |
|---|---|---|---|
| G1 | With `info != 0` from `potrf` the NaN check is skipped, so a NaN pivot before a negative one (`[NaN 0; 0 -1]`) would report the later pivot (2) instead of 1 | not reproduced: the NaN reaches every later pivot (0/NaN is NaN), so `potrf` does not stop at the −1; the CUDA path gives `info = 1` and `2` on Gemini's two inputs, as the generic kernel does | no code change; both matrices added to `cuda_ext.jl` |

Gemini read the rest (the generic kernel's change, `scale_shift.jl` and `nonlinear.jl` on CPU and dual
inputs, the runner and the workflow) and found nothing.

## Not checked

A CI job (the GPU tests run locally only; `gpu/GPU.yml` is commented out); the GPU tests on another GPU or CUDA
version; VariationalDiffusion's device tests on CUDA (inline in its `runtests.jl`, not in a file the
runner includes).
