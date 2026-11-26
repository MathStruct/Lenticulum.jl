# Device array storage

## Device arrays (Q7)

`device_arrays.jl` holds the storage hooks the factor code calls instead of assuming `Array`:
`_device_leaf` finds the first device array among a call's numeric inputs (parameters, messages,
observations), `_on_device` places a host constant beside it, `_eye`, `_matrix` and `_vector`
allocate on the same backend. The CPU methods are the old code paths, so `Array` inputs give the same
results as before, bit for bit (Codex's record of the 99 one-dimensional fixture operations of Q4, and
the session's record of the 102 operations of the wide fixtures, `lib/Lenticulum.jl/test/support/wide_fixtures.jl`,
against `dev-danielb` at `7e6a6659`). The device methods live in `ext/LenticulumCoreGPUArraysCoreExt.jl`.

**Dense SPD algebra on the device.** A right-looking Cholesky over column slices (`DeviceCholesky`),
forward and backward substitution over row slices, the inverse as a solve against an identity on the
same backend, and the log-determinant as a sum over the pivots' logarithms. It is written in broadcasts
and views only, so it runs on any GPUArrays backend; an $n\times n$ factorisation is $O(n)$ kernel
launches and one pivot read per column (to report the first nonpositive pivot as LAPACK does). Factor
dimensions are small, so this is meant to be correct everywhere, not fast. The factor is a
`DeviceCholesky` (in `device_arrays.jl`: the dense lower factor and LAPACK's `info`), and its solves go
through the hook `_device_triangular_solve`, so a backend with native kernels replaces both.

**CUDA (Q8).** `ext/LenticulumCoreCUDAExt.jl` (weak dependency CUDA) does that for `CuMatrix{Float32}`
and `CuMatrix{Float64}`: cuSOLVER's `potrf` on the upper triangle and cuBLAS's `trsm`. cuSOLVER reports a
nonpositive pivot but not a NaN one, and the generic kernel does (as reference LAPACK's `dpotrf2`), so
the extension checks the factor's diagonal with one reduction and the two device paths agree. The CPU
path does not: OpenBLAS's `potrf` returns `info = 0` for a matrix with a NaN (found by Mistral's review,
LP-093), so a NaN precision fails a device factorisation and passes a CPU one. That difference predates
Q8 and is left as it is. Other element types (dual numbers,
integers) keep the generic kernels. On an RTX 3080 the generic kernel is 3–36× slower from n = 8 on, and
faster only at n = 2 (38 against 48 µs, the NaN check being 13 µs of those; `gpu/device_benchmarks.md`).
`test/cuda_ext.jl` compares the two with each other and with LAPACK on random positive definite,
indefinite, singular, NaN and Inf matrices; that comparison found the generic kernel computing the
diagonal entry of an infinite pivot as Inf/√Inf = NaN where LAPACK gives Inf (fixed: the pivot is set to
its square root). A device
matrix in a `Symmetric`, `Adjoint` or `Transpose` wrapper is materialised on the device before it reaches
these kernels (LP-092; before that the CPU fallback's `Matrix(M)` copied it to the host without a word).

**What goes to the host, explicitly.**
- Spectral decisions (eigen, SVD: hard constraints, `ExtendedGaussian`, the flat residual metric, the
  diffuse cluster integral) go through `_host_spectral`: one copy of the small matrix to the host, the
  decomposition there, the results back on the input's array type. A wrapped device matrix (`Diagonal`,
  `Adjoint`) is copied by its stored array (LP-092).
- `_entry(A, i)` reads one element through a one-element reduction. It is used where a scalar is control
  flow or metadata (a branch on a count, the set of active regimes, a mixture weight multiplying a
  component), 57 call sites in `lib/*/src`. Each read is a device synchronisation: correct on every
  backend, but per element, so loops over regimes or quadrature nodes pay one synchronisation per term.
  Measured on an RTX 3080 (Q8): 5 µs per read, against 7 µs for a broadcast kernel and 15 µs for a
  `dot`, so they do not dominate any measured path (`gpu/device_benchmarks.md`).
- The Poisson likelihood's observed counts are validated and their log-factorials taken on the host
  (`loglikelihood_derivatives`, LP-092), since `_count`'s error message cannot compile into a kernel.

**On hardware (Q8).** `gpu/runtests.jl` runs the device test files of every package on CUDA (the files
read their device array type from `Main.LENTICULUM_DEVICE_ARRAY`, JLArray when it is not set); the
tests run locally only (a CI job for a self-hosted runner is kept, commented out, in `gpu/GPU.yml`). JLArrays runs its kernels on the CPU, so it
accepts a broadcast that mixes a host and a device array; CUDA does not. The first hardware run found two
such broadcasts that every JLArray test had passed: `ScaleShiftFactor`'s message given no information
(a host `extended(nothing, …)` against device parameters) and `NonlinearFactor`'s rank cleaning of a
difference Jacobian (host scales against the device rounding bounds); both fixed. A device is slower
than the CPU for single factors up to dimension 256 (`gpu/device_benchmarks.md`).

**Not done.** Eigen and SVD have no device algorithm. AD through the device paths is not tested here
(the device AD tests of Mycelium, MyceliumLearning and ImplicitLayers stay behind
`LENTICULUM_DEVICE_TESTS`, and run on CUDA in `gpu/runtests.jl`). VariationalDiffusion's device tests
are inline in its `runtests.jl` and run on JLArrays only. Views and `SArray` are checked for values only;
the array type of their results is free.

**Tests.** `test/support/device_matrix.jl` runs each package's element-type fixtures (and Lenticulum's
M-step fixtures) on `Array`, views, `SArray` (dimensions up to 4), and `JLArray` in Float64 and Float32
with `allowscalar(false)`, checking values against the `Array` call and, on `JLArray`, that every
numeric array of the result is a `JLArray` (on CUDA, a `CuArray`: `device_array()`). Lenticulum's wide fixtures (dimensions 2 and 3, several
channels, diagonal and dense noise) run in the slow tier (`test/device_wide.jl`).

**Difficulties.** The one-dimensional fixtures hid three defects that Mistral's wider fixtures and probes
showed (LP-092): `Array(::Diagonal{<:JLArray})` indexes elementwise, a throwing helper broadcast over
device data does not compile into a kernel, and `Symmetric` of a device matrix took the CPU path silently.
The CPU path stayed bitwise equal but became up to 1.65× slower (the benchmark, LP-092), through lost
inference only: a `_host_spectral` closure that assigned to a name of the enclosing function (which boxes
it), varargs passed on to `_first_device` without specialisation, and a loop over heterogeneous tuples in
`_first_device`; and Mycelium's `_ClusterJoint`, parametric now, read behind a `@nospecialize` argument
(a function barrier in `_batch_joint`). Closures handed to `_host_spectral` use fresh local names; `lib/Lenticulum.jl/test/lp092.jl`
checks inference of these paths.

## Q20 device message paths

Host-side mixture selection and energy-wrapper assembly use `_host_values` to download a numeric working array in one transfer. The CPU method returns its argument directly.
