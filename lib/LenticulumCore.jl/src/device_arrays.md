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
dimensions are small, so this is meant to be correct everywhere, not fast; a backend with a native
factorisation (CUSOLVER on CUDA, Q8) specialises `_symmetric_cholesky` on its array type. A device
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
  Not measured; Q8 measures it on hardware.
- The Poisson likelihood's observed counts are validated and their log-factorials taken on the host
  (`loglikelihood_derivatives`, LP-092), since `_count`'s error message cannot compile into a kernel.

**Not done.** No hardware GPU has run any of this (Q8). Eigen and SVD have no device algorithm. AD
through the device paths is not tested here (the device AD tests of Mycelium, MyceliumLearning and
ImplicitLayers stay behind `LENTICULUM_DEVICE_TESTS`). Views and `SArray` are checked for values only;
the array type of their results is free.

**Tests.** `test/support/device_matrix.jl` runs each package's element-type fixtures (and Lenticulum's
M-step fixtures) on `Array`, views, `SArray` (dimensions up to 4), and `JLArray` in Float64 and Float32
with `allowscalar(false)`, checking values against the `Array` call and, on `JLArray`, that every
numeric array of the result is a `JLArray`. Lenticulum's wide fixtures (dimensions 2 and 3, several
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
