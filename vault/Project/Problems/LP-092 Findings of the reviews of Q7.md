#problem

> **Status: closed (2026-11-25).** Q7 (device-generic factors; Codex) and what its two reviews found. Eight
> defects were reproduced by a failing test and fixed, each test kept in `lib/LenticulumCore.jl/test/device_arrays.jl`
> or `lib/Lenticulum.jl/test/lp092.jl`. Three came from Mistral's probes and five from Gemini's reading. All of them
> sat on shapes or configurations that Q4's one-dimensional fixtures do not build: wrapped device matrices,
> dimension 2, a default finite-difference Jacobian, a predictor that fixes the latent. Gemini also listed device
> defects on the message paths that Q7's row does not cover: mean-field, plates, factor beliefs, EP and mixture
> reduction, clusters. Two were reproduced; the rest are claims. They are row Q20. The benchmark against
> `dev-danielb` then found the CPU path up to 1.65× slower in four places, from lost inference; fixed, with an inference test. Related:
> [[Engineering Standard]], [[Backlog Q Quality and Engineering]] (Q7, Q8, Q19, Q20), [[Feature Backlog]].

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, medium effort; brief in `scratch/q7/brief.md` of
> `../Lenticulum.jl-codex`; it stopped at its 5-hour usage limit with the implementation done and most gates run, but
> without its final report, so this note is written from its logs, `scratch/q7/`). Mistral Large (`vibe --agent
> review`, \$0, 100 turns, no answer file: many of its commands were refused in the headless run; its probes in
> `scratch/q7r/` of `../Lenticulum.jl-mistral`, which the session ran). Gemini Hydrogen (`agy --mode plan`,
> `gemini-4-hydrogen-high`, read-only, no commands; a first run ended with no output because a shell command was
> refused; the answer in the session's scratch directory). code: `lib/LenticulumCore.jl/src/device_arrays.jl`
> (and `.md`), `lib/LenticulumCore.jl/ext/LenticulumCoreGPUArraysCoreExt.jl`,
> `lib/LenticulumCore.jl/test/support/device_matrix.jl`, `lib/Lenticulum.jl/test/support/wide_fixtures.jl`,
> `lib/*/test/device_*.jl`, `lib/Lenticulum.jl/test/lp092.jl`.

## What Q7 does

Every factor type's messages, `energy`, `local_free_energy` and M-step run on a device array type with scalar
indexing disallowed, and return arrays of the input's array type. The design, written before Codex started:
dense SPD algebra on GPU arrays is written in broadcasts and views (a right-looking Cholesky, triangular
substitution, inverse, log-determinant); eigen and SVD rank decisions go through one helper,
`_host_spectral`, which copies the small matrix to the host and returns the results on the input's array type;
there is no `@allowscalar` in `lib/*/src`; Adapt.jl is a hard dependency of LenticulumCore, Mycelium and
Lenticulum, and every struct holding numeric arrays has `adapt_structure`, the factor graph among them; the CPU
path is unchanged. Before Q7, LenticulumCore refused dense device Gaussians on purpose ("Hardware dense Gaussian
algebra is not claimed", `gaussian_belief.md`); Q7 reverses that.

The test (`device_matrix.jl`) runs each package's element-type fixtures, and Lenticulum's M-step fixtures, on
`Array`, views, `SArray` and `JLArray` (Float64 and Float32). It checks values against the `Array` call and, on
`JLArray`, that every numeric array of the result is a `JLArray`. Lenticulum 458 checks in the default tier
(2 min 55 s alone), AlgebraicFactors 68, LenticulumCore 112. `device_graphs.jl` adapts a two-dimensional Kalman
chain and a switching model to `JLArray`, runs inference and compares beliefs and free energy with the CPU run.

One deviation from the design: a scalar needed as control or metadata is read through `_entry(A, i)` (a
one-element reduction), at 57 call sites. The design asked for one read of a reduced value. Each `_entry` is a
device synchronisation, so loops over regimes or quadrature nodes pay one per term. This is correct on every
backend and not measured; Q8 measures it on hardware.

CPU results are bitwise unchanged: Codex's record of Q4's 99 fixture operations (`scratch/q7/record.jl`), rerun
by the session after all fixes (99/99), and the session's record of the 102 operations of the wide fixtures
against `dev-danielb` at `7e6a6659` (identical).

## Mistral's probes (run by the session)

| # | finding | right? | done |
|---|---|---|---|
| 1 | `_chol`, `_precision_factor` on a non-SPD device matrix throw `PosDefException`, not `ArgumentError` | no: the CPU `_precision_factor` throws `PosDefException` too; `_chol` throws `ArgumentError` on both | none |
| 2 | Float32 device solve at condition $10^{10}$, 13 % relative error | no: Float32 at that condition number has no correct digits on any backend | none |
| 3 | A `Symmetric` (or `Adjoint`, `Transpose`) wrapper around a device matrix took the CPU path: `Matrix(M)` copied it to the host silently, and `belief_mean`, `belief_cov` of `GaussianBelief(JLArray, Symmetric(JLArray))` returned host arrays; `_symmetric_cholesky` of a device `Diagonal` or `Adjoint` threw | right (test "LP-092: wrapped device matrices stay on the device", 20 failed, 3 errors) | the GPU extension materialises wrapped device matrices on the device before the kernels |
| 4 | Nonlinear factor with a device `Diagonal` Jacobian, dimension 2: `message_x_given_y` scalar-indexed (`Array(::Diagonal{<:JLArray})` in `_host_spectral`) | right (probe 2, wide fixtures; `lp092.jl`) | `_host_matrix` copies a wrapped device array by its stored array |
| 5 | Poisson EP with a two-dimensional latent: `_count.(y)` broadcast on the device; its error message cannot compile into a kernel | right (probe 2; `lp092.jl`) | counts validated and log-factorials taken on the host, moved to `z`'s array type |

Mistral's probe 2 (`probe2_fixtures.jl`: dimensions 2 and 3, several channels, diagonal and dense noise, M-steps
3 → 2) never ran in its session, because it built its case list with the wrong element type. Fixed by the session,
it found 4 and 5. It is now `lib/Lenticulum.jl/test/support/wide_fixtures.jl`, run on every array type in the
slow tier (`test/device_wide.jl`, 510 checks, about 3 min).

## Gemini's review

Gemini listed 11 groups of findings. The session reproduced each claim inside Q7's scope with a probe
(`scratch/q7/gemini_probe.jl` of `../Lenticulum.jl-codex`):

| finding | right? | done |
|---|---|---|
| `NonlinearFactor` with its default Jacobian (central differences) scalar-indexes (`x[i]`, `e[i] = h`) | right (`lp092.jl`, both directions threw) | steps chosen on the host from one copy of `x`, step vectors moved to the device; CPU path unchanged |
| A likelihood whose predictor fixes the latent returns a host `uninformative` | right (`lp092.jl`) | `_uninformative_like` allocates on the input's array type |
| `BivariatePoisson`'s conditional entropy reads `λ[1]`, `λ[2]`, `λ[3]` | right (`lp092.jl`) | the support sizes, which are control, are computed on the host |
| `variable_entropy(::CategoricalBelief)` iterates the device log-probabilities (the Bethe entropy of a categorical variable of degree > 1) | right (`lp092.jl`) | a reduction on the device; the CPU generator unchanged |
| The device Cholesky reads the lower triangle, the CPU one (`Symmetric`, `:U`) the upper: an asymmetric input succeeds on one and fails on the other | right (`lp092.jl`; asymmetric inputs only, every caller symmetrises first) | the device path reads the upper triangle |
| `Diagonal` noise with trivial inputs and `MixtureReduction`'s `_expanded_message` call `inv(C.L)` | not reproduced (both ran on the device with the right values) | none |
| `_symmetric_inv` of a device view, `C \ Diagonal(JLArray)` | not reproduced | none |
| `HardConstraintFactor` with singular nonzero noise | not reproduced | none |
| `learn_noise = :full` messages and free energy; `LogNormalVariancePrior` | not reproduced | none |
| `moment_match` of a Gaussian mixture | not reproduced | none |
| CPU `_symmetric_solve` and `_symmetric_logdet` accept symmetric indefinite matrices, the device ones throw | right as stated, not a defect: every caller passes an SPD matrix (noise covariances, a damped Hessian made positive definite first) | none |
| `reduce_mixture` (`w[keep]`, host covariances) | right (probe 4b threw) | outside Q7's row: Q20 |
| mean-field messages, plate methods (`factor_messages`, `factor_beliefs`, `belief_free_energies`), `factor_belief`, EP (`ep_divide`, `ep_project`), cluster nodes and `cluster_batch`, blocks, `_pin_coordinates`, `belief_logdensity` of a mixture, `SampsonNorm` localisation, `ScoreFunctionCoupling`, `scale_shift_parameters`, constructors called with device arrays | not reproduced (claims from reading) | outside Q7's row: Q20 |
| Test gaps: views of device arrays are not in the matrix; `device_graphs.jl` covers belief propagation on two small graphs only | right | Q20 |

## The benchmark: four CPU slowdowns (session)

`benchmark/compare.jl --base dev-danielb --packages LenticulumCore,Mycelium,MyceliumLearning,Lenticulum
--interleave` (one thread, two interleaved runs each) on the reviewed and fixed tree flagged
`Lenticulum/message/nonlinear` at 1.65× and `Mycelium/grid/flooding_sweep` at 1.61× (with 55 % more
allocations), although the CPU results were bitwise unchanged. The causes were all inference, none
arithmetic:

| cause | effect | fix |
|---|---|---|
| `F = _host_spectral(A -> (F = svd(A); …), …)` in `_rank_clean` (and the same in `_range_svd` of `constraint.jl`): the closure assigns to the enclosing function's `F`, which boxes it | the rest of the function, and `_jacobian` and `linearize` with it, inferred as `Any` (base: `Matrix{Float64}`) | the closure's local renamed |
| `_covariance_at(Q, inputs...)` passed its varargs on to the new `_first_device(inputs)`; Julia does not specialise a method on varargs it only passes on | a dynamic call per covariance, 1 µs where the specialised call takes 0.05 µs | `inputs::Vararg{Any,N} where {N}`; the same for `_compute_eltype` |
| `_first_device` looped over heterogeneous tuples | dispatch per element on every call | unrolled by recursion; an all-host call folds to `nothing` |
| `_ClusterJoint` became parametric (`P, J, M, V`, so that it holds device arrays); `_batch_joint` takes `ps` unspecialised, so the joint's type is not inferred there, and its field types, concrete before, became `Any` | `regime/mstep` 1.15× slower (found by the second benchmark run, after the three fixes above) | a function barrier after the joint is built; now 0.92× the base, `fit_em` 0.84× |

Test: "LP-092: CPU paths stay inferred" in `lib/Lenticulum.jl/test/lp092.jl` (`@inferred` on `_jacobian`,
`linearize`, `_range_svd`, `_first_device`); the first three fail on the boxing code. After the fixes the
CPU records are still bitwise unchanged (99/99, 102/102; `em_step` and `fit_em` on the regime plate: equal
values); the timings by hand, base against head: `message/nonlinear` 2.82 against 2.85 µs (7984 bytes both),
the grid sweep 1430 against 1415 µs, `em_step` 122 against 113 µs. The barrier has no test of its own: the
benchmark is its check. The benchmark rerun on the fixed tree is below.

## Gates (2026-11-25, this machine, `JULIA_NUM_THREADS=auto`, `Pkg.test()`)

Third run, after every fix (`scratch/q7/final3/summary.txt` of `../Lenticulum.jl-codex`): all passed. LenticulumCore
(258 s), Mycelium and MyceliumLearning with `LENTICULUM_DEVICE_TESTS=true` (459 s, 451 s), Lenticulum fast tier
(2267 s) and slow tier alone (2718 s), AlgebraicFactors (257 s), VariationalDiffusion (391 s), ImplicitLayers with
the device tests (1521 s). The two earlier runs (`final/`, `final2/`) passed too.

Benchmark on the same tree (`scratch/q7/bench3/report.md`, one thread, two interleaved runs): no regression flag;
every warm case between 0.84× and 1.09× of `dev-danielb`, the first-call latencies between 1.01× and 1.05×.

Not run: a hardware GPU (Q8), AD through the device paths (the device AD tests of Mycelium, MyceliumLearning and
ImplicitLayers ran with `LENTICULUM_DEVICE_TESTS=true`), Julia 1.12, the docs builds.
