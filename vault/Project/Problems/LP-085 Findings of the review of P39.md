#problem

> **Status: closed (2026-11-23).** P39 (phase 2 of the shared solver layer: Newton-or-descent and the
> trust region as one column driver in ImplicitSolvers; Codex) and what its review found. Mistral found
> no defect; one behaviour change the design did not list (a non-finite trial residual shrinks the
> radius) is recorded with a regression test. Related: [[Shared Solver Layer]] ("Phase 2"),
> [[LP-084 Findings of the review of P38|LP-084]], [[Feature Backlog]] (P39).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; brief
> `~/.cache/lenticulum/codex-queue/p39-brief.md`, answer
> `../Lenticulum.jl-Review-2026-11-23/p39-solver-driver/IMPL-p39-solver-driver.md`, scripts and logs in
> `scratch/p39/` of `../Lenticulum.jl-codex`). Mistral Large (`vibe --agent review`, five probes, stopped at
> its turn limit with a partial answer file: `../Lenticulum.jl-Review-2026-11-23/p39-solver-driver/ANSWER-mistral.md`,
> probes in `scratch/p39r/` of `../Lenticulum.jl-mistral`). code: `lib/ImplicitSolvers.jl/src/descent.jl`,
> `lib/VariationalDiffusion.jl/src/implicit.jl` (`implicit_infer`), `lib/VariationalDiffusion.jl/src/many.jl`
> (`implicit_infer_many`). tests: `lib/ImplicitSolvers.jl/test/descent.jl`, `lib/ImplicitSolvers.jl/test/device.jl`.

## What P39 does

`solve_columns(problem, U0, alg; tol, maxiters)` with `NewtonDescent` or `TrustRegion` solves one root
problem per column; `problem` is a named tuple of callbacks on the unfinished columns. The steps, radii,
the ratio rule, the noise rule and the radius floor are VariationalDiffusion's, moved unchanged (D8–D9).
`implicit_infer` and `implicit_infer_many` build the callbacks and call it; the four loops (one query and
a batch, each method) are gone. Codex recorded $z$, `iters`, `converged`, `stable` of every call of the two
functions in VariationalDiffusion's suite (2,146), the eight smoke examples (235, the threaded
calibration examples matched by input), the Reactant suite (14), a reduced JLArrays version of
`gpu_energy.jl` (8) and a `Float32` probe (100) before W7 and compared after: bitwise equal, no call near
D10's rounding boundary (`scratch/p39/comparison.log`). On the way it found and fixed a precision change
of its own first version (a `Float64` Newton direction narrowed through a `Float32` residual buffer); the
optional `round_steps`, `residual_norms` and `trial_residual` members keep the old rounding and evaluation
order.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| — | no defect: a seven-column batch (columns finishing at iterations 0 to 60, one NaN column) equals each column alone bitwise in `U`, report and `stable`, for both algorithms, and in permuted order; `nfe` equals a counting callback; `maxiters` 0 and 1, `tol = 0`, the empty batch, $n_F = 0$, the `Float32` noise rule, a Gauss–Newton non-root minimum (`:stalled`), `precision = 1e-3`, the hard case of the exact step, `residual_norms`, `round_steps` (probes 1, 2) | — | — |
| — | old against new VariationalDiffusion (probes 3, 4; run, outputs written, not entered in the answer file): `implicit_infer` with `:auto`, `:newton`, `:trust_region` × `DenseSolve`, `KrylovSolve`, continuation, `maxiters` 0, 1, 3, `tol = 1e-12`, a fixed radius (114 lines); `implicit_infer_many` on the ring, the analytic relation and mixtures, hard starts, `Z_init`, soft clamps, `maxiters` 0 and 1, every coordinate clamped (268 lines) | — | the session compared the outputs: byte-identical (`out3_*.txt`, `out4_*.txt` in the review folder) |
| O1 | a trial point with a non-finite residual: the old loop computed a NaN ratio, which fails every comparison, so the radius stayed and the same step was retried until `maxiters`; the driver sets the ratio to 0 and shrinks the radius (probe 2) | right; a behaviour change the design's "must not change" did not list. No recorded call reaches it (Codex's 2,503). The new behaviour is the intended one (the iteration backs away from where the relation is undefined) | kept; documented in `descent.md` and in the design note; test "LP-085" in `lib/ImplicitSolvers.jl/test/descent.jl` (root at 0, NaN below 0.5: ends `:stalled` just above 0.5 in two iterations). With the old rule restored in the driver the test fails (`:maxiters` after 40 iterations, and the existing Gauss–Newton `:stalled` test fails too) |

Not a finding, noted by the session while reading the diff: `many.jl`'s trust-region `scale` callback
evaluates the batch Jacobian and caches it for `stable` and `model`, which look their column up by
`searchsortedfirst`. That depends on the driver calling `scale` on all unfinished columns before
`stable` and `model` in each iteration, which it does; the design note says so now. Codex's comparison
and Mistral's probe 4 (columns leaving in different orders) cover it.

## Gates

On `dev-danielb` with the change applied: ImplicitSolvers, VariationalDiffusion and ImplicitLayers suites
passed (`~/.cache/lenticulum/p39/`); ImplicitSolvers again after the LP-085 test. Codex ran both
documentation builds. Not run: MyceliumLearning and Lenticulum (they use ImplicitSolvers' linear solves
only; the change adds names and touches no existing function), device tests on a GPU (`gpu_energy.jl`
found no CUDA device; its JLArrays reduction was compared instead), the MNIST examples.
