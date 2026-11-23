#problem

> **Status: closed (2026-11-23).** P14 W4 (MyceliumLearning's graph adjoint onto ImplicitSolvers, Codex)
> and what its review found. `:dense` goes through `ImplicitSolvers.linsolve`, a new matrix-free
> `:gmres` through `fixed_point_adjoint`, `:krylov` is unchanged until W5. Mistral's probes found no
> defect; two of its missing tests are added. Codex found one defect that predates W4: the device
> pinning of I30 fails under JLArrays 0.5 (fixed, below). Related: [[Shared Solver Layer]] (W4),
> [[LP-079 Findings of the reviews of P14 W2 and W3|LP-079]], [[LP-077 Findings of the review of I30|LP-077]],
> [[Feature Backlog]] (P14).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; brief and answer in
> `../Lenticulum.jl-Review-2026-11-23/p14-w4/` (`w4-brief.md`, `IMPL-p14-w4.md`), its scripts in
> `scratch/w4/` of `../Lenticulum.jl-codex`). Mistral Large (`vibe --agent review`, \$0, finished with an
> answer file: `../Lenticulum.jl-Review-2026-11-23/p14-w4/ANSWER-mistral.md`, probes in `scratch/w4r/` of
> `../Lenticulum.jl-mistral`). code: `lib/MyceliumLearning.jl/src/graph_adjoint.jl` (`_graph_linear_solve`),
> `lib/MyceliumLearning.jl/ext/MyceliumLearningGPUArraysCoreExt.jl`, `lib/ImplicitSolvers.jl/src/linsolve.jl`.
> tests: `lib/MyceliumLearning.jl/test/graph_adjoint.jl`, `test/graph_adjoint_devices.jl`.

## What W4 measured (Codex)

Gradients recorded on the old code (`8954aa78`) for the 18 implicit cases of `test/graph_adjoint.jl`,
recomputed after the change (`scratch/w4/record.jl`, `compare.jl`): all 52 `:dense` and 27 `:krylov`
gradients bitwise equal; `:gmres` against the old `:dense` at most $2.0\cdot10^{-10}$ absolute,
$8.3\cdot10^{-10}$ relative (the reference-difference cases; under ForwardDiff and Zygote at most
$2\cdot10^{-16}$). `@time using MyceliumLearning`: median 32.8 → 33.7 ms (+2.8 %). The tolerance passed to
the shared solvers is `Tolerance(atol = linear_tol/2, rtol = linear_tol/2)`, whose bound
$\tau(1+s)/2 \le \tau\max(1,s)$ implies the graph's own residual check, which stays for all three solvers.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| — | no defect: dense bitwise equal to the old code on random, ill-conditioned, nearly and exactly singular systems, same errors; `:gmres` solves $(I-D^\top)\lambda = r$ (not $I-D$), restarts, budgets, `Float32`, without Krylov loaded, `parameter_step` (probes 1–6) | — | — |
| O1 | a singular $I-D^\top$ with a nonfinite right-hand side now gives "produced nonfinite values" where the old code gave "I-DT is singular" (`probe1b.jl`) | right | kept: both refuse with `ArgumentError`, and the input is nonfinite; not pinned by a test |
| O2 | in `Float32`, the rounded shared bound can exceed the graph's by about $10^{-8}$ relative, so a solve accepted by GMRES could be refused by the graph's check (`probe4.jl`) | right, not hit by any solve | kept: the graph's check decides, a refusal there is the specified behaviour |
| M1, M2 | no test crosses GMRES's restart (the largest test system, $n = 36$, converges in one step), and none checks `:gmres` on a nonsymmetric system directly | right | added: "P14 W4: graph adjoint solvers on a nonsymmetric system across GMRES restarts" ($n = 50$, upper-triangular $D$, `:dense` and `:gmres` against $(I-D^\top)^{-1}r$; `linear_maxiter = 30` refused, so the solve needs more than one cycle) |
| M3 | no test of `:gmres` in a process without Krylov | right | not added (a subprocess test); Codex's `scratch/w4/gmres-without-krylov.jl` and Mistral's `probe3.jl` show it works |
| M4 | `Float32` implicit adjoints run only in the device tests | right | not changed: those run in CI's device job |
| M5 | O1's corner not pinned | right | not added |

The dense path tells a failed factorization by `report.iterations == 0`. That was ImplicitSolvers'
behaviour but not written down; `DenseSolve`'s docstring and `linsolve.md` now state it.

## The device pinning under JLArrays 0.5 (Codex, before W4)

With JLArrays 0.5.0 (what a fresh resolve gives; the local Manifests had 0.4.2), the testset "I30: pinning
flat device coordinates" of `test/graph_adjoint_devices.jl` errors: `KernelError: passing non-bitstype
argument`. `_pin_coordinates_wide`'s broadcast closure captured the type `T` (to promote with `zero(T)`),
and a closure over a `Type` is not isbits, so it cannot be a kernel argument; JLArrays 0.5 compiles its
broadcasts as kernels, and a real GPU would refuse it the same way. Reproduced by the maintainer's session
in a scratch environment with JLArrays 0.5.0 (the header of `graph_adjoint_devices.jl` and that testset):
on the old code 1 error, with the fix (the closure captures the value `zero(T)`) 7 of 7 pass. That testset
is the regression test; it shows the defect only with JLArrays ≥ 0.5.

## Gates (2026-11-23)

With the review's test and the fix (`dev-danielb`): ImplicitSolvers' suite and MyceliumLearning's with
`LENTICULUM_DEVICE_TESTS=true`, each ending `Testing … tests passed` (`~/.cache/lenticulum/w4/`);
`test/graph_adjoint.jl` and `test/graph_adjoint_devices.jl` under JLArrays 0.5.0, no error or failure
(`devices-jl05.log`). Lenticulum, both tiers (`LENTICULUM_SLOW_TESTS=true`), on Codex's commit `4d25453f`:
passed in 57 min 40 s (`Lenticulum.log`); the fix and the added test came after it and touch only the
device extension and MyceliumLearning's tests. Codex ran MyceliumLearning's suite with and without the
device tests on its commit. ImplicitLayers (which depends on MyceliumLearning) on the final tree: passed in
19 min 19 s (`ImplicitLayers.log`). Not run: Lenticulum's suite on the final commit, the docs build,
VariationalDiffusion, AlgebraicFactors and AdversarialFactors (they do not depend on MyceliumLearning), a
real GPU.
