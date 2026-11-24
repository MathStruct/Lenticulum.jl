#problem

> **Status: closed (2026-11-24).** Q5 (the AD backend matrix; Codex) and what its review found. Mistral's review
> was static (its sandbox refused every command). One finding was right and is fixed: a `@test_broken` row passed
> when a backend returned a wrong value, not only when it threw. One was right about the environment: the
> DifferentiationInterfaceTest dependency capped JLArrays at 0.3 in MyceliumLearning's test environment, and it is
> removed there and in Mycelium. Its headline finding (every finite-difference reference throws) was refuted by the
> suites. The session found a row whose Enzyme outcome depends on what was compiled before it, and slow rows that no
> CI job ran. Related: [[Engineering Standard]] (C2), [[Backlog Q Quality and Engineering]] (Q5, Q6),
> [[Feature Backlog]].

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, medium effort, two usage-limit stops, one stop for
> memory; brief and answer in the session's scratch directory, the answer also in `scratch/q5/ANSWER.md` of
> `../Lenticulum.jl-codex`, logs in `~/.cache/lenticulum/q5/`). Mistral Large (`vibe --agent review`, static
> only, \$0: `scratch/q5-review/ANSWER.md` of `../Lenticulum.jl-mistral`). code: `lib/*/test/ad_matrix.jl`,
> `lib/LenticulumCore.jl/test/support/ad_matrix.jl`, `.github/workflows/Slow.yml`.

## What Q5 does

Each of ImplicitSolvers, LenticulumCore, Mycelium, MyceliumLearning and Lenticulum has a `test/ad_matrix.jl`. It
differentiates each entry point with Zygote, ForwardDiff, Enzyme (reverse; forward on two rows) and Mooncake
through DifferentiationInterface, and compares the result with central differences (FiniteDifferences
`central_fdm(5, 1)`, or the package's own `ad = nothing` path), `rtol = 1e-6`. The entry points are factor energies,
`factor_message` and `local_free_energy` of every factor type in Mycelium and Lenticulum, the graph `free_energy`,
`infer!`, `free_energy_gradient` (`:unrolled`, `:implicit`), `moment_statistics`, LenticulumCore's
`energy_gradient`, `energy_jacobian` and `parameter_gradient`, and ImplicitSolvers' `linsolve` (four algorithms),
`solve_root` (Picard, Broyden) and `solve_columns`. A backend that fails today is a `@test_broken` with the
exception in a comment. The shared helper copies the fixture for each backend, so a backend that mutates captured
data before throwing cannot change another backend's input or the reference. The tables are in the source notes:
`linsolve.md`, `energy.md`, `free_energy.md`, `graph_adjoint.md`, `Lenticulum.md`. No `src/*.jl` changed.

The matrix over 71 rows (counted from the five note tables; about ten rows per backend are skips, an entry point
the factor does not have): ForwardDiff passes all 61 it runs; Zygote 43 pass, 18 broken; Mooncake 31 pass, 30
broken; Enzyme 14 pass, 44 broken, 2 unstable. The causes, from the comments: Mooncake fails mostly with
`MooncakeRuleCompilationError` on a `try`/`catch`, 17 times in LenticulumCore's `_chol` (`gaussian_belief.jl` 122),
otherwise in `_emit_plan` and `_improper_check`, or with a missing foreign-call rule. Enzyme fails mostly on runtime
activity, illegal type analysis, and no derivative for LAPACK's `dpotri`. Zygote fails on mutation and on missing
`rrule`s for `sum`/`maximum` with `init`. `free_energy_gradient` refuses Enzyme and Mooncake itself
(`ArgumentError`). No backend returned a wrong value. So one change to `_chol` would move much of the Mooncake
column; this is input to Q6.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| F1 | `only(FD.grad(...))` throws for a vector argument, so every `gradient_row` errors before a backend runs | no: `FiniteDifferences.grad` returns a tuple with one gradient per argument, and `only` unwraps it. LenticulumCore's suite passes with `energy/ParameterResidual` at 27 passing checks through that line | — |
| S1 | DifferentiationInterfaceTest 0.11 and JLArrays collide in the Mycelium and MyceliumLearning test environments | right for MyceliumLearning: its resolved JLArrays went from 0.4.2 to 0.3.3 with the change (Mycelium's local Manifest was at 0.3.3 already). MyceliumLearning did not use it; Mycelium used it for one row | removed from both test environments; Mycelium's `DataFactor` row is a plain check. MyceliumLearning resolves to JLArrays 0.4.2 again |
| S2 | a `@test_broken` row also passes when the backend returns a wrong value | right: the helper wrapped `isapprox` in `try` | the helper runs the backend first: a throw is broken, a correct value an unexpected pass (an error), a wrong value a failure. All five matrices pass with it (gates below), so no broken row returns a value today |
| S3 | `rtol` without `atol` fails on an exact-zero reference component | not a defect: no reference has a zero component (the suites pass) | — |
| S4 | `@test_skip` still runs the skipped backend | no: `@test_skip` does not evaluate its expression | — |
| S5 | MyceliumLearning's `free_energy after infer!` takes its reference from `free_energy_gradient(...; ad = nothing)`, not from differences of its own loss | not a defect: that reference is central differences through the same two sweeps, and Zygote and ForwardDiff agree with it | — |
| S6 | the timings in the notes come from no script | the script is the test file itself, run alone; the notes say so | — |

## Found by the session

- **An Enzyme row whose outcome depends on order.** `local_free_energy/LossFactor` (and `ExposedFactor`, slow) throws
  an Enzyme-internal `UndefVarError: vaEltypes` when the matrix runs alone and returns the correct value after the
  rest of Mycelium's suite. As `@test_broken` it failed the suite as an unexpected pass. The helper has a third class,
  `unstable`: a throw is broken, a returned value must be right. Not reported upstream yet.
- **Slow rows that no CI job ran.** The slow rows of ImplicitSolvers (`KrylovJL`, `solve_columns`), Mycelium
  (`infer!`, `free_energy`, the cluster, overlay and exposed factors) and MyceliumLearning (`moment_statistics`) sit
  behind `LENTICULUM_SLOW_TESTS`, which only Lenticulum's job in `Slow.yml` set. `Slow.yml` has a job `ad-matrix`
  that runs those three suites with `LENTICULUM_SLOW_TESTS=only`, and its push paths include the matrix files.
- **Stale local Manifests.** LenticulumCore's and Mycelium's local Manifests held EnzymeCore 0.8.26 and LLVM 9, so
  `Enzyme = "0.13.221"` did not resolve. Updated (LenticulumCore's old Manifest moved to
  `~/.cache/lenticulum/q5/main/`). CI resolves fresh. Mycelium's JLArrays stays at 0.3.3 after a full update, which
  this change does not cause (not examined).
- **Out-of-scope changes by Codex** were kept: fixtures moved into `test/support/` so the matrix and the existing
  tests share them (no assertion lost; Mistral checked). Its edit of the backlog row was replaced.

## Gates (2026-11-24, this machine, `JULIA_NUM_THREADS=auto`, `Pkg.test()`)

| suite | tier | result | AD matrix (pass, broken) | wall |
|---|---|---|---|---|
| ImplicitSolvers | default | passed | 12, 9 | 6 min 14 s |
| ImplicitSolvers | `only` | passed | 3, 5 | 6 min 35 s |
| LenticulumCore | default | passed | 39, 8 | 4 min 0 s |
| Mycelium | default | passed | 33, 23 | 7 min 28 s |
| Mycelium | `only` | passed | 19, 13 | 13 min 4 s |
| MyceliumLearning | default | passed | 6, 6 | 5 min 33 s |
| MyceliumLearning | `only` | passed | 1, 3 (EM) | 6 min 44 s |
| Lenticulum | slow (`only`) | passed | 35, 41; path A 2, 2; path C 1, 3 | 39 min 41 s |
| Lenticulum | fast (default) | passed, before the S2 and `unstable` changes of the helper | 45, 21 | 34 min 18 s |
| Lenticulum `ad_matrix.jl` alone | default rows, after those changes | passed | 45, 21 | 5 min 44 s |

Lenticulum's fast tier ran in Codex's worktree with the earlier helper; its default matrix rows were run again alone
with the final helper (in a scratch environment with the five packages and the test dependencies), not the whole
fast tier. All other runs are of the final change. Logs: `~/.cache/lenticulum/q5/main/` and `q5/gate/`.

## Not checked

Reactant (out of scope: the inference core mutates its stores). The device tests (`LENTICULUM_DEVICE_TESTS`) were not
run. Enzyme forward mode was tried on two rows only. Second derivatives are out of scope. The new `ad-matrix` job of
`Slow.yml` has not run on CI.
