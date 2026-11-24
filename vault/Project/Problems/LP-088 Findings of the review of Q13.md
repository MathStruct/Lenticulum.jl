#problem

> **Status: closed (2026-11-24).** Q13 (a benchmark gate for pull requests; Codex) and what its review found.
> Mistral found one defect: a benchmark case with fixed `evals` had its graph's caches (polarities, emit plans)
> filled by its first timed sample, because `tune!` does nothing for such a case and `run` does not warm up. It
> was right and is fixed (`BenchmarkTools.warmup` after `tune!` in `benchmark/worker.jl`), with a check. Its second
> point was an artefact of the session's diff, not of Q13. Related: [[Engineering Standard]] (C5),
> [[Backlog Q Quality and Engineering]] (Q13), [[Feature Backlog]].

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; brief and answer in
> `../Lenticulum.jl-Review-2026-11-24/q13/`, logs in `scratch/q13/` of `../Lenticulum.jl-codex`). Mistral Large
> (`vibe --agent review`, seven probes, a finished answer file, \$0: `../Lenticulum.jl-Review-2026-11-24/q13/ANSWER-mistral.md`,
> probes beside it in `probes/`). code: `benchmark/worker.jl`, `benchmark/report.jl`, `benchmark/compare.jl`,
> `benchmark/fixtures.jl`, `lib/*/benchmark/benchmarks.jl`, `.github/workflows/Benchmarks.yml`. check:
> `benchmark/warmup_check.jl` (run by `benchmark/smoke.jl`).

## What Q13 does

`lib/<Pkg>/benchmark/benchmarks.jl` defines a BenchmarkTools `SUITE` for each of ImplicitSolvers, LenticulumCore,
Mycelium, MyceliumLearning, Lenticulum and VariationalDiffusion: 31 warm cases on paths A, B and C, and 10
first-call latency keys, each timed in a fresh process. `benchmark/compare.jl` checks out each revision in a
temporary worktree and runs the head's suite definitions against that revision's library code. It writes a
Markdown report with the flags of `benchmark/thresholds.toml` (the standard's values). With `--aa` it writes the
per-key noise. `Benchmarks.yml` runs it on pull requests to `dev` and `master`. The flags are advisory. See
`benchmark/README.md`.

## Mistral's review

| # | finding | right? | done |
|---|---|---|---|
| F1 | the first sample of a case measures cache construction: `cached_fixture` shares the built `FactorGraph` across samples, and its caches (`polarities`, `emit_plans`, `energy_rows`) are filled during the first sample (`Mycelium/kalman/tree_infer`: 0 → 481 entries, 1.49 ms against a steady 1.22 ms) | right, but not as the probe ran it: the probe called `run` without `tune!`. Repeated with the worker's own sequence (`../Lenticulum.jl-Review-2026-11-24/q13/probes/session_tune.jl`): `tune!` returns early for a case with fixed `evals`, `run` does not warm up, and the caches are still empty after `tune!` (first sample 1.32 ms, median 1.16 ms). With 1,000 samples the median hides one slow sample; in smoke mode it is one of four | `benchmark/worker.jl` calls `BenchmarkTools.warmup` after `tune!`; the caches are now filled before the first sample (481/481 after `tune!` and `warmup`). `benchmark/warmup_check.jl` repeats the sequence and asserts that the worker warms up; without the fix it fails. All-package smoke A/A after the fix: 31 warm and 10 latency keys, no failed suite (`../Lenticulum.jl-codex/scratch/q13/smoke-fix/report.md`) |
| F2 | the diff reverts the Q1 acceptance in the vault | not a defect of Q13: the session made the review diff against `dev-danielb`, which had the Q1 commit after the branch point. The merge applies the branch's diff against its own base | — |
| — | suspicions not shown by a probe: a revision whose `run.jl` fails as a whole aborts the comparison with no report; `worker.jl` assumes keys of depth three | the first is the specified behaviour (a tooling error exits nonzero); the second is the suites' convention (`Package/group/case`), stated in `benchmark/README.md` | — |
| — | checked clean: the flag rules (BenchmarkTools' `time_tolerance` is relative to the base; boundaries probed), allocation and latency rules, keys when a base lacks a package or a case, `--interleave`, `--also-base`, exit codes, latency timing only the first call, every case of the six suites runs, the provenance hook, the workflow's choice of base (read only) | — | — |

## Found by the session

- MyceliumLearning's warm gradients ran on a one-step chain (four factors). They now run on three steps
  (1.4 ms ForwardDiff, 4.8 ms Zygote, 6.3 ms implicit `:dense`, 26 ms implicit `:gmres`; 22–386 samples). At eight
  steps the run takes 216 ms for `:gmres`, so the size stays small.
- The implicit graph adjoint with `linear_solver = :gmres` under ForwardDiff grows faster than the chain: 4.6,
  26 and 216 ms at 1, 3 and 8 steps, with allocations of 48 k, 401 k and 4.1 M. `:dense` takes 1.7, 6.3 and 31 ms.
  Not examined; recorded as open in [[Status]].

## Gates

A full A/A on this machine (Codex, Julia 1.13.1, one thread, before the fix): 519 and 511 s per revision. The
median |log ratio| is 0.021 and the maximum 0.143; six keys are above 0.05, the largest
`Mycelium/belief/gaussian_product` (a sub-microsecond case) and `Mycelium/kalman/flooding_infer`
(`../Lenticulum.jl-Review-2026-11-24/q13/aa/report.md`). Against `dev-danielb~5`, before P41, both `KrylovJL`
cases are reported as failed on the base and the rest is compared. `bench/smoke.jl` passed 39/39 (Codex, with
provenance in every study). `benchmark/smoke.jl` passed (the session, after the fix:
`~/.cache/lenticulum/q13/benchmark-smoke.log`).

Not run: the GitHub workflow (it needs a push; the noise on CI runners is measured by its `aa` dispatch, and
until then the thresholds stay *to calibrate*); a full, non-smoke A/A after the fix; the packages' suites
(no library code changed).
