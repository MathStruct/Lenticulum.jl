#open-problem

> **Status: resolved (2026-11-19).** The review of P15 (every benchmark and example at small size,
> `bench/smoke.jl`; a coverage report, `tools/coverage.jl`; their CI jobs) found no defect. Its eleven
> probes were all run; two were recorded in its answer file, the others' results were read by the session.
> Found on the way, before the review and outside it: `proxdm_backprop_noise.jl` fails at full size (row P35).
> Related: [[Feature Backlog]] (P15, P35), [[Backlog Done 2026-11]].

> Sources: the switch in the scripts implemented by Codex (GPT, two runs in `../Lenticulum.jl-codex`); the
> runner, the coverage tool and the workflows by the maintainer's session (Claude); reviewed by Mistral Large
> (`vibe`, review agent, \$0, 100 turns). Briefs, diff, answers and probes: `../Lenticulum.jl-Review-2026-11-19/p15/`.
> code: `bench/smoke.jl`, `tools/coverage.jl`, `.github/workflows/Examples.yml`, `.github/workflows/Weekly.yml`
> (job `coverage`), the `SMOKE` lines of `bench/*.jl` and of the examples of Lenticulum and VariationalDiffusion.

**Problem.** The requirement given to the reviewer: unset, `LENTICULUM_SMOKE` leaves every script exactly
as it was; set, each runs small through every method, variant and table row; the runner records `ok` only
for exit 0 within the timeout and exits 1 otherwise; the coverage tool reports covered and coverable lines
from the package's `.cov` files, refuses when there are none, and deletes them unless `--keep`; the
workflows work from a fresh checkout.

| # | probe | result |
|---|---|---|
| 1 | `@isdefined(SMOKE) \|\| const SMOKE = …` twice and in a module; `$flags` (a `Cmd`) splices into the command | passes (probe 1, probe 10: the two examples the tests include see `SMOKE = false` when it is unset) |
| 2 | the real runner on `ring_bias`: summary, log, exit code | passes (probe 2) |
| 3 | the runner over a script that exits 3, one that hangs (`SMOKE_TIMEOUT = 8`), one that passes; name selection; an unknown name | passes: `exit 3`, `timeout` after 13 s, `ok`; runner exit 1 (probe 3) |
| 4 | every one of the 30 scripts reads the switch; helpers, `gpu_energy.jl` and `lp002_*` are skipped | passes (probe 9) |
| 5 | `ring_bias.jl` full size, old against new | identical output (probe 7b) |
| 6 | table rows of the smoke runs against the full runs | rows present (probe 8); its "2 of 6" for `product_sampling` was its own count: the smoke log has all six sampler rows |
| 7 | `coverage.jl`: `line_ranges`, `.cov` files of two processes, an `ext/` folder, `--keep` | passes on a synthetic package (probes 5, 6) |
| 8 | `coverage.jl --run lib/LenticulumCore.jl` end to end | written, run at the turn limit, no result recorded (probe 11); the session ran the same command: 71.8 % |

Before the review, Codex compared the full-size output of 18 scripts before and after the change (same up to
timings; the others take longer than two minutes and were not compared), and the session checked by reading
that every changed value keeps its old value when the switch is unset (`implicit_sample`'s default `steps` is
the 100 now passed explicitly by `product_sampling.jl`).

Not checked: the workflows on GitHub (they run on the schedule or on demand; neither has run yet), the full-size
runs of the slow scripts.
