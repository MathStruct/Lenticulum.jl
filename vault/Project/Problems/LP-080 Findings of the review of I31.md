#problem

> **Status: closed (2026-11-23).** I31 (the residual schedule keeps its priority queue between sweeps,
> Codex) and what its review found. Codex's first run stopped at the determinism gate, rightly: RED-Diff
> draws from an RNG in its prox config, so a repeated probe differs. Decided: a trait
> `repeatable_messages(f)`, `false` by default; reuse only on graphs whose factors all declare it.
> Mistral's probes found no defect; one documented limit is confirmed (contents changed in place are not
> detected). Related: [[LP-072 Findings of the review of I4|LP-072]] (the queue), [[Feature Backlog]] (I31).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; briefs
> `~/.cache/lenticulum/codex-queue/heap-brief.md`, `heap2-brief.md`; answers
> `../Lenticulum.jl-Review-2026-11-22/residual-heap/IMPL-heap-run1.md`, `IMPL-heap-run2-partial.md`, the
> second written before its usage limit stopped it during the Lenticulum suite). Mistral Large
> (`vibe --agent review`, \$0): it reached its turn limit without writing its answer; its nine probes in
> `scratch/heap/` of `../Lenticulum.jl-mistral` were run by the maintainer's session, output in
> `~/.cache/lenticulum/heap-review/`. code: `lib/Mycelium.jl/src/passing.jl` (`sweep!` for
> `ResidualSchedule`, `_take_residual_memory!`, `_external_residual_writes!`, `_max_pending`),
> `lib/Mycelium.jl/src/schedules.jl`. tests: `lib/Lenticulum.jl/test/residual_queue.jl`.

## The determinism gate (Codex's first run)

`DiffusionFactor`'s default prox `REDDiff()` (`lib/VariationalDiffusion.jl/src/factor.jl:45`) draws a time
and noise from `cfg.rng` in `regulariser_gradient` (`lib/VariationalDiffusion.jl/src/reddiff.jl:90–101`),
an RNG held in the prox, not in `st`. Two probes of one task with identical inputs gave residuals
0.002518506159659007 and 0.0027546060218798796 (checked in the source; Codex's script
`scratch/heap/probe-reddiff.jl` in `../Lenticulum.jl-codex`). Hence the trait, declared `true` for
Mycelium's `DataFactor`, `PriorFactor`, `RelayFactor`, `TableFactor`, Lenticulum's linear-Gaussian,
constraint, switching, nonlinear, likelihood and regime factors, `GaussianComposite`, `SplitFactor`,
`ScaleShiftFactor` and the exposed Gaussian; `ClusterFactor` and `OverlayFactor` follow their members.

## Mistral's probes

| # | probe | result |
|---|---|---|
| 1 | in-place change of a `ps` array between sweeps (`probe10_inplace.jl`) | commits equal, **stores differ** from `reuse = false`: the documented limit (the docstring: use `reuse = false` when changing factor data, parameters, state or stored beliefs in place). Kept: the library changes nothing in place between sweeps (`pin!` is detected; `learn!` and EM build new parameter trees) |
| 2 | a contradictory cavity created by a commit (`probe2_cavity.jl`) | both modes throw at the same sweep with the same commits |
| 3 | a factor that throws mid-sweep (`probe3_throw.jl`) | the same throw, then equal records, stores and states; the snapshot is cleared before the heap is touched |
| 4 | garbage collection of a store, `reset!`, `copyto!`, `deepcopy` (`probe4_gc.jl`) | the memory entry goes with the store; equal records. (Its "should be 12" label is wrong: it counts one factor's calls) |
| 5 | one schedule on two stores and two graphs alternately (`probe5_stores.jl`) | equal records; one entry per store |
| 6 | `_max_pending` against the scan, damping 0 and 0.3, `propagate!` with `:marginals` (`probe6_pending.jl`) | equal sweeps, pending residuals and reports |
| 7 | duals and floats alternately on one store (`probe7_duals.jl`) | equal records; each switch re-probes |
| 8 | two tasks sweeping different stores with one schedule at once (`probe8_threads.jl`) | no errors, records equal to the controls |
| 9 | a graph with one RNG factor; cluster and overlay delegation (`probe9_refusal.jl`) | reuse refused for the whole graph; equal records |

## Gates (2026-11-23)

On `dev-danielb` with I31 merged (in `../Lenticulum.jl-mistral`): `LENTICULUM_SLOW_TESTS=true julia -t auto
--project=lib/Lenticulum.jl -e 'using Pkg; Pkg.test()'` and VariationalDiffusion's suite, each ending with
`Testing … tests passed`. Codex's run on its branch: Mycelium,
MyceliumLearning and VariationalDiffusion passed; the oracle 568/568, and 160 of the 361 retained-queue
checks failed with the snapshot checks removed. Measurements (Codex, `bench/residual_queue.jl`, one BLAS
thread, suites running beside it): to a message residual of 1e-10 on the grids, 40.5–43.0 % fewer probes
and 28.6–29.9 % fewer message evaluations in all. Not run: the device modes of Mycelium, the examples'
smoke runner beyond the benchmark, the docs build.
