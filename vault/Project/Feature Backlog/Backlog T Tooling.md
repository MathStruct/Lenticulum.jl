#overview

> Section T of the [[Feature Backlog]]: benchmarks for optimiser design and regression tracking. Open rows only; a row that is
> done moves to [[Backlog Done 2026-11]] with its id. Status, priority, ids and the column *needs*
> are explained in [[Feature Backlog]], which also holds the dependency graph.

> Sources: original to this repository; the sources of each row are in its *source* column.

| id | feature | status | needs | priority | source | why |
|---|---|---|---|---|---|---|
| T1 | A benchmark of inference and learning problems for optimiser design: problems with exact or reference answers where today's methods are slow or fail, run by one script that records sweeps, wall time, error against the reference, the free energy reached, and failures. Candidates: the loopy divider (flooding: 326 sweeps to an error of 1e-10, and 97 224 with equal resistors; `bench/bp_spectrum.jl`, `bench/bp_schedules.jl`), a small discrete Ising grid (exact by junction tree), Gaussian grids near the walk-summability limit (measured 2026-11-09 for schedules and Anderson, `bench/bp_grid.jl`), the switch learning of `lib/Lenticulum.jl/examples/structure_clusters.jl` (EM with SQUAREM, restarts, beam), the calibration benchmark, MNIST in both directions. Methods and compositions to compare: damping, SQUAREM, Anderson acceleration, heavy ball, mirror and accelerated mirror descent, Langevin, learned optimisers ([[The Regulariser and the Loss]] §3); the composition laws and an order of experiments are in [[Composing Optimisers]] | proposed | — | medium | Daniel, 2026-11-07 ("needs a serious learning problem to test various methods or compositions") | every optimiser idea so far was tried on one graph or one example; a fixed set of problems makes them comparable and keeps a method that wins on one from being adopted on that alone |
| T2 | Benchmark regression tracking (scaling, allocations per message) | proposed | — (P15 done: `bench/smoke.jl` runs every benchmark small) | low | ChatGPT review §6 | |
