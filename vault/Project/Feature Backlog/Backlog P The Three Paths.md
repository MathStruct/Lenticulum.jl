#overview

> Section P of the [[Feature Backlog]]: what the refactor review of 2026-11-10 asked for on the three paths; it comes before every other section. Open rows only; a row that is
> done moves to [[Backlog Done 2026-11]] with its id. Status, priority, ids and the column *needs*
> are explained in [[Feature Backlog]], which also holds the dependency graph.

> Sources: original to this repository; the sources of each row are in its *source* column.

After the refactor ([[Refactor 2026-11]]) work on the paths comes before every other section of the
[[Feature Backlog]]. The rows here are the ones that review asked for; rows of other sections that
serve the same end are named. The structure-search driver (path C) is S7.

| id | feature | path | status | needs | source | why |
|---|---|---|---|---|---|---|
| P10 | `noisefree_nodes` against `field_nodes` on the calibration benchmark; a default for the nodes | B | proposed | F2 | [[Implicit Diffusion Learners]] §5 (2026-11-10) | nodes with noise have twice the smoothing bias at a level |
| P14 | A shared solver layer (root finding, the implicit-function adjoint, one tolerance rule, one report), shared by `VariationalDiffusion` and `ImplicitLayers` | B | proposed | — | [[Refactor 2026-11]] | three implementations with three conventions; `ImplicitLayers`' tolerances are absolute and written for `Float64`. Its first fix, now that it is back in `lib/` |
| P16 | Owned scalarisations: one graded energy, a scalarisation chosen per parameter key (each block descends its own), for games such as a GAN; with a stability check of the joint Jacobian | — | proposed | — | [[GANs as Two Factors]] §4 (2026-11-10) | every learnable node descends the one scalar $F$, so no game is expressible; a per-factor sign is the zero-sum special case |
| P35 | `examples/proxdm_backprop_noise.jl` (VariationalDiffusion) fails at full size: `MixtureProx exceeded maxiters=100 without a local minimiser` in its five-training-set part. Decide whether `MixtureProx`'s solver or the example's settings are at fault, and fix that | B | proposed | — | P15 (2026-11-19): found by Codex comparing full-size runs, before and after the smoke switch, and reproduced by the session with one `proximal` call (`~/.cache/lenticulum/p15/prox-repro.jl`: 64 means, `s = 0.05`, `tol = 1e-12`, `t = 0.005`, `λ = 0.01`) | the example's results in `proxdm.md` cannot be regenerated; the smoke run (4 means) does not reach it |
| P36 | Close the gaps of the first coverage report: files no test of their own package reaches (`Mycelium/src/exposure.jl` 0 %, `MyceliumLearning/src/sample_gradient.jl` 0 %, Mycelium's ForwardDiff extensions 0 %, `LenticulumCore/src/statistical_game.jl` 35 %, `LenticulumCore`'s ExponentialFamily extension 0 %, `VariationalDiffusion`'s Reactant extension 0 %). For each: tested downstream only (then move or add a test in the owning package), or untested (then a test against a closed form, or a note that it is unsupported) | all | proposed | — | P15 (2026-11-19): `tools/coverage.jl`, report in `~/.cache/lenticulum/p15/coverage/` (`summary.md`, `uncovered.md`) | a package's own suite should exercise its own code; a change to Mycelium is not caught by Mycelium's suite there |
