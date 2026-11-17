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
| P8 | Element type in, element type out: the graph free energy in the element type of its messages, with a test | A | partial: LenticulumCore (2026-11-10) | — | LP-024 | `Float32` graphs are promoted to `Float64` in the total |
| P9 | A relative convergence criterion for Gaussian messages | A | proposed | — | LP-024 | the bound on natural parameters is absolute |
| P10 | `noisefree_nodes` against `field_nodes` on the calibration benchmark; a default for the nodes | B | proposed | F2 | [[Implicit Diffusion Learners]] §5 (2026-11-10) | nodes with noise have twice the smoothing bias at a level |
| P12 | `density_lambda` as the default for non-point messages | B | partial: a warning (2026-11-10) | — | LP-024 | uncalibrated covariances reach the graph silently |
| P14 | A shared solver layer (root finding, the implicit-function adjoint, one tolerance rule, one report), shared by `VariationalDiffusion` and `ImplicitLayers` | B | proposed | — | [[Refactor 2026-11]] | three implementations with three conventions; `ImplicitLayers`' tolerances are absolute and written for `Float64`. Its first fix, now that it is back in `lib/` |
| P15 | A coverage report; a job that runs the examples and benchmarks at small size | all | proposed | — | review of 2026-11-09 | which branches no test reaches is unknown; examples and benchmarks are run by nothing (`Aqua.jl` in every suite: done 2026-11-10) |
| P16 | Owned scalarisations: one graded energy, a scalarisation chosen per parameter key (each block descends its own), for games such as a GAN; with a stability check of the joint Jacobian | — | proposed | — | [[GANs as Two Factors]] §4 (2026-11-10) | every learnable node descends the one scalar $F$, so no game is expressible; a per-factor sign is the zero-sum special case |
| P29 | A rank tolerance for a `NonlinearFactor`'s difference Jacobian, tied to its accuracy (about $10^{-10}$ relative), so the rank tests of its messages see what $f$ depends on | A | proposed | — | [[LP-049 Findings of the review of P28]], `test/p28_no_input.jl` (`@test_broken`) | with $\operatorname{rank}(\partial f/\partial x)$ below full, central differences have full numerical rank: the no-input forward message loses directions (conservative), and a backward message is not declared improper |
