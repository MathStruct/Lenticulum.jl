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
| P3 | A `TrivialBelief()` input (no message) to `GaussianFactor` takes the exact flat path, as an everywhere-flat `ExtendedGaussian` does | A | proposed | — | [[lib/Lenticulum.jl/src/gaussian\|gaussian]] "Extended Gaussian inputs" (2026-11-11) | with $n_{\text{in}} < n_{\text{out}}$ the shortcut returns no message where the exact one fixes $y$ off $\operatorname{range}(A)$; changes the first sweep of every schedule, so measure it on the existing tests first |
| P5 | `HardConstraintFactor` with a singular nonzero noise in the free energy (hard along its null space, soft along its range); learnable (its term needs an SVD, which ForwardDiff lacks) | A, C | proposed | — | [[hard_constraint]] "Free energy" (2026-11-11) | refused today |
| P8 | Element type in, element type out: the graph free energy in the element type of its messages, with a test | A | partial: LenticulumCore (2026-11-10) | — | LP-024 | `Float32` graphs are promoted to `Float64` in the total |
| P9 | A relative convergence criterion for Gaussian messages | A | proposed | — | LP-024 | the bound on natural parameters is absolute |
| P10 | `noisefree_nodes` against `field_nodes` on the calibration benchmark; a default for the nodes | B | proposed | F2 | [[Implicit Diffusion Learners]] §5 (2026-11-10) | nodes with noise have twice the smoothing bias at a level |
| P12 | `density_lambda` as the default for non-point messages | B | partial: a warning (2026-11-10) | — | LP-024 | uncalibrated covariances reach the graph silently |
| P14 | A shared solver layer (root finding, the implicit-function adjoint, one tolerance rule, one report), shared by `VariationalDiffusion` and `ImplicitLayers` | B | proposed | — | [[Refactor 2026-11]] | three implementations with three conventions; `ImplicitLayers`' tolerances are absolute and written for `Float64`. Its first fix, now that it is back in `lib/` |
| P15 | A coverage report; a job that runs the examples and benchmarks at small size | all | proposed | — | review of 2026-11-09 | which branches no test reaches is unknown; examples and benchmarks are run by nothing (`Aqua.jl` in every suite: done 2026-11-10) |
| P16 | Owned scalarisations: one graded energy, a scalarisation chosen per parameter key (each block descends its own), for games such as a GAN; with a stability check of the joint Jacobian | — | proposed | — | [[GANs as Two Factors]] §4 (2026-11-10) | every learnable node descends the one scalar $F$, so no game is expressible; a per-factor sign is the zero-sum special case |
| P27 | A Gaussian mixture times an extended Gaussian in Mycelium's product (`_mixture_product`), for a `SwitchingConstraintFactor` (mixture or EP rule) on a variable a hard relation reaches | A, C | proposed | — | P4 probe (2026-11-16), `test/extended_plates.jl` (broken) | `ArgumentError: no product rule for mixture components`, in plates and single graphs |
