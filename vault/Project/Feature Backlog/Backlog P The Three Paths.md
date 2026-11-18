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
| P12 | `density_lambda` as the default for non-point messages | B | partial: a warning (2026-11-10) | — | LP-024 | uncalibrated covariances reach the graph silently |
| P14 | A shared solver layer (root finding, the implicit-function adjoint, one tolerance rule, one report), shared by `VariationalDiffusion` and `ImplicitLayers` | B | proposed | — | [[Refactor 2026-11]] | three implementations with three conventions; `ImplicitLayers`' tolerances are absolute and written for `Float64`. Its first fix, now that it is back in `lib/` |
| P15 | A coverage report; a job that runs the examples and benchmarks at small size | all | proposed | — | review of 2026-11-09 | which branches no test reaches is unknown; examples and benchmarks are run by nothing (`Aqua.jl` in every suite: done 2026-11-10) |
| P16 | Owned scalarisations: one graded energy, a scalarisation chosen per parameter key (each block descends its own), for games such as a GAN; with a stability check of the joint Jacobian | — | proposed | — | [[GANs as Two Factors]] §4 (2026-11-10) | every learnable node descends the one scalar $F$, so no game is expressible; a per-factor sign is the zero-sum special case |
| P30 | Lenticulum's linear-Gaussian factors in the element type of their parameters: messages and energies of `GaussianFactor`, `GaussianPrior` and the constraint factors in Float32 for Float32 parameters, noise and data, with a test against the Float64 graph | A | proposed | — | P8 (2026-11-18) | measured: a tree of `GaussianPrior` and `GaussianFactor`s with Float32 parameters and noise sends `ExtendedGaussian{Float64}` and `GaussianBelief{…Float64…}` messages, so its free energy is Float64 although Mycelium's sum no longer widens; seen in `gaussian.jl` (`Matrix{Float64}(I, …)`, `zeros(…)` beliefs, `_RANK_MEMO` for `Float64` only), and the review of P8 (Codex) found the `GaussianPrior` energy's Float64 normaliser (`complexity` is Float64 for Float32 data; `gaussian.jl`, the per-node and batched prior and `GaussianFactor`); the rest not searched |
