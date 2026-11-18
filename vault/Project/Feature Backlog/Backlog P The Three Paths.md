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
| P33 | What P32 left in Float64: the shared batch path of cluster plates (`lib/Mycelium.jl/src/cluster_batch.jl`, Float64 storage; Float32 plates go replica by replica, correct but slower), accelerated EM (`fit_em(...; accelerate = true)`, an empty flattened subtree is `Float64[]` in `lib/LenticulumCore.jl/src/parameter_tree.jl`), unweighted sample reweighting (`lib/Mycelium.jl/src/messages.jl`, Float64 zeros) | A, C | proposed | — | P32 (2026-11-18), [[LP-061 Findings of the review of P32]] | measured: 2 `@test_broken` (accelerated EM) in `lib/MyceliumLearning.jl/test/element_type.jl`; the batch path by source reading (`cluster.md`, "Element type") |
