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
| P14 | A shared solver layer (root finding, the implicit-function adjoint, one tolerance rule, one report). Design in [[Shared Solver Layer]] (2026-11-22). W1 done and reviewed ([[LP-078 Findings of the reviews of the Mycelium coverage tests and the Anderson and block AD fixes|LP-078]]). W3 done (2026-11-23, VariationalDiffusion onto it; reviewed by Mistral, [[LP-079 Findings of the reviews of P14 W2 and W3|LP-079]]). W2 done (2026-11-23, ImplicitLayers onto it; reviewed by Mistral, LP-079). W4 (MyceliumLearning's graph adjoint) and W5 (Krylov.jl out) open. Newton and the trust region are in phase 2. | B | partial | — | [[Refactor 2026-11]] | three implementations with three conventions; `ImplicitLayers`' tolerances are absolute and written for `Float64`. Its first fix, now that it is back in `lib/` |
| P16 | Owned scalarisations: one graded energy, a scalarisation chosen per parameter key (each block descends its own), for games such as a GAN; with a stability check of the joint Jacobian | — | proposed | — | [[GANs as Two Factors]] §4 (2026-11-10) | every learnable node descends the one scalar $F$, so no game is expressible; a per-factor sign is the zero-sum special case |
