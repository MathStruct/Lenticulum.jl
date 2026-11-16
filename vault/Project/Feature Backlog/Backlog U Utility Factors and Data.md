#overview

> Section U of the [[Feature Backlog]]: data loading and the small exact factors written by hand today (asked for 2026-11-05). Open rows only; a row that is
> done moves to [[Backlog Done 2026-11]] with its id. Status, priority, ids and the column *needs*
> are explained in [[Feature Backlog]], which also holds the dependency graph.

> Sources: original to this repository; the sources of each row are in its *source* column.

Daniel: "there should be a whole bunch of utility factors that are still lacking." Mycelium has
`DataFactor`, `PriorFactor`, `LossFactor`, `RelayFactor` (an equality between two variables),
`OptimiserFactor`, `TableFactor` and `StraightThroughFactor`. Today the operations below are
written by hand, from `LinearConstraintFactor`s with selection matrices or from graph code.

| id | feature | status | needs | priority | source | why |
|---|---|---|---|---|---|---|
| U1 | Data loading from `MLUtils.jl` (`DataLoader`): minibatches fed into a plate's `DataFactor`s, shuffling, epochs, as an optional extension | proposed | — | medium | Daniel, 2026-11-05 | every example builds one `DataFactor` per point by hand and keeps all data in the graph; minibatch training needs the plate's data swapped per batch |
| U2 | A split / concatenate factor: one channel of dimension $d_1 + d_2$ related exactly to two channels of $d_1$ and $d_2$ (and more pieces, any index sets), with exact messages in every direction | proposed | — | medium | Daniel, 2026-11-05 | today a selection-matrix `LinearConstraintFactor` with tiny noise, which is neither exact nor cheap. Since 2026-11-11 a `HardConstraintFactor` with selection matrices imposes it exactly (not tried for this use); a dedicated factor would still be cheaper |
| U3 | A normalising change of variables: an affine factor $z = (x - b) \oslash s$ (per coordinate or per group) with exact messages both ways and the log-Jacobian $\sum_i \log\lvert s_i\rvert$ in its free energy; fixed $b$, $s$ from the data (standardising data channels) or learned with data-dependent initialisation (Glow's ActNorm) | proposed | — | medium | Daniel, 2026-11-05 (batchnorm and group norm in deep-learning libraries) | conditioning: the flat-direction threshold, eigenvalue floors and EM are relative to scale. Without the log-Jacobian a learned scale collapses like LP-022 (squash $x$, raise the density of $z$). Test: a chain with and without it differs by exactly $\sum \log\lvert s_i\rvert$ (mind the deterministic factor's Bethe bookkeeping, as for the relay in LP-019) |
| U4 | Shared plate statistics: a constructor for a plate whose replicas share mean and scale variables ($x_i \sim \mathcal N(\mu, \sigma^2)$), the probabilistic reading of batchnorm's batch statistics | proposed | — | low | Daniel, 2026-11-05 | expressible today (`lib/Lenticulum.jl/examples/hierarchical_models.jl`); the posterior over $\mu, \sigma$ replaces running averages at test time. Not batchnorm's forward pass, which couples every replica through a non-local factor and is no density |
| U5 | Further utility factors: a deterministic map (a function as an exact relation, `FunctionFactor`), a permutation or reshape of a channel, a sum or average of several channels, a constant (clamp) factor with its own schedule | proposed | — | low | Gemini review §4 (a `FunctionFactor`), Daniel, 2026-11-05 | the common glue of a model, written by hand each time |
