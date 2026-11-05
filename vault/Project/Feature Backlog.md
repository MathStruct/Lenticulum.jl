#overview

> The features Lenticulum.jl should get, in one list: what each is for, where it was asked for,
> what measurement motivates it, and how far it is. A feature is added here when a
> specification, a problem, a review or a measurement asks for it, and moved to "done" with the
> commit that implemented it. Priorities are proposals until Daniel sets them.

> Sources: original to this repository; the open lists of [[Specifications]] and [[Problem Log]], [[LP-023 Findings of the external reviews of 2026-11-04]].

Status: *planned* (specified or asked for, not started), *proposed* (an idea with a reason, no
specification), *partial*, *done*. Priority: **high** (it limits something measured now),
**medium**, **low**.

## Inference and graph execution

| feature | status | priority | source | why |
|---|---|---|---|---|
| Compilation per structure in the free-energy and EM paths | partial: message loop (2026-11-04), free energy, cluster code and EM outside AD (2026-11-05) | medium | `passing.md` "Compilation per structure" | a new structure still compiles 1.14 s for F, `em_step` and a fit (0.31 s warm); the AD paths and ForwardDiff's chunks remain |
| Typed execution plans: factors grouped by type and rule, kernels compiled once per signature | proposed | medium | both reviews (LP-023) | the graph stores factors abstractly; plates are the first step |
| A dependency-aware priority queue for `ResidualSchedule` | proposed | medium | ChatGPT review (LP-023) | each sweep probes every task, $O(T^2)$ evaluations |
| Graph-wide parameters by integer slot instead of `NamedTuple` lookup by symbol | proposed | medium | ChatGPT review, [[Parallelism and Compilation]] | quadratic scaling with the number of keys |
| Batched cluster nodes with per-outcome forms over a plate | planned | medium | [[L-SPEC-009 Cluster Nodes]] open | `RegimeFactor` plates go replica by replica |
| Mixture-reduction and EP nodes batched in plates | planned | low | [[L-SPEC-008 Expectation Propagation and Mixture Reduction]] open | edge by edge today |
| Clusters with loops; several mean-field variables in one cluster | planned | low | [[L-SPEC-007 Hybrid Schedules]] open | |
| Nested plates; stateful tied factors | planned | low | [[L-SPEC-006 Plates]] open | |
| Incremental inference (reuse factorisations and beliefs after a local change, iSAM2-like) | proposed | low | ChatGPT review | online learning, SLAM-like use |
| A lazy product-of-potentials belief with explicit approximation policies | proposed | low | ChatGPT review | learned potentials without silent Gaussian projection |

## Structure search and editing

| feature | status | priority | source | why |
|---|---|---|---|---|
| Template edits of plated graphs, stable ids | planned | low | [[L-SPEC-011 Plate Edits]] | edits and `as_plate` take about 15% of a search today |
| Moves and warm starts as library code (beam search, final refit) | proposed | low | `examples/structure_clusters.md` | the example's beam with refit reaches the best fit in 62% of the gradient search's time; a reusable driver would serve other families |
| Collapsed M-steps as a library feature (integrate out the intermediates of a switch-over-sum) | proposed | medium | `examples/structure_clusters.md` | the per-factor M-step crawls with nearly deterministic switches |

## Learning

| feature | status | priority | source | why |
|---|---|---|---|---|
| The warning of LP-022 for data that arrive other than by `DataFactor` (pins, point messages) | proposed | low | [[LP-022 Constraint factors are normalised in their residual, not their channels]] | |
| MAP M-steps for noise priors (Wishart, log-normal); M-steps for further families | proposed | low | `src/m_steps.md` §4 | `GaussianFactor` and `GaussianPrior` have ML M-steps since 2026-11-05; with a prior their noise is not updated |
| Stop-gradient semantics per wire, factor and parameter subset | planned | medium | [[L-SPEC-002 Stop Gradient]] | |
| Exposed parameters: generic approximate `parameter_message`, energy taps, risk scalarisations | partial | low | [[L-SPEC-003 Exposed Parameters and Energy Channels]] §2–§3 | |
| Explicit gradient semantics in results (envelope, finite replay, implicit fixed point) | proposed | low | ChatGPT review §4.2 | a caller should see which objective was differentiated |
| Calling `validate_parameters` automatically before inference | proposed | low | LP-023 | it exists (2026-11-04) but is opt-in |

## Utility factors and data (asked for 2026-11-05)

Daniel: "there should be a whole bunch of utility factors that are still lacking." Mycelium has
`DataFactor`, `PriorFactor`, `LossFactor`, `RelayFactor` (an equality between two variables),
`OptimiserFactor`, `TableFactor` and `StraightThroughFactor`. Today the operations below are
written by hand, from `LinearConstraintFactor`s with selection matrices or from graph code.

| feature | status | priority | source | why |
|---|---|---|---|---|
| Data loading from `MLUtils.jl` (`DataLoader`): minibatches fed into a plate's `DataFactor`s, shuffling, epochs, as an optional extension | proposed | medium | Daniel, 2026-11-05 | every example builds one `DataFactor` per point by hand and keeps all data in the graph; minibatch training needs the plate's data swapped per batch |
| A split / concatenate factor: one channel of dimension $d_1 + d_2$ related exactly to two channels of $d_1$ and $d_2$ (and more pieces, any index sets), with exact messages in every direction | proposed | medium | Daniel, 2026-11-05 | today a selection-matrix `LinearConstraintFactor` with tiny noise, which is neither exact nor cheap |
| Mirrored (overlaid) factors: two or more *different* factors acting on exactly the same variables, each with its own parameters (Daniel, 2026-11-05: "two different factors overlayed and acting on the same variables. They should have different parameters"), declared once instead of wiring each by hand | proposed | medium | Daniel, 2026-11-05 | a product of experts on one set of variables; open: whether the overlay is several nodes on shared variables (exact on a tree, a loop of length two otherwise: two factors on the same two variables form a cycle for belief propagation) or one node whose messages are the product of its members' (exact, like a cluster of the members) |
| Further utility factors: a deterministic map (a function as an exact relation, `FunctionFactor`), a permutation or reshape of a channel, a sum or average of several channels, a constant (clamp) factor with its own schedule | proposed | low | Gemini review §4 (a `FunctionFactor`), Daniel, 2026-11-05 | the common glue of a model, written by hand each time |

## Families

| feature | status | priority | source | why |
|---|---|---|---|---|
| Calibration benchmarks for implicit relations | partial: circle, lemniscate, unbalanced rings (2026-11-05, `lib/VariationalDiffusion.jl/examples/calibration.md`); a pitchfork and a redundant arm remain | medium | both reviews (LP-023) | baseline measured: every off-domain query returns a root, the lemniscate's recall is at best 0.52, the 10% ring is never found |
| Calibration remedies: off-relation detection (the full score's norm, an energy threshold, a round trip), annealing the field levels, ridge residuals | proposed | medium | both reviews; `calibration.md` | measured against the benchmark above |
| Heteroscedastic and heavy-tailed likelihoods | proposed | low | ChatGPT review §5 | |
| Polynomial factor: higher-degree ideal quotients, missing-data EM | planned | low | `polyfit.md` | |
| Algebraic factors: MLE existence by LP, Markov bases, splitting large tables | planned | low | [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] open, §2.4 | |
| Polarity-dependent energies (allowed by the interface, used by no factor) | proposed | low | [[L-SPEC-001 The Factor Interface]] open | |

## Tooling

| feature | status | priority | source | why |
|---|---|---|---|---|
| Benchmark regression tracking (scaling, allocations per message) | proposed | low | ChatGPT review §6 | |

## Done (2026-11-04/05)

CI with a second Julia version and the device tests (2026-11-05). The matrix runs the current
release and 1.12, the oldest supported. Two separate jobs run the Mycelium and ImplicitLayers device
files (JLArrays). The supported minimum was declared as 1.10 but never tested, and it was wrong:
the packages find each other through `[sources]`, which needs 1.11, so on 1.10 every package but
LenticulumCore fails to instantiate. On 1.11 the Enzyme backend of the DEQ solve fails
(`MixedReturnException`, Enzyme 0.13.205); the same Enzyme passes on 1.12. All seven suites
pass on 1.12, and every `Project.toml` now says `julia = "1.12"`.

A warning for `DEQFactor`s trained by their residual with the state observed (2026-11-05; measured
in `lib/ImplicitLayers.jl/test/lp022.jl`). Cluster forms and closed-form M-steps for
`GaussianFactor` and `GaussianPrior`, learned noise included (2026-11-05; `test/em.jl`).

Cluster nodes with per-outcome coefficients and noise (`RegimeFactor`); flat directions in
clusters; `output` channels and the LP-022 warning (constraint and toric factors); fixes from the
external reviews (LP-023); frozen paths inside cluster nodes; closed-form M-steps and
SQUAREM-accelerated EM (`fit_em`), batched over plates; the message loop no longer specialised on
the parameter tree; `validate_parameters`; beam search and final refit in
`examples/structure_clusters.jl`.
