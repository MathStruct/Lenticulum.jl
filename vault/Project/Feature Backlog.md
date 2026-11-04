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
| A guard for `DEQFactor` trained by its residual energy with the state observed | proposed | medium | [[Why Learned Relations Do Not Collapse]] | $g(z, x) = z$ satisfies every pair; nothing warns |
| The warning of LP-022 for data that arrive other than by `DataFactor` (pins, point messages) | proposed | low | [[LP-022 Constraint factors are normalised in their residual, not their channels]] | |
| Closed-form M-steps for `GaussianFactor`, `GaussianPrior` with learned noise, other families | proposed | medium | `src/m_steps.md` | EM covers only the constraint factors |
| Stop-gradient semantics per wire, factor and parameter subset | planned | medium | [[L-SPEC-002 Stop Gradient]] | |
| Exposed parameters: generic approximate `parameter_message`, energy taps, risk scalarisations | partial | low | [[L-SPEC-003 Exposed Parameters and Energy Channels]] §2–§3 | |
| Explicit gradient semantics in results (envelope, finite replay, implicit fixed point) | proposed | low | ChatGPT review §4.2 | a caller should see which objective was differentiated |
| Calling `validate_parameters` automatically before inference | proposed | low | LP-023 | it exists (2026-11-04) but is opt-in |

## Families

| feature | status | priority | source | why |
|---|---|---|---|---|
| Calibration benchmarks for implicit relations (spurious roots, branch recall; lemniscate, pitchfork, redundant arm, unbalanced mixtures) | proposed | medium | both reviews (LP-023) | only the circle example exists |
| Heteroscedastic and heavy-tailed likelihoods | proposed | low | ChatGPT review §5 | |
| Polynomial factor: higher-degree ideal quotients, missing-data EM | planned | low | `polyfit.md` | |
| Algebraic factors: MLE existence by LP, Markov bases, splitting large tables | planned | low | [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] open, §2.4 | |
| Polarity-dependent energies (allowed by the interface, used by no factor) | proposed | low | [[L-SPEC-001 The Factor Interface]] open | |

## Tooling

| feature | status | priority | source | why |
|---|---|---|---|---|
| CI with a second Julia version and the device tests | proposed | medium | ChatGPT review §6 | CI runs Julia 1 on Ubuntu without device tests |
| Benchmark regression tracking (scaling, allocations per message) | proposed | low | ChatGPT review §6 | |

## Done (2026-11-04)

Cluster nodes with per-outcome coefficients and noise (`RegimeFactor`); flat directions in
clusters; `output` channels and the LP-022 warning (constraint and toric factors); fixes from the
external reviews (LP-023); frozen paths inside cluster nodes; closed-form M-steps and
SQUAREM-accelerated EM (`fit_em`), batched over plates; the message loop no longer specialised on
the parameter tree; `validate_parameters`; beam search and final refit in
`examples/structure_clusters.jl`.
