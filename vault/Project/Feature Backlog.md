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
| A spanning-tree schedule for loopy graphs (`spanning_tree_schedule(g)`: the messages off a spanning tree, then the tree's inward and outward sweeps, in place; the updates of tree-based reparameterisation) | partial: done in Mycelium (2026-11-09; `schedules.md`); the choice of tree and alternating trees remain | high | [[Composing Optimisers]] §4.2 (2026-11-09) | measured on the divider (`bench/bp_schedules.jl`): 0.17–0.18 of flooding's message evaluations to an error of 1e-10, in all four settings, with no knowledge of the loop; with `infer!` 82 sweeps against 463 (`test/runtests.jl`). Graphs with several loops are untested |
| A dependency-aware priority queue for `ResidualSchedule` | proposed | medium | ChatGPT review (LP-023) | each sweep probes every task, $O(T^2)$ evaluations: 2.1 times flooding's on the divider. A queue prototype in `bench/bp_schedules.jl` needs 0.24–0.31 of flooding's, behind the spanning-tree order ([[Composing Optimisers]] §4.2) |
| Graph-wide parameters by integer slot instead of `NamedTuple` lookup by symbol | proposed | medium | ChatGPT review, [[Parallelism and Compilation]] | quadratic scaling with the number of keys |
| Batched cluster nodes with per-outcome forms over a plate | planned | medium | [[L-SPEC-009 Cluster Nodes]] open | `RegimeFactor` plates go replica by replica |
| Mixture-reduction and EP nodes batched in plates | planned | low | [[L-SPEC-008 Expectation Propagation and Mixture Reduction]] open | edge by edge today |
| Clusters with loops; several mean-field variables in one cluster | planned | low | [[L-SPEC-007 Hybrid Schedules]] open | |
| Nested plates; stateful tied factors | planned | low | [[L-SPEC-006 Plates]] open | |
| Incremental inference (reuse factorisations and beliefs after a local change, iSAM2-like) | proposed | low | ChatGPT review; LenticulumLM's [Incremental Next-Token Prediction](https://mathstruct.org/LenticulumLM/Design/Incremental-Next-Token-Prediction) (an idea, 2026-11-05) | online learning, SLAM-like use, and next-token prediction as appending factors; `transfer` gives warm starts today, not reuse of factorisations |
| A lazy product-of-potentials belief with explicit approximation policies | proposed | low | ChatGPT review | learned potentials without silent Gaussian projection |
| Underdamped Langevin dynamics on the Bethe free energy: beliefs $b$ (with a mass, e.g. Fisher) get a momentum $p$, with $\dot b = p$, $\dot p = -\nabla F(b) - \gamma p + \sqrt{2\gamma T}\,\xi$. At $T = 0$ the friction $\gamma$ brakes $H = F + \tfrac12\lVert p\rVert^2$ at every step ($\dot H = -\gamma\lVert p\rVert^2$), a Lyapunov function that a BP sweep lacks; with $T$ annealed to 0 it could also hop between fixed points. An idea, untested. To settle: must it descend $\nabla F$ directly (a BP sweep is not a gradient of $F$, so momentum on messages may amplify oscillation); how to stay on the constraints (normalised, consistent beliefs: mirror or natural coordinates); Gaussian Bethe energies can be unbounded below on loops (from memory, to check). Test: iterations and answers on the loopy graphs that oscillate today, against damped BP and the exact marginals. **Interface (checked 2026-11-06):** a schedule defined outside the package plugs in through `AbstractSchedule` and a `sweep!` method, and `infer!` runs it (scratch test on the divider graph: at $\beta = 0$ it reproduces flooding exactly; the sweep count it gave is not recorded, `bench/bp_spectrum.jl` measures 326 sweeps to an error of 1e-10). Missing, all additive: the gradient of $F$ in the beliefs (`free_energy_gradient` is for parameters), coordinates and a retraction for beliefs, and a state holding primal beliefs (off a fixed point the store's messages do not give consistent beliefs). Naive heavy ball on BP messages failed there for every $\beta \ge 0.3$ tried (negative precisions; with momentum on $\eta$ only, NaN) | proposed | low | Daniel, 2026-11-06 | loopy BP has no convergence guarantee and `damping` only sometimes cures oscillation ([[Loopy Message Passing]]); `Momentum` exists for parameters but nothing for beliefs |
| Accelerated mirror descent on the Bethe free energy: descend $F$ in the beliefs with the entropy (log-partition) as the mirror map, so every iterate is a valid belief by construction (for exponential families, mirror descent in the mean parameters is a natural-gradient step in the natural ones: Raskutti & Mukherjee, [arXiv:1310.7780](https://arxiv.org/abs/1310.7780)), plus momentum (accelerated mirror descent: Krichene, Bayen & Bartlett, NeurIPS 2015, metadata to check; stochastic version with averaging, the analogue of the Langevin row: Krichene & Bartlett, [arXiv:1707.06219](https://arxiv.org/abs/1707.06219)) | proposed | low | Daniel, 2026-11-07 | the failure of naive momentum in the row above was an invalid belief (a negative precision); a mirror map rules that out. Needs the same three interface additions. A complicated optimiser: test it on the benchmark below (Tooling), not on one graph |
| A spectral diagnostic for message passing (partial: Anderson done), and Anderson acceleration as a schedule: report the leading eigenvalues of the linearised sweep at a fixed point (exact for Gaussian means once the precisions converge; AD or finite differences of `sweep!` otherwise), and an `AbstractSchedule` that wraps another with Anderson mixing of the last $m$ sweeps | proposed | medium | Daniel, 2026-11-07 (optimiser composition); [[Composing Optimisers]] §4–5 | the spectrum says in advance whether momentum (real eigenvalues near 1), damping (near −1) or Anderson (complex) should help; heavy ball destabilises complex eigenvalues of modulus 0.95 for every β ≥ 0.3 (closed form). Anderson keeps BP's fixed points and needs none of the belief-geometry additions above. **Measured once (2026-11-08)**: `bench/bp_spectrum.jl` computes the spectrum on the divider (all slow eigenvalues complex, on a ring; heavy ball diverges for β ≥ 0.07; a schedule round the loop converges at ρ⁶ per sweep), [[Composing Optimisers]] §4.1. **Anderson measured (2026-11-09)**, `bench/bp_anderson.jl`, §4.3: around `spanning_tree_schedule` with $m = 2$–3 it needs 0.09–0.19 of flooding's evaluations, and 414 instead of 297 108 where the loop gain is near 1; around flooding it is worse in the ordinary cases or fails. **`AndersonSchedule(inner; m)` done in Mycelium (2026-11-09, `anderson.md`)**: around the spanning-tree order, by `infer!`, 24 sweeps instead of 20 881 with equal resistors. The diagnostic as a library function remains |

## Out-of-core graphs (planned subrepository)

A graph too big for memory, kept in a store and loaded where a query reaches. It is planned as its
own package, `lib/MyceliumStore.jl` (working name), on top of Mycelium. Daniel, 2026-11-05: not only
for language models but also for heterogeneous multi-omics or streaming time series (markets,
sensors). Most of the computation can be avoided in most cases, while particular data is still
needed many times.

| feature | status | priority | source | why |
|---|---|---|---|---|
| Lazy factor nodes: a stub with a store key that loads its factor and parameters on first use (a message, a free energy), with an LRU cache and eviction; the graph's structure stays resident | planned | medium | Daniel, 2026-11-05 | a model whose parameters do not fit in memory; mixture-of-experts serving does this for its experts (offloading, memory-mapped weights) |
| Nodes stored as description plus arrays: a registered type tag and version, the constructor's arguments as data, and `ps`/`st` in a memory-mappable format (Arrow, SafeTensors, HDF5/JLD2 datasets, `Mmap`); no serialised functions or closures (Julia's `Serialization` is not a stable format, and deserialising untrusted data is unsafe) | planned | medium | Daniel, 2026-11-05 | factors are already split the Lux way, a small description and arrays; only `init_*` closures hold code, and they are needed only for initial parameters |
| A store interface with back ends: files keyed by node id, LMDB (memory-mapped, read-heavy), SQLite, RocksDB, Redis, object stores | planned | low | Daniel, 2026-11-05 | one interface; the choice per deployment |
| The subgraph a query reaches: from the clamps and the query variables, the factors connected through unobserved variables (Bayes-ball / d-separation), and a schedule restricted to it; incremental appends (iSAM-like) touch only the newest factors | planned | medium | Daniel, 2026-11-05; LenticulumLM's Incremental Next-Token Prediction | what makes lazy loading pay; full loopy inference would load everything |
| Parameters by key on demand instead of one graph-wide `NamedTuple` (the integer-slot store of the inference section), and training on a loaded subgraph with write-back of parameters and optimiser state | planned | low | Daniel, 2026-11-05; ChatGPT review | gradients are local per parameter key, so a loaded subgraph trains as a block of coordinates |

## Structure search and editing

| feature | status | priority | source | why |
|---|---|---|---|---|
| Template edits of plated graphs, stable ids | planned | low | [[L-SPEC-011 Plate Edits]] | edits and `as_plate` take about 15% of a search today |
| Moves and warm starts as library code (beam search, final refit) | proposed | low | `examples/structure_clusters.md` | the example's beam with refit reaches the best fit in 62% of the gradient search's time; a reusable driver would serve other families |
| Collapsed M-steps as a library feature (integrate out the intermediates of a switch-over-sum) | proposed | medium | `examples/structure_clusters.md` | the per-factor M-step crawls with nearly deterministic switches |
| Gated inclusion of edges (spike-and-slab): a binary gate and a slab per candidate edge, inferred with the rest; a structure edit becomes the rounding of a gate | proposed | medium | [[The Regulariser and the Loss]] §2, §6; Daniel, 2026-11-05 ("valuable but rarely used, or to regularise?") | separates whether an edge exists from how large it is, so a rare but large edge is not shrunk away |
| Per-edge diagnostics: usage (posterior mass through the edge), gain when used, net value against its complexity, bootstrap stability | proposed | medium | Daniel, 2026-11-05 | tells a rare but valuable connection from one to prune; also says what an out-of-core store must keep loadable |
| Structure scores beyond BIC: a Laplace evidence (the Hessian at the fit) and a held-out free energy | proposed | medium | [[The Regulariser and the Loss]] §6 | BIC is the crude Laplace term; held-out scores also judge parts that are not normalised densities |
| A structure-search driver in the library: move sets (edits, gates, cluster merges), the scores above, beams, restarts and a final refit | proposed | medium | [[The Regulariser and the Loss]] §5–§6; `examples/structure_clusters.md` | every search today is example code; merges the "moves and warm starts" row above |

## Learning

| feature | status | priority | source | why |
|---|---|---|---|---|
| The warning of LP-022 for data that arrive other than by `DataFactor` (pins, point messages) | proposed | low | [[LP-022 Constraint factors are normalised in their residual, not their channels]] | |
| MAP M-steps for noise priors (Wishart, log-normal); M-steps for further families | proposed | low | `src/m_steps.md` §4 | `GaussianFactor` and `GaussianPrior` have ML M-steps since 2026-11-05; with a prior their noise is not updated |
| Stop-gradient semantics per wire, factor and parameter subset | planned | medium | [[L-SPEC-002 Stop Gradient]] | |
| Exposed parameters: generic approximate `parameter_message`, energy taps, risk scalarisations | partial | low | [[L-SPEC-003 Exposed Parameters and Energy Channels]] §2–§3 | |
| Explicit gradient semantics in results (envelope, finite replay, implicit fixed point) | proposed | low | ChatGPT review §4.2 | a caller should see which objective was differentiated |
| Calling `validate_parameters` automatically before inference | proposed | low | LP-023 | it exists (2026-11-04) but is opt-in |
| `train!`: interleave inference sweeps and `OptimiserFactor` updates, with a check of condition (O) (a rule's fixed points are the stationary points) for user rules | proposed | medium | [[The Regulariser and the Loss]] §3; `lib/Mycelium.jl/src/factors.md` §5 | the optimiser is a factor, but no loop drives it; `GradientDescent`, `Momentum` and `Nesterov` satisfy (O) by their definitions |
| Hyperparameters by the evidence (type-II ML): an M-step for the $\lambda$ of normalised priors inside `fit_em` | proposed | medium | [[The Regulariser and the Loss]] §2 | MacKay's evidence has an interior optimum when the prior is normalised |
| Hypergradients through fixed points: $d\mathcal L_{\text{outer}}/d\lambda$ by the implicit function theorem, through `fit_em` and the message fixed point | proposed | low | [[The Regulariser and the Loss]] §5 | the adjoint machinery exists for implicit learners and DEQs; it would make held-out tuning of priors and loss weights a gradient problem |
| Learned scalarisations and loss weights with condition (L) enforced: a warning, as for LP-022, when a learned weight or penalty is fitted on its own objective | proposed | low | [[The Regulariser and the Loss]] §4 | derived there: a loss weight or an unnormalised penalty fitted on the training objective goes to zero |
| Amortised inference graph-wide, with the amortisation gap $F[q_\phi] - \min_b F$ reported | proposed | low | [[The Regulariser and the Loss]] §2; [[The Inferencer and the Optimizer]] §7 | `AmortisedInversion` exists per factor only |
| Learned optimisers (an update lens with parameters, meta-trained on an outer criterion) | proposed | low | [[The Regulariser and the Loss]] §3 | research; needs `train!` and the outer problem first |
| Energy discrepancy as the loss of a general energy factor (toric with `normalised = false`, DEQ energies, user energies): energies at perturbed data, no MCMC and no score ([arXiv:2307.06431](https://arxiv.org/abs/2307.06431)) | proposed | medium | [[Training Energy-Based Models]] §4, §6.1 | these factors are only warned about today (LP-022); this would train them correctly. Test: recovers a Gaussian's energy up to a constant |
| The absolute level of a learned energy: NCE against a known noise, or telescoping density ratios, to fix an `EnergyNetwork`'s constant per level and model | proposed | medium | [[Training Energy-Based Models]] §6.2 | needed to compare models and structures by energy, and for a log-evidence beyond the Gaussian case. Test: a Gaussian mixture's `logpartition` |
| A time-independent energy for the implicit learner (Energy Matching, [arXiv:2504.10612](https://arxiv.org/abs/2504.10612); Equilibrium Matching, [arXiv:2510.02300](https://arxiv.org/abs/2510.02300)): one field whose roots are the relation and whose descent samples | proposed | low | [[Training Energy-Based Models]] §4, §6.3 | replaces the sum over noise levels; compare on the calibration benchmark |
| Sliced score matching for energy factors on clean data (no noise level) | proposed | low | [[Training Energy-Based Models]] §4 | score matching where no diffusion model is wanted |
| Diffusion recovery likelihood (short conditional chains per noise level) | proposed | low | [[Training Energy-Based Models]] §4 | a fallback if denoising score matching proves insufficient |
| For LenticulumLM: a sequence-level residual energy per diffusion step (EDLM, [arXiv:2410.21357](https://arxiv.org/abs/2410.21357)) and concrete score matching for a discrete implicit learner ([arXiv:2211.00802](https://arxiv.org/abs/2211.00802)) | proposed | low | [[Training Energy-Based Models]] §4 | the parallel-decoding correction and the discrete learner's loss |

## Utility factors and data (asked for 2026-11-05)

Daniel: "there should be a whole bunch of utility factors that are still lacking." Mycelium has
`DataFactor`, `PriorFactor`, `LossFactor`, `RelayFactor` (an equality between two variables),
`OptimiserFactor`, `TableFactor` and `StraightThroughFactor`. Today the operations below are
written by hand, from `LinearConstraintFactor`s with selection matrices or from graph code.

| feature | status | priority | source | why |
|---|---|---|---|---|
| Data loading from `MLUtils.jl` (`DataLoader`): minibatches fed into a plate's `DataFactor`s, shuffling, epochs, as an optional extension | proposed | medium | Daniel, 2026-11-05 | every example builds one `DataFactor` per point by hand and keeps all data in the graph; minibatch training needs the plate's data swapped per batch |
| A split / concatenate factor: one channel of dimension $d_1 + d_2$ related exactly to two channels of $d_1$ and $d_2$ (and more pieces, any index sets), with exact messages in every direction | proposed | medium | Daniel, 2026-11-05 | today a selection-matrix `LinearConstraintFactor` with tiny noise, which is neither exact nor cheap |
| A normalising change of variables: an affine factor $z = (x - b) \oslash s$ (per coordinate or per group) with exact messages both ways and the log-Jacobian $\sum_i \log\lvert s_i\rvert$ in its free energy; fixed $b$, $s$ from the data (standardising data channels) or learned with data-dependent initialisation (Glow's ActNorm) | proposed | medium | Daniel, 2026-11-05 (batchnorm and group norm in deep-learning libraries) | conditioning: the flat-direction threshold, eigenvalue floors and EM are relative to scale. Without the log-Jacobian a learned scale collapses like LP-022 (squash $x$, raise the density of $z$). Test: a chain with and without it differs by exactly $\sum \log\lvert s_i\rvert$ (mind the deterministic factor's Bethe bookkeeping, as for the relay in LP-019) |
| Shared plate statistics: a constructor for a plate whose replicas share mean and scale variables ($x_i \sim \mathcal N(\mu, \sigma^2)$), the probabilistic reading of batchnorm's batch statistics | proposed | low | Daniel, 2026-11-05 | expressible today (`examples/hierarchical_models.jl`); the posterior over $\mu, \sigma$ replaces running averages at test time. Not batchnorm's forward pass, which couples every replica through a non-local factor and is no density |
| Further utility factors: a deterministic map (a function as an exact relation, `FunctionFactor`), a permutation or reshape of a channel, a sum or average of several channels, a constant (clamp) factor with its own schedule | proposed | low | Gemini review §4 (a `FunctionFactor`), Daniel, 2026-11-05 | the common glue of a model, written by hand each time |

## Families

| feature | status | priority | source | why |
|---|---|---|---|---|
| Graded block levels in `ImplicitDiffusion`: a fixed noise level per block that rises with the block's age (old blocks nearly clean, recent ones noisier, free ones annealed), generalising today's annealed-or-clean blocks | proposed | low | LenticulumLM's [Incremental Next-Token Prediction](https://mathstruct.org/LenticulumLM/Design/Incremental-Next-Token-Prediction) (an idea, 2026-11-05) | the schedule of Rolling Diffusion and Diffusion Forcing as a query option; holding clamps clean already helps (`implicit.md` §9); whether untrained-for schedules help is open |
| Calibration benchmarks for implicit relations | partial: circle, lemniscate, unbalanced rings (2026-11-05, `lib/VariationalDiffusion.jl/examples/calibration.md`); a pitchfork and a redundant arm remain | medium | both reviews (LP-023) | baseline measured: every off-domain query returns a root, the lemniscate's recall is at best 0.52, the 10% ring is never found |
| Calibration remedies: off-relation detection (the full score's norm, an energy threshold, a round trip), annealing the field levels, ridge residuals | proposed | medium | both reviews; `calibration.md` | measured against the benchmark above |
| Diffusion posterior samplers, as query methods of `ImplicitDiffusion` beside root finding, RED-Diff (`reddiff.jl`), ProxDM (`proxdm.jl`) and replacement/RePaint (`implicit_sample`). Each is checked against the exact posterior of `GaussianMixtureEps` under a linear-Gaussian measurement, and compared on the calibration benchmark (rows below) | proposed | medium | Daniel, 2026-11-05 ("whatever useful sampling method appears") | today the library samples a query by replacement or optimises it (RED-Diff, ProxDM); a noisy or nonlinear measurement needs guided or particle samplers |
| — DPS: the reverse SDE guided by $\nabla \log p(y \mid \hat x_0(x_t))$ through Tweedie's estimate ([arXiv:2209.14687](https://arxiv.org/abs/2209.14687); `chung2023dps` in refs.bib) | proposed | medium | as above | the standard baseline; needs the network's VJP, which the adjoints provide |
| — ΠGDM: pseudoinverse guidance for linear measurements (Song, Vahdat, Mardani & Kautz, ICLR 2023; not on arXiv, metadata to check on OpenReview) | proposed | low | as above | cheaper and better-conditioned guidance for linear operators |
| — DiffPIR as a sampler: half-quadratic splitting with the noise predictor's Tweedie denoiser as the prior's proximal step, then re-noising ([arXiv:2305.08995](https://arxiv.org/abs/2305.08995); `zhu2023diffpir`) | proposed | low | as above | `prox_infer` already does the splitting with a proximal network; this uses an ordinary noise predictor |
| — DDRM: SVD-based sampling for linear inverse problems ([arXiv:2201.11793](https://arxiv.org/abs/2201.11793)) | proposed | low | as above | exact handling of a linear operator's singular directions |
| — DAPS: decoupled noise annealing ([arXiv:2407.01521](https://arxiv.org/abs/2407.01521)) | proposed | low | as above | large moves early, against DPS's local corrections |
| — Twisted diffusion sampler: sequential Monte Carlo with twisting functions, asymptotically exact conditional sampling ([arXiv:2306.17775](https://arxiv.org/abs/2306.17775)) | proposed | medium | as above | the one *exact* sampler of the family as particles grow: an oracle for the others, and weighted samples are a `SampleBelief` the graph can use |
| — MCG: manifold-constrained gradient ([arXiv:2206.00941](https://arxiv.org/abs/2206.00941)) | proposed | low | as above | a correction to DPS-style guidance |
| Heteroscedastic and heavy-tailed likelihoods | proposed | low | ChatGPT review §5 | |
| Polynomial factor: higher-degree ideal quotients, missing-data EM | planned | low | `polyfit.md` | |
| Algebraic factors: MLE existence by LP, Markov bases, splitting large tables | planned | low | [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] open, §2.4 | |
| Polarity-dependent energies (allowed by the interface, used by no factor) | proposed | low | [[L-SPEC-001 The Factor Interface]] open | |

## Tooling

| feature | status | priority | source | why |
|---|---|---|---|---|
| A benchmark of inference and learning problems for optimiser design: problems with exact or reference answers where today's methods are slow or fail, run by one script that records sweeps, wall time, error against the reference, the free energy reached, and failures. Candidates: the loopy divider (flooding: 326 sweeps to an error of 1e-10, and 97 224 with equal resistors; `bench/bp_spectrum.jl`, `bench/bp_schedules.jl`), a small discrete Ising grid (exact by junction tree), Gaussian grids near the walk-summability limit, the switch learning of `examples/structure_clusters.jl` (EM with SQUAREM, restarts, beam), the calibration benchmark, MNIST in both directions. Methods and compositions to compare: damping, SQUAREM, Anderson acceleration, heavy ball, mirror and accelerated mirror descent, Langevin, learned optimisers ([[The Regulariser and the Loss]] §3); the composition laws and an order of experiments are in [[Composing Optimisers]] | proposed | medium | Daniel, 2026-11-07 ("needs a serious learning problem to test various methods or compositions") | every optimiser idea so far was tried on one graph or one example; a fixed set of problems makes them comparable and keeps a method that wins on one from being adopted on that alone |
| Benchmark regression tracking (scaling, allocations per message) | proposed | low | ChatGPT review §6 | |

## Done (2026-11-04/05)

Overlaid ("mirrored") factors (2026-11-05): `Mycelium.OverlayFactor`, several factors with their own
parameters on the same variables as one node; unnormalised the product (a cluster of the
members), with `normalise = (:y, …)` a product or mixture of experts normalised in `y`, which a
learned relation on data cannot collapse (`lib/Mycelium.jl/src/overlay.md`). Members without a
`cluster_form` (nonlinear energies) remain open (RQ-008).

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
