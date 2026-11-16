#overview

> Section L of the [[Feature Backlog]]: M-steps, gradient semantics, `train!`, hyperparameters, and losses for energy factors. Open rows only; a row that is
> done moves to [[Backlog Done 2026-11]] with its id. Status, priority, ids and the column *needs*
> are explained in [[Feature Backlog]], which also holds the dependency graph.

> Sources: original to this repository; the sources of each row are in its *source* column.

| id | feature | status | needs | priority | source | why |
|---|---|---|---|---|---|---|
| L1 | The warning of LP-022 for data that arrive other than by `DataFactor` (pins, point messages) | proposed | — | low | [[LP-022 Constraint factors are normalised in their residual, not their channels]] | |
| L2 | MAP M-steps for noise priors (Wishart, log-normal); M-steps for further families | proposed | — | low | `lib/Lenticulum.jl/src/m_steps.md` §4 | `GaussianFactor` and `GaussianPrior` have ML M-steps since 2026-11-05; with a prior their noise is not updated |
| L3 | Stop-gradient semantics per wire, factor and parameter subset | planned | — | medium | [[L-SPEC-002 Stop Gradient]] | |
| L4 | Exposed parameters: generic approximate `parameter_message`, energy taps, risk scalarisations | partial | — | low | [[L-SPEC-003 Exposed Parameters and Energy Channels]] §2–§3 | |
| L5 | Explicit gradient semantics in results (envelope, finite replay, implicit fixed point) | proposed | — | low | ChatGPT review §4.2 | a caller should see which objective was differentiated |
| L6 | Calling `validate_parameters` automatically before inference | proposed | — | low | LP-023 | it exists (2026-11-04) but is opt-in |
| L7 | `train!`: interleave inference sweeps and `OptimiserFactor` updates, with a check of condition (O) (a rule's fixed points are the stationary points) for user rules | proposed | — | medium | [[The Regulariser and the Loss]] §3; `lib/Mycelium.jl/src/factors.md` §5 | the optimiser is a factor, but no loop drives it; `GradientDescent`, `Momentum` and `Nesterov` satisfy (O) by their definitions |
| L8 | Hyperparameters by the evidence (type-II ML): an M-step for the $\lambda$ of normalised priors inside `fit_em` | proposed | — | medium | [[The Regulariser and the Loss]] §2 | MacKay's evidence has an interior optimum when the prior is normalised |
| L9 | Hypergradients through fixed points: $d\mathcal L_{\text{outer}}/d\lambda$ by the implicit function theorem, through `fit_em` and the message fixed point | proposed | — | low | [[The Regulariser and the Loss]] §5 | the adjoint machinery exists for implicit learners and DEQs; it would make held-out tuning of priors and loss weights a gradient problem |
| L10 | Learned scalarisations and loss weights with condition (L) enforced: a warning, as for LP-022, when a learned weight or penalty is fitted on its own objective | proposed | — | low | [[The Regulariser and the Loss]] §4 | derived there: a loss weight or an unnormalised penalty fitted on the training objective goes to zero |
| L11 | Amortised inference graph-wide, with the amortisation gap $F[q_\phi] - \min_b F$ reported | proposed | — | low | [[The Regulariser and the Loss]] §2; [[The Inferencer and the Optimizer]] §7 | `AmortisedInversion` exists per factor only |
| L12 | Learned optimisers (an update lens with parameters, meta-trained on an outer criterion) | proposed | L7, L9 | low | [[The Regulariser and the Loss]] §3 | research; needs `train!` and the outer problem first |
| L13 | Energy discrepancy as the loss of a general energy factor (toric with `normalised = false`, DEQ energies, user energies): energies at perturbed data, no MCMC and no score ([arXiv:2307.06431](https://arxiv.org/abs/2307.06431)) | proposed | — | medium | [[Training Energy-Based Models]] §4, §6.1 | these factors are only warned about today (LP-022); this would train them correctly. Test: recovers a Gaussian's energy up to a constant |
| L14 | The absolute level of a learned energy: NCE against a known noise, or telescoping density ratios, to fix an `EnergyNetwork`'s constant per level and model | proposed | — | medium | [[Training Energy-Based Models]] §6.2 | needed to compare models and structures by energy, and for a log-evidence beyond the Gaussian case. Test: a Gaussian mixture's `logpartition` |
| L15 | A time-independent energy for the implicit learner (Energy Matching, [arXiv:2504.10612](https://arxiv.org/abs/2504.10612); Equilibrium Matching, [arXiv:2510.02300](https://arxiv.org/abs/2510.02300)): one field whose roots are the relation and whose descent samples | proposed | F2 | low | [[Training Energy-Based Models]] §4, §6.3 | replaces the sum over noise levels; compare on the calibration benchmark |
| L16 | Sliced score matching for energy factors on clean data (no noise level) | proposed | — | low | [[Training Energy-Based Models]] §4 | score matching where no diffusion model is wanted |
| L17 | Diffusion recovery likelihood (short conditional chains per noise level) | proposed | — | low | [[Training Energy-Based Models]] §4 | a fallback if denoising score matching proves insufficient |
| L18 | For LenticulumLM: a sequence-level residual energy per diffusion step (EDLM, [arXiv:2410.21357](https://arxiv.org/abs/2410.21357)) and concrete score matching for a discrete implicit learner ([arXiv:2211.00802](https://arxiv.org/abs/2211.00802)) | proposed | — | low | [[Training Energy-Based Models]] §4 | the parallel-decoding correction and the discrete learner's loss |
