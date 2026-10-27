#overview #open-problem

> Problems found while specifying or building Lenticulum.jl, numbered `LP-nnn` (Moonkale uses `P-nnn`, Sophia `SP-nnn`). Each problem gets a note made from the problem template (`vault/templates/Problem Template.md`) and a line here. Closed problems stay listed, with what closed them.

Implementation notes also record what their code does not do; a problem gets an entry here when it needs a decision, crosses several files, or is waiting for someone to pick it up. GitHub issues are linked from the problem they belong to.

| # | Problem | Status |
|---|---|---|
| [[LP-001 The vault build uses GNU-only sed]] | `sed -i` in `docs/site/build.sh` fails on macOS (issue #9) | closed (2026-10-23; CI guard 2026-10-25) |
| [[LP-002 The Jacobian side is not implemented]] | Optional DI parameter Jacobians/gradients, graph parameter steps, DEQ IFT rules cover ForwardDiff/Zygote/Enzyme/Mooncake; Euler/RK4 covers the same backends with runtime activity for Enzyme; opt-in matrix-free DEQ IFT and empty-state Tsit5 sensitivities cover tested CPU backends; broader SciML/state/GPU scope remains | implemented for first-order CPU paths (2026-10-26); follow-ups open |
| [[LP-003 SquaredNorm does not check its metric]] | An indefinite `M` makes the Jensen gap negative, silently | closed (2026-10-25, reopened once) |
| [[LP-004 link_sources rewrites unrelated notes]] | The bibliography linker drops links in three unrelated notes | closed (`review-argon`, 2026-10-23) |
| [[LP-005 Learning the noise covariance]] | Opt-in full/diagonal precision parameters for `Q` and `Σ`; normalised free energy with Wishart/log-normal noise priors or an eigenvalue floor | resolved (2026-10-26; ForwardDiff/Zygote CPU paths) |
| [[LP-006 Gaussian entropies are relative to Lebesgue measure]] | Categorical KL entropy of a continuous belief is infinite; ours are relative to Lebesgue | accepted, documented |
| [[LP-007 Missing implementation notes]] | `gaussian.md` and `beliefs.md` are cited but not written | closed (`review-argon`, 2026-10-23) |
| [[LP-008 No conformance test for the factor interface]] | Common checker covers every declared exact domain, independent product games and evaluated diffusion bounds; all 22 production types and eight test fixtures audited with deliberate semantic boundaries | resolved 2026-10-27 |
| [[LP-009 Game-level free energy is declared but not implemented]] | Labelled diffusion ELBO bounds; exact affine/source, scalar Plucker, degree-one polynomial and Gaussian composition/product games; other domains throw with reasons | resolved within agreed scope 2026-10-27 |
| [[LP-010 Where the graph of a problem comes from]] | Pipeline note and Gaussian single-factor search vs graphical lasso; measured BIC/Bethe limitations; chain recovery test | resolved within note + example scope (2026-10-27) |
| [[LP-011 Gradient couplings and isfrozen are declared but unused]] | Per-edge cuts, subset freezing, static schedule gradients, terminal score estimates and straight-through factor | implemented in documented CPU domains (2026-10-27); extensions remain |
| [[LP-012 Factor graphs and schemes]] | Cited/proved theory note: RLCT parameter evidence, dual-number tangents, linear sheaf consistency | resolved within theory-note scope (2026-10-27) |
| [[LP-013 Defects found in the review of master]] | Codex's review of master: entropy type, iterated nonlinear site, predictive fit, mixture weights, prior cotangent, a rate claim | closed (2026-10-25) |
| [[LP-014 Emitting edges drop the cavity message from the free energy]] | Cavity-reading factors now reject `Emitting()` edges during graph free-energy evaluation (`uses_cavity`, option 2); marginals unchanged | closed (2026-10-26) |
| [[LP-015 Momentum had the wrong sign on its decay]] | `Momentum` and `Nesterov` used $s' = -\gamma s - \eta\bar p$ instead of $\gamma s - \eta\bar p$ (Cruttwell et al. Example 3.16) | closed (2026-10-25) |
| [[LP-016 RatioFactor emits a particle posterior as a likelihood]] | Extrinsic log-potential messages, particle pooling once, target exclusion and posterior energy contribution checked | resolved (2026-10-26) |
| [[LP-017 DiffusionFactor omits supported polarities]] | Complete capped support, lazy traversal and joint unobserved-block inversion | resolved (2026-10-26) |
