#open-problem

> **Status: open, partly addressed (2026-10-25). The 2026-10-23 resolution named a function that did not exist.** Related: [[L-SPEC-001 The Factor Interface]], [GitHub issue #8](https://github.com/MathStruct/Lenticulum.jl/issues/8).

**Problem.** The factor interface is tested factor by factor. Issue #8 was a mismatch between two parts of the contract (`scalarisation` and `energy`) that no test covered, because each factor's tests exercised only the paths it used. Other factor types (the diffusion, equilibrium, neural-ODE and adversarial factors) may have similar gaps.

**Options.**
1. A conformance test: a function `check_factor(f, ps, st, samples)` in a test utility that runs every applicable part of [[L-SPEC-001 The Factor Interface]] (§1–§5) and the consistency conditions between them, called from each package's tests for each factor type.
2. Keep per-factor tests and add the cross-checks by hand.

**Resolution so far.** The `review-argon` commits claimed option 1 (`check_factor_conformance`) but only added a loop over traits; the function did not exist ([[LP-013 Defects found in the review of master]]). Since bb3af9e9 it does, in `test/factor_conformance.jl`: for a factor and polarity it checks the testable parts of [[L-SPEC-001 The Factor Interface]] §1–§5 (traits, `invert` = `combine(π, likelihood_message)`, the pointwise energy and its scalarisation, game-level `free_energy` and `entropy` in `energyspace`). It runs over every supported polarity of `GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor` and `LikelihoodFactor` (297 checks). **Open:** the factors of `Mycelium`, `ImplicitLayers`, `VariationalDiffusion` and `Adversarial` are not run through it yet.
