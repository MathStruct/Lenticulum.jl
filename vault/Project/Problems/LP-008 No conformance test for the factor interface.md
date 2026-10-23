#open-problem

> **Status: closed (2026-10-23, `review-argon`).** Related: [[L-SPEC-001 The Factor Interface]], [GitHub issue #8](https://github.com/MathStruct/Lenticulum.jl/issues/8).

**Problem.** The factor interface is tested factor by factor. Issue #8 was a mismatch between two parts of the contract (`scalarisation` and `energy`) that no test covered, because each factor's tests exercised only the paths it used. Other factor types (the diffusion, equilibrium, neural-ODE and adversarial factors) may have similar gaps.

**Options.**
1. A conformance test: a function `check_factor(f, ps, st, samples)` in a test utility that runs every applicable part of [[L-SPEC-001 The Factor Interface]] (§1–§5) and the consistency conditions between them, called from each package's tests for each factor type.
2. Keep per-factor tests and add the cross-checks by hand.

**Resolution.** Option 1 implemented in `test/runtests.jl` (`check_factor_conformance`), testing §1–§5 of [[L-SPEC-001 The Factor Interface]] across `GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor`, `NonlinearFactor`, and `LikelihoodFactor`, plus pointwise/free-energy contract checks in `ImplicitLayers`.
