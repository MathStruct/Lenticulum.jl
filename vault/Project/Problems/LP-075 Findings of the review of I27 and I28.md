#open-problem

> **Status: resolved (2026-11-22).** A review of I27 (the gradient drivers refuse a free energy that is not
> finite) and I28 (a `MeanField()` node next to a variable with hard directions). Codex made three findings.
> One was a defect of the change and is fixed with a regression test: the coupled drivers (the replay and
> the implicit adjoint) skipped the new check, since they differentiate their own stores. One was a missing
> test, now added (the switching factor shares the changed helper). One is how Nesterov is defined and was
> not changed. A side probe, a replay from a store whose free energy is refused, is not a defect.
> Related: [[Feature Backlog]] (I27, I28), [[Backlog Done 2026-11]], [[LP-071 Findings of the review of I26]].

> Sources: review by Codex (GPT, `codex exec -s workspace-write`, default model and effort), with the
> requirement and the diff, not the reasoning. Brief, diff, answer and probes:
> `../Lenticulum.jl-Review-2026-11-22/i27/`.
> code: `lib/MyceliumLearning.jl/src/gradient.jl` (`_check_finite_free_energy`), `gradient_couplings.jl`
> (`_coupled_gradient`), `graph_adjoint.jl`; `lib/Lenticulum.jl/src/constraint.jl` (`_product_belief`);
> notes `gradient.md` ("The free energy must be finite (I27)"), `constraint.md` ("Mean-field energy").
> tests: `lib/Lenticulum.jl/test/hard_overlap.jl`, testset "I27, I28: review of I26".

**Problem.** LP-071 left two older defects as rows. I27: at fixed beliefs the envelope gradient reads
neither the variable terms nor a factor's own rank check, so two exact observations of one direction
($Y_2 = 2Y_1$, an infinite free energy) got finite gradients from every backend while
`scalar_free_energy` threw (Codex's I26 probe 2, rerun: `probe2.out`). I28: `_product_belief` refused an
`ExtendedGaussian` marginal, so a mean-field likelihood next to two exact projections threw where belief
propagation gives the dense $-\log Z$ (I26 probe 6: `probe6.out`). After the change both probes give the
refusal and the dense value (`probe2.after`, `probe6.after`).

**What was done.** `free_energy_gradient` evaluates `scalar_free_energy` once before the envelope driver and
refuses a value that is not finite; `_product_belief` takes an extended Gaussian flat nowhere by its mean and
its covariance, singular along the hard directions (a flat one is still refused).

## Codex

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | the coupled drivers return before the check: a constant extra cost of `Inf` or `NaN` (a `LossFactor`) gave finite derivatives under `ExactCoupling` with a replay (`NaN` with reference differences), moved `parameter_step`'s bias $0.3 \to 0.3026$, and the mixed-coupling implicit adjoint, which calls the envelope driver at its converged store, did the same; a score replay with infinite cost at both outcomes gave a `NaN` estimate | right (`nonfinite_replay.out`, `score_nonfinite.out`) | the check also runs at the replay's reference store (each draw for a score edge) and at the implicit adjoint's converged fixed point; regression test "non-finite free energy through the coupled drivers" (`Inf`, `NaN`; replay and mixed implicit; reference differences and ForwardDiff; gradient and step) |
| 1b | a replay from a store whose free energy is refused (redundant exact observations) returns a gradient after the coefficients were made independent | not a defect: the replay starts from empty messages at the given parameters and does not read the store's beliefs; its own free energy is finite and now checked | none |
| 2 | Nesterov's `parameter_step` checks the look-ahead parameters, so it steps from parameters whose free energy is infinite when the look-ahead's is finite | accepted as designed: Nesterov's gradient is the one at the look-ahead, which exists; the step uses no value at the current parameters | none |
| 3 | the new tests cover the switching factor, which shares `_product_belief`, only through the helper | right (a gap in the tests, not in the code: its probe passed) | its probe as a regression test, "MeanField() switching factor next to hard directions": the average under two extended Gaussians (one singular) against the closed form, a flat marginal refused, and a graph with the outcome observed against the dense $-\log Z$ and its gradient under three backends |

## Found by the suites

The first fix of finding 1 checked `scalar_free_energy` at the replay's reference store, before the
gradient. Two suites failed: Lenticulum's `coupled_keys.jl` ("I17 state versions and seeded score
replay"), since the free energy refuses tied factors with state, which the replay supports; and
ImplicitLayers' `lp002_followups.jl`, which requires an unsupported AD backend to be refused before the
ODE's right-hand side is evaluated (252 calls instead of 84). The replay now checks the value of its own
loss, and every check runs after the gradient, so the drivers' own refusals come first.

Its other probes passed: a three-replica plate with hard observations and a tied soft likelihood (belief
propagation and mean field against the dense value, gradients to $3\cdot10^{-10}$ across backends, with and
without the plate schedule), MyceliumLearning's `gradient_couplings.jl` (245 tests), score estimates.

Not run by the review: the full suites, device tests, the Krylov adjoint. The cost of the check (one
forward pass of the free energy per gradient) was not measured.
