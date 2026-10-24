#open-problem

> **Status: closed (2026-10-25).** Related: [[L-SPEC-001 The Factor Interface]], [[L-ADR-0006 Separate likelihood_message from invert]], [[LP-003 SquaredNorm does not check its metric]], [[LP-008 No conformance test for the factor interface]], [[LP-009 Game-level free energy is declared but not implemented]].

**Problem.** A read-only review of master by Codex (2026-10-25, of the 16 commits after 5a850764: the issue #8 energy work, the cherry-picked `review-argon` fixes, L-ADR-0006) found defects that the test suites passed over, each with a reproduction. Besides reopening [[LP-003 SquaredNorm does not check its metric|LP-003]], [[LP-008 No conformance test for the factor interface|LP-008]] and [[LP-009 Game-level free energy is declared but not implemented|LP-009]]:

1. **`entropy` returned a bare number** for the five `Lenticulum` factors, though its contract makes it an element of the energy space; applying the factor's own `GradedScalarisation` to it threw a `MethodError`.
2. **The nonlinear factor's `likelihood_message` and `invert` disagreed** for `iterations > 1`: only `invert` relinearised at the posterior, so the BP message was not the site of the posterior (against L-ADR-0006). For $f(x) = x^2$, noise 0.1, prior $\mathcal N(1, 1)$, $y = 4$, four iterations: posterior mean 1.9938 from `invert`, 2.4634 from the prior times the message.
3. **Predictive `LikelihoodFactor` free energy set `fit` and the conditional entropy both to 0**, because they cancel in the plain sum; any other scalarisation got the wrong value (ordinal, thresholds $\pm1$, $x = 0$: both are 0.8448).
4. **The mixture `likelihood_message` (`VariationalDiffusion`) kept the posterior's weights** after dividing out the anchor; the weights also need each component's log-partition change (two modes, anchor $\mathcal N(0.4, 4)$: weights 0.68/0.32 against 0.59/0.41).
5. **The prior's cotangent in `implicit_factor_pullback`** was the first unobserved block's cotangent even when an incoming point overrode the prior: 0.20 where finite differences give 0.
6. **RQ-002 and *Universal Approximation for Implicit Diffusion Learners* §3** claimed an $O((\sigma^\ast)^2)$ error from uniform continuity alone; uncentred shifts give $O(\sigma^\ast)$.

**Resolution.** Codex implemented the fixes test-first on `codex/review-fixes`, one commit each; Claude reviewed them, kept predictive mode working for the count likelihoods (Codex had made it throw) by summing their conditional entropy as a series, and corrected the notes: 1 → 4a49ca5c; 2 → c03014c3, with the graph message on the same routine in bb3af9e9; 3 → c5f07b5a, 3be38b3e; 4 → 4c184538; 5 → 84b69146; 6 → the notes now say $o(1)$, $O(\sigma^\ast)$ for $\hat G \in C^3$, and $O((\sigma^\ast)^2)$ only for centred shifts and $\hat G \in C^4$. The review is `Lenticulum.jl-Review/10_Review_of_master_(Codex).md` (outside the repository).

**Lesson.** Records were marked closed on the strength of commit messages. A resolution names the code or test that closes it, and is checked against the code before the status changes.
