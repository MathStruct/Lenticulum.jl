#open-problem

> **Status: resolved (2026-11-20).** The review of U2 (`SplitFactor`, a channel and pieces of it,
> exactly) found a wrong free energy where an input has a hard direction oblique to the
> coordinates, a cut that removed true small variances, a Float32 product that went flat, a
> collision of a channel name with an internal marker, unchecked constructors, a rounding defect in
> the core fix that came with it, and four overstatements in the note. All twelve were right. Seven
> are fixed with tests, three are notes corrections, and two are limits of the extended-Gaussian core
> that are recorded and not changed.
> Related: [[Feature Backlog]] (U2), [[Backlog Done 2026-11]], [[LP-019 Defects found in the review of the core packages]].

> Sources: review by Codex (GPT, `codex exec -s workspace-write`, reasoning effort high, one run,
> probes of its own in Julia, each with its output). Specification, briefs, diff, the tests Codex
> wrote from the specification, and the probes: `../Lenticulum.jl-Review-2026-11-20/u2/`.
> code: `lib/Lenticulum.jl/src/split.jl`, `lib/Lenticulum.jl/src/split.md`,
> `lib/LenticulumCore.jl/src/extended_gaussian.jl` (`pushforward`, `preimage`). tests:
> `lib/Lenticulum.jl/test/split.jl` (testset "the findings of its review (LP-069)").

**Problem.** The requirement given to the reviewer was the specification of `SplitFactor`
(`spec.md`), which a second Codex run had already turned into the oracle tests of
`test/split.jl` without seeing the code. Those tests found two defects before the review: a
Float32 marginal that came back Float64 (in `LenticulumCore.pushforward`), and a piece covariance
of pure rounding that was taken for a variance (tree 6). Both were fixed before the review.

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | two points that disagree on a shared coordinate pass as agreeing when another coordinate is $10^{10}$: the consistency tolerance of `_conjoin` scales with ‖μ‖ | right (probe); a limit of the core, not of the factor | recorded in `split.md`; no change to the core's tolerance |
| 2 | `_split_marginal` cut at $q$'s tolerance and turned a true variance of $10^{-16}$ (against 1) into a point; the oracle tests' absolute `atol = 1e-7` could not see it | right (probe) | cut at the square of $q$'s tolerance over its scale (rounding there is second order); regression test |
| 3 | two proper Float32 pieces with variances 1 and $10^6$ gave a flat whole (the precision-form product cut the second direction); the input left out still set the float type | right (probe) | disjoint pieces are joined in covariance form; the float type comes from the inputs that take part; regression test of the message. The free energy of that tree still fails, by a third limit of the core: a Float32 extended Gaussian takes a variance below $32\varepsilon_{32}$ of the largest for a hard direction, so the entropy drops the unit direction (`@test_broken`); at $10^4$ it passes (test) |
| 4 | an improper Gaussian site that is no extended Gaussian (a linear potential with zero precision, or a negative precision) loses information or is refused when converted | right (probe); the core's `extended` conversion, shared with `HardConstraintFactor` | recorded in `split.md`; not changed |
| 5 | a channel named `Symbol("")` collided with the marker for "no channel left out" in the free energy | right (probe) | the marker is `nothing`; regression test |
| 6 | the core fix rounded an explicit integer offset to Float32 before Float64 arithmetic (16 777 217 became 16 777 216) | right (probe) | an integer offset of `pushforward` and `preimage` takes the promoted type of the belief and the map; regression tests in LenticulumCore |
| 7 | the positional constructor was unchecked; `Bool` indices and `dim = true` were accepted; an index above `typemax(Int)` threw `InexactError`; `supports_polarity` accepted a role other than the three | right (probe) | every constructor goes through an inner constructor with the checks; the polarity check names the three roles; regression tests |
| 8 | the term $-H(q)$ (from the specification) differs from $-\log Z$ by $\log\sqrt2$ for a prior on the line $z = (T, T)$ with $x = z_1$ clamped: a hard direction oblique to the coordinates leaves a Jacobian that $-H(q)$ does not have | right (probe, run as a regression test) | the factor's term is now `HardConstraintFactor`'s for $Sz - x = 0$ when an input has a hard direction, and $-H(q)$ (equal to it, by Sylvester's determinant identity) otherwise; regression test $F = \tfrac12\log 2\pi$. Consequence: two inputs fixing the same direction (two points on a shared coordinate) now throw, as the evidence has a $\delta(0)$; the conformance test's example changed accordingly |
| 9 | the `RelayFactor` equivalence of the specification does not hold for mixtures (passed through by the relay, collapsed here) nor for all-flat inputs (0 from the relay, a refusal here) | right | docstring and note restrict it to proper, non-mixture inputs |
| 10 | the note's comparison with `HardConstraintFactor` used the full Jacobian where the inputs reduce the support, and wrote $G$ as a vertical block | right | the section "Free energy" rewritten: the relation's term in general, $-H(q)$ only without hard directions |
| 11 | "flat along any coordinate whose pieces send no information" should speak of directions | right | that sentence went with the rewrite |
| 12 | the note said the code had no mutation for Zygote; the message path mutates (`combine` fold, `marginalise`), and Zygote failed in the pullback | right (probe) | the note says it is not differentiable with Zygote |

**Not run by the review:** the package suites and the documentation builds (the session ran the
suites; see [[Backlog Done 2026-11]], U2).
