#open-problem

> **Status: resolved for U3 (2026-11-20); one finding open as row I26.** Two reviews of U3
> (`ScaleShiftFactor`, the normalising change of variables). Codex made six findings, all right.
> Four are defects of the factor, now fixed with regression tests: messages that the core's
> eigenvalue cuts made hard or flat, an input energy lost to cancellation, the numerics of
> `scale_shift_parameters`, and the constructor's checks. Finding 1 is a defect of
> `HardConstraintFactor` that is there without the new factor (`@test_broken`, row I26). Finding 4 and
> the rest of finding 2 are limits of the core, recorded and not changed. Mistral wrote no answer;
> its probes, run by the session, showed no defect.
> Related: [[Feature Backlog]] (U3, I26), [[Backlog Done 2026-11]], [[LP-069 Findings of the review of U2]].

> Sources: reviews by Codex (GPT, `codex exec -s workspace-write`, reasoning effort high, one run,
> eight probes in Julia, each with its output) and Mistral Large 4 (`vibe --agent review`, 100 turns,
> \$0, seven probes, no answer file). Specification, briefs, diff, the tests Codex wrote from the
> specification, the reviews and the probes: `../Lenticulum.jl-Review-2026-11-20/u3/`.
> code: `lib/Lenticulum.jl/src/scale_shift.jl`, `scale_shift.md`, `hard_constraint.jl`. tests:
> `lib/Lenticulum.jl/test/scale_shift.jl` (testset "ScaleShiftFactor: review of U3"),
> `lib/Lenticulum.jl/test/hard_free_energy.jl` (testset "LP-070: two oblique exact projections of one variable").

**Problem.** The reviewers were given the specification (`spec.md`) and the diff. A first Codex
run had already turned the specification into the oracle tests of `test/scale_shift.jl`, without
seeing the code. On the first implementation those tests found two defects, both fixed before the
review: a non-integer `dim` threw `MethodError`, and Float32 messages came back Float64
(`HardConstraintFactor` takes its element type from its noise, a Float64 zero). The gradient
mechanism was settled before the tests: on a prototype chain, the envelope gradient without a
factor belief gave $(0, 0, 1, 2)$ where the evidence's gradient is $(-0.28, -3.17, 0.89, -11.09)$
(`scale_shift.md`, "The parameter gradient").

## Codex

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | a tree with exact observations on both sides (z observed through two oblique scalar projections, each a `HardConstraintFactor`) has the right marginals and gradients but a wrong free energy: 4.807 for 7.136 with one factor in between | right (probe). **Not a defect of U3**: with no `ScaleShiftFactor` at all the same tree gives 6.005 for 6.983 (the session's run of the probe with $n = 0$). Each factor's $\tfrac12\log\det(\tilde G\tilde G^\top)$ is taken on a cavity support the other observation has already cut | `@test_broken` in `test/hard_free_energy.jl`; row I26; noted in `scale_shift.md` |
| 2 | in Float32 with $s = (10^{-3}, 1)$, the message to x had the first coordinate hard (variance 0 for $10^{-6}$) and the message to z had it flat; an identity on a proper Float64 precision $\operatorname{diag}(10^{-16}, 1)$ gave a flat direction | right (probe). The messages went through `HardConstraintFactor`'s pushforward and preimage, whose eigenvalue cuts did it. The flat direction of the identity case is the core's conversion of the input (`extended`) | messages computed directly, the diagonal map applied to the mean, covariance and flat directions (exact, $O(d^2)$); regression test. Not changed: the input conversion, and the Float32 tree with unit priors on both sides, whose marginals and free energy still go wrong through the Float32 hard-direction threshold of LP-069 (the messages are right) |
| 3 | the input energy, uncentred ($\tfrac12\mu^\top\Lambda\mu - \eta^\top\mu$), lost its change to cancellation: the default gradient at a mean of $10^8$ was $5\cdot10^5$ for $-1$, and `belief_free_energy` was NaN at $10^{155}$. The default backend (`ad = nothing`, central differences) is also 2–5 % off on Float32 data | right (probe) | centred on the input's mean ($\tfrac12\delta^\top\Lambda\delta$); regression tests (value equal to `local_free_energy`, ForwardDiff gradient exact, default gradient within $10^{-2}$ at $10^8$, where it is now $-0.998$). Not changed: the default backend's step (Mycelium's, shared by every factor) |
| 4 | a mixture of points crashes the messages and the free energy (`moment_match` has no method for a `MixtureBelief` of `DiracBelief`s) | right (probe); `HardConstraintFactor` and `SplitFactor` call the same conversion | recorded in `scale_shift.md`; not changed |
| 5 | `scale_shift_parameters` overflowed an `Int64` sum, overflowed and underflowed squared deviations at $10^{\pm200}$, took the rounded spread of equal entries $[0.1, 0.1, 0.1]$ for a scale, and gave 1.41 for $[10^{16}, 10^{16} + 2]$ | right (probe) | floating point from the start, deviations scaled before squaring, equal entries refused exactly; regression tests. The last case is a rounded midpoint ($10^{16} + 1$ is not a Float64) and is left |
| 6 | the constructor took `groups = nothing`, threw `MethodError` for `dim` = `nothing`, `"2"`, `2 + im`, and refused `Any[1, 2]` | right (probe) | every argument checked in the inner constructor; the groups by their elements, so `Any[1, 2]` is accepted; regression tests |

## Mistral

It stopped at its 100-turn limit without an answer file. 20 of its commands were refused (a
headless run cannot approve them). The session ran its seven probes on the final code:

- **Probes 2, 3 and 3b pass.** These cover the fixed-message gradient against a closed-form
  $-\log Z_f$, and trees with hard-direction messages into the factor.
- **Probe 2b's sign difference is its own convention.** It compares the input energy with
  $+\log Z_f$; the total gradient matches the reference.
- **Probe 4 has one expected refusal that is not a defect:** `Any[1, 2]` as groups. Mistral read
  the specification's "a vector of integers" as the element type, Codex as the elements. The
  elements reading was kept (Codex finding 6).
- **Probe 5's failures are its own errors or rounding.** The star tree's dense formula left out
  each Gaussian's $\tfrac12\eta^\top\Lambda^{-1}\eta$; with it, $F$ matches to $7\cdot10^{-15}$. The
  near-singular prior (condition $10^{10}$) is off by $3\cdot10^{-7}$ relative, within rounding.
- **Probe 5b's dense marginal is wrong.** It multiplies the prior by its own pushforward taken as
  a likelihood; with $x$ observed exactly, $v_0$ is a point. $F$ matches to $10^{-14}$.

So Mistral found no defect.

**Not run by the reviews:** loopy graphs, device tests, the docs builds; the package suite was run by
the session (see the commit).
