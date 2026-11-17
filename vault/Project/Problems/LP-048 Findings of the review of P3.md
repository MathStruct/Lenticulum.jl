#open-problem

> **Status: resolved (2026-11-17).** The review of P3 (a `GaussianFactor` with no input on $x$ sends
> the exact forward message) found no defect in the new message or polarity. One finding was right: a
> mixture whose components all carry no information threw in `moment_match`, older than P3 but named
> by its requirement; fixed with a regression test.
> Related: [[Feature Backlog]] (P3), [[lib/Lenticulum.jl/src/gaussian|gaussian]] "No input on x".

> Sources: a review of the uncommitted P3 change by Mistral (`vibe --agent review`, Mistral Large 4,
> `--max-price 3`, $0 recorded), in the helper worktree `Lenticulum.jl-mistral` at `0ca25bcd` with the
> diff applied. The first run crashed on its first command inside vibe (a validation error of its
> own, `stdout-run1-crash.md`); the second ran to its 100-turn limit. The brief, the answer, the
> probes and their outputs are in `../Lenticulum.jl-Review-2026-11-17/p3/`.
> code: `lib/Lenticulum.jl/src/gaussian.jl`; test: `lib/Lenticulum.jl/test/p3_no_input.jl`.

**Problem.** The requirement given to the reviewer: the forward message with no information on $x$
(no entry, `nothing`, `TrivialBelief()`, zero precision, everywhere flat, a mixture that collapses to
one of those) is the limit of $x \sim \mathcal N(0, \tau I)$, $\tau \to \infty$, for every shape and
rank of $A$, under `factor_message`, `_gradient_message` and ForwardDiff; the backward message stays
`TrivialBelief()`; informative inputs are unchanged; the new polarity breaks nothing that enumerates
polarities.

The answer file records probe 1 and its diagnostics; probes 2, 3 and 3b are not in it. The
maintainer ran them (outputs `probe2.out`, `probe3.out`, `probe3b.out`).

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | F1 (probe 1): `MixtureBelief` of zero-precision Gaussians or of everywhere-flat extended Gaussians as $x$ throws (`belief_mean needs a proper belief`, from `moment_match` in `_moments`), for every shape of $A$, both message functions | **right**; the throw predates P3 (the old shortcut also called `_moments`), but the requirement and the note named the case | `_moments` of a mixture whose every component carries no information returns "no input" before `moment_match` | two mixture forms added to the first testset of `p3_no_input.jl`; it errored without the fix |
| 2 | Probe 1: all shapes (zero row or column, wide, tall, singular square, singular values $10^{-6}$ and 1, $A = 0$, 1 × 1) and the five non-mixture forms against the closed form | no defect: agreement to $4\cdot10^{-15}$; one flag at $\tau = 10^{30}$ was the probe's own Float64 SVD rounding times $\tau$ (probes 1b–1d) | — | — |
| 3 | Probe 2: informative inputs against the pre-change formulas; one failure, a partly flat input whose message is `TrivialBelief()` while the $\tau = 10^8$ oracle has precision $1.6\cdot10^{-6}$ | not a defect: the residue of a finite $\tau$, on the partly flat path, which P3 did not change | — | — |
| 4 | Probe 3: ForwardDiff by $A$, $b$, $Q$ against finite differences of the closed form; dual-$Q$ failures | not a defect: the oracle perturbed $Q$ without the constructor's symmetrisation; the reviewer's corrected probe 3b (and rank-preserving perturbations of a rank-deficient $A$) passes | — | — |

Not reviewed: plates, `MeanField()` nodes, loopy schedules and the free energy with the new polarity
(named in the brief; no probe reached them).
