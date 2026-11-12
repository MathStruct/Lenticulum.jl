#open-problem

> **Status: resolved (2026-11-12).** No defect in the code. Two notes were stale or missing, one
> backlog finding was already handled, and one finding was a hazard (two internal functions of one
> name with different behaviour), removed by a rename. Related: [[Feature Backlog]] (P17, P18),
> [[LP-028 Findings of the review of P1]].

> Sources: a review of the uncommitted moves P17 and P18 by Mistral (`vibe --agent review`), run with
> Julia in the helper worktree `Lenticulum.jl-mistral` (2026-11-12). Its answer is in
> `../Lenticulum.jl-Review-2026-11-12/P17-18-from-mistral-2.md` (the first run stopped at its turn
> limit); its scripts are `scratch/p17-18/check_p17.jl` and `scratch/p17-18/check_p18.jl` in that
> worktree.
> code: `lib/Mycelium.jl/src/gaussian.jl` (`reduce_mixture`, `reduce_message`),
> `lib/LenticulumCore.jl/src/gaussian_belief.jl` (`belief_mean`, `belief_cov` of `SampleBelief`),
> `lib/Adversarial.jl/src/ratio.jl`, `lib/Lenticulum.jl/src/Lenticulum.jl` (the re-export);
> tests: `lib/Mycelium.jl/test/mixture_reduction.jl`, `lib/LenticulumCore.jl/test/sample_moments.jl`.

**Problem.** Two moves out of type piracy were required to change no result: Runnalls' mixture
reduction from Lenticulum to Mycelium (P17) and the sample moments from Adversarial to LenticulumCore
(P18), with the old public names still working.

Mistral checked behaviour first. It compared the moved functions with the code before the move,
copied verbatim from `HEAD`: P17 on 40 random mixtures with every budget and several prune values,
and P18 on 40 random sample beliefs (matrix, vectors, scalars, weights scaled by 1e300, uniform). Both
were identical. `Lenticulum.reduce_mixture === Mycelium.reduce_mixture`, Aqua passes in the four
touched packages, and no example, benchmark or docs page used the old locations.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | The header comment of LenticulumCore's `gaussian_belief.jl` still placed `reduce_mixture` in Lenticulum | right | corrected | — |
| F2 | The backlog rows P17 and P18 were still "proposed" | right for the diff Mistral had; the backlog was updated after the diff was taken | nothing more | — |
| F3 | Mycelium's `gaussian.md` had no section on the code moved into `gaussian.jl` | right (the convention of a note beside each source file) | section "Mixture reduction" added | — |
| F4 | LenticulumCore's new `_sample_weights` and Mycelium's existing `_sample_weights` (`messages.jl`) differ for all-zero weights (an error against uniform weights) | right as a hazard, not a defect: each module calls its own, and the `SampleBelief` constructor refuses all-zero weights | LenticulumCore's renamed `_moment_weights`, with a comment | — |

**Leaning / resolution.** No code defect. F1, F3 and F4 were fixed in notes and a name.

Checked after the triage: `julia --project=docs docs/make.jl` passed (API docs and vault), with
`reduce_mixture` on Mycelium's API page. Not checked: nothing outside this repository calls the old
locations (no registered dependants).
