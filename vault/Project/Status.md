#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after P17 and P18 (committed and pushed).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P1 (`597dc8e2`) and P17/P18 (`fbbb6cdf`) committed and pushed, with the rest of I18; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `597dc8e2`, with P17/P18 applied for Mistral's review, its scratch scripts in `scratch/p1/` and `scratch/p17-18/` (untracked). Two stale stashes: `mistral-wt-polarity-cache-draft-2026-11-12` and `mistral-wt-i18-predfix-2026-11-12`; both can be dropped |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done today

- **P1** (`597dc8e2`): tests where the flagged and the extended form meet, and three fixes
  ([[LP-028 Findings of the review of P1]]). Left over: P21 (`@test_broken`).
- **P17, P18**: `reduce_mixture`/`reduce_message` of Gaussian mixtures moved from Lenticulum to
  Mycelium (`gaussian.jl`; Lenticulum re-exports `reduce_mixture`); `belief_mean`/`belief_cov` of
  `SampleBelief` moved from Adversarial to LenticulumCore (`gaussian_belief.jl`). The two Aqua piracy
  exceptions are gone. Tests: Mycelium `test/mixture_reduction.jl` (Mycelium alone), LenticulumCore
  `test/sample_moments.jl`. Review: Mistral, no code defect ([[LP-029 Findings of the review of P17 and P18]]).
- Verified for P17/P18: all eight suites passed (`~/.cache/lenticulum/p17-18/`), LenticulumCore and
  Adversarial again after the last rename, and the docs build. Not run: device tests (no device code changed).
- Next: by the proposed order in [[Feature Backlog]]: P11 with F3 for
  path B, and I6 for path C; or I16/I17, P21.

## Open

- **P21:** ForwardDiff through a relation factor's flat-input path, and `NonlinearFactor` at a dual
  expansion point (`@test_broken` in Lenticulum `test/forms_crossing.jl`).
- **P20:** Zygote through a plated free energy fails on `sum(...; init)` (found by Mistral; the
  same before P7).
- AlgebraicFactors, ImplicitLayers and Adversarial run in CI only weekly: a change to the core
  should run their suites locally too (that is how the AlgebraicFactors break was found).
- Mistral review agent: `vibe ... --agent review --trust --workdir ../Lenticulum.jl-mistral`
  (`~/.claude/CLAUDE.md`). A review needed about 140 turns; give it 100 and continue the session
  with `vibe -c` if it stops.
