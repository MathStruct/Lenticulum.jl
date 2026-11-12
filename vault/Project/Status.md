#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after P1 (the crossings of the flagged and the extended form; not committed).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P1 uncommitted (below), on top of the rest of I18 (`1806e17e`, not pushed); `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `1806e17e`, with the first version of P1 applied for Mistral's review (without the fix of LP-028 F1), its scratch scripts in `scratch/p1/` (untracked). Two stale stashes: `mistral-wt-polarity-cache-draft-2026-11-12` and `mistral-wt-i18-predfix-2026-11-12`; both can be dropped |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done: P1

- Tests: Lenticulum `test/forms_crossing.jl` (the dual path's flagged message against the value
  path's extended one, values and derivatives, for `GaussianFactor`, `LinearConstraintFactor`,
  `NonlinearFactor`; one `_split_moments` rule; ForwardDiff through `infer!` with one and two flat
  factors on a variable) and Mycelium `test/forms_crossing.jl` (`combine`, `belief_distance`, `damp`
  on mixed pairs, plain and dual).
- Three defects fixed, each with a test that failed before: `_likelihood_message` built the extended
  form of a dual η (ForwardDiff by `b`, `c` or an observation through a rank-deficient backward
  message threw since 2026-11-11); `damp` returned the new message undamped on a mixed pair; `combine`
  threw on a variable hearing a dual flagged and a plain extended message (found by Mistral). Notes:
  Lenticulum `gaussian.md` "Where the two forms meet", Mycelium `gaussian.md` "Extended Gaussians".
- Review: Mistral, triaged in [[LP-028 Findings of the review of P1]].
- Known limits, `@test_broken` and backlog row P21: a dual flagged message into a relation factor's
  flat-input path; `NonlinearFactor` at a dual expansion point.
- Verified: all eight suites passed twice, on the first version (logs `~/.cache/lenticulum/p1/run1/`)
  and with the F1 fix (`~/.cache/lenticulum/p1/`), and Mycelium's device tests passed with the fix.
  MyceliumLearning's device tests were not run.
- Not run: the docs build (no docstring changed).
- Next: commit and push when Daniel asks; then by the proposed order in [[Feature Backlog]]: P17
  and P18 (mechanical, suited to a helper), then P11 with F3, I6; or I16/I17.

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
