#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after P11 (the query report), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the P11 commit, pushed; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | reset to `e55bf9af`, with the P11 diff applied for Mistral's review (before the LP-031 fixes), its scratch scripts in `scratch/p11/` and `scratch/f3/` (untracked). Stale stashes, all droppable: `mistral-wt-f3-roundtrip-applied-2026-11-12` (the reviewed `round_trip` diff, committed as `bd6778ee`), `mistral-wt-p17-18-stale-2026-11-12`, `mistral-wt-polarity-cache-draft-2026-11-12`, `mistral-wt-i18-predfix-2026-11-12` |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done today

- Committed and pushed earlier today: P1, P17/P18, I18, F3 in part (`round_trip`, `bd6778ee`).
- **P11, the query report**: `query_report(m, z₀, ρ, ps, st)` and `QueryReport`,
  `trusted` in VariationalDiffusion (`src/report.jl`, `src/report.md`; docs page `implicit.md`, "How
  far an answer can be trusted"); `_near_singular` is now shared with `implicit_pullback`. Testset
  "query report: …" (34 tests). Review: Mistral, four defects in the flags, all fixed with
  regression tests ([[LP-031 Findings of the review of the query report]]).
- Verified: the VariationalDiffusion suite 943/943 and Aqua 11/11 after the LP-031 fixes
  (`~/.cache/lenticulum/p11/vd-suite.log`), the docs build (`docs.log`; only the usual size warnings). Not run: device tests (no device code),
  the other suites (no other package changed), the calibration benchmark with the report.
- Next: I6 for path C, or the rest of F3 (a ridge residual: membership in the relation without the
  round trip's failure where the relation is parallel to the clamped coordinates).

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
