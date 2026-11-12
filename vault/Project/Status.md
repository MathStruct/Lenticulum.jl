#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after F3 in part (`bd6778ee`, pushed).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at `bd6778ee` (F3 in part), pushed; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `883f8655`, with the `round_trip` diff applied for Mistral's review (before the LP-030 fix), its scratch scripts in `scratch/f3/` (untracked). Three stale stashes, all droppable: `mistral-wt-p17-18-stale-2026-11-12` (an older copy of P17/P18), `mistral-wt-polarity-cache-draft-2026-11-12`, `mistral-wt-i18-predfix-2026-11-12` |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done today

- Committed and pushed earlier today: P1 (`597dc8e2`), P17/P18 (`fbbb6cdf`), the rest of I18.
- **F3 in part** (`bd6778ee`): `round_trip(m, sol, z₀, ρ, ps, st)` in VariationalDiffusion
  (`answers.jl`, exported; `implicit.md` §5), with the testset "round trip: …"; the calibration
  benchmark measures the round trip, `coverage` and annealing as filters, its queries run on all
  threads, `CALIBRATION_SMOKE=1` runs it small (`examples/calibration.md` "Remedies"). Review:
  Mistral, one defect fixed ([[LP-030 Findings of the review of round_trip]]); LP-028 and LP-029
  added to the Problem Log, which lacked them.
- Verified: VariationalDiffusion suite 909/909 and Aqua (`~/.cache/lenticulum/f3/vd-suite.log`), the
  benchmark (`~/.cache/lenticulum/f3/calibration.log`), the docs build. Not run: device tests (no device
  code), the other suites (no other package changed).
- Next: P11 (the query report), built on `round_trip` and `coverage`,
  or a ridge residual (the rest of F3: the round trip fails where the relation is parallel to the
  clamped coordinates); I6 for path C.

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
