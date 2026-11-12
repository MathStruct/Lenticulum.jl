#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-13, after F3's ridge residual, implemented and reviewed, committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the I6 commit (`0ea55cab`, pushed), then the ridge residual commit, pushed; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `7277fe4f` with the ridge diff applied (a fast-forward to `0ea55cab` was refused by the stale untracked `lib/Lenticulum.jl/test/regime_plate.jl`, which I6 committed: delete it, then `git merge --ff-only dev-danielb`); Mistral's probes in `scratch/ridge/` and `scratch/i6/` (untracked). Stashes, all droppable: `mistral-wt-i6-applied-stale-2026-11-12` (I6 is committed), `mistral-wt-p11-applied-2026-11-12` and the four older ones of 2026-11-12 |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- Committed and pushed on 2026-11-12: P1, P17/P18, I18, F3 in part (`round_trip`), P11, I6 (`0ea55cab`).
- **F3, the ridge residual** (committed and pushed): `ridge_residual` in VariationalDiffusion `answers.jl`
  (the normal Newton step to the ridge from the field and the symmetric part of its full Jacobian;
  codimension defaults to the number of free coordinates), in `query_report` as the field `ridge`
  and the flag `:off_ridge` (`ridge_tol`; dense solves only), and a "ridge" filter in
  `examples/calibration.jl`. Notes: `implicit.md` §5, `report.md`, `calibration.md` "Remedies" point 4,
  the docs page `docs/src/packages/variationaldiffusion/implicit.md`, [[Feature Backlog]] (F3: an
  energy threshold left), [[LP-033 Findings of the review of the ridge residual]].
- Review: Mistral, this time with the answer written probe by probe as briefed (it finished, $0):
  no wrong result, four missing tests, each added and checked to fail on its mutant.
- Verified: the VariationalDiffusion suite passes after the review's tests
  (`~/.cache/lenticulum/f3/suite_vd2.log`); the calibration benchmark ran in full with the ridge
  filter (`calibration-ridge.log`, about 45 min; its "all roots" rows reproduce the earlier run).
  The docs build passes (`~/.cache/lenticulum/f3/docs.log`). Not run: other packages' suites (nothing else changed), device tests (no
  device code; Mistral probed a JLArray state).
- Next: I19 (profile the batched `RegimeFactor` plate). The rest of F3,
  an energy threshold, needs an energy network trained on the benchmark's relations.

## Open

- **P21:** ForwardDiff through a relation factor's flat-input path, and `NonlinearFactor` at a dual
  expansion point (`@test_broken` in Lenticulum `test/forms_crossing.jl`).
- **P20:** Zygote through a plated free energy fails on `sum(...; init)` (found by Mistral; the
  same before P7).
- AlgebraicFactors, ImplicitLayers and Adversarial run in CI only weekly: a change to the core
  should run their suites locally too (that is how the AlgebraicFactors break was found).
- Mistral review agent: `vibe ... --agent review --trust --workdir ../Lenticulum.jl-mistral`
  (`~/.claude/CLAUDE.md`). A review needed about 140 turns; give it 100 and continue the session
  with `vibe -c` if it stops. On I6 it kept probing through three continuations (about 175 turns,
  one session) and wrote no answer. Briefs now ask for an answer file appended after every probe and
  a stop by turn 70 (`~/.claude/CLAUDE.md`, "Reviews that never answer"); `--model glm-5-3` (priced, new
  sessions only) was tried on I6: it wrote its answer probe by probe as asked, five probes, no
  defect, then also stopped at 100 turns ($2.35) before its summary.
