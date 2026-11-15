#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-15, after P22 (a plate schedule on a store with another block numbering), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the P22 commit `2289cf0e` (pushed): `lib/Mycelium.jl/src/plates.jl`, `plates.md`, Lenticulum `test/regime_plate.jl`, [[Feature Backlog]]. verified (below) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `0a1ef31c` (I21) with the P22 diff applied; probes in `scratch/p22/`, older in `scratch/i21/` etc. (untracked). Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-i21-applied-stale-2026-11-15`, `mistral-wt-i20-applied-stale-2026-11-15`, `mistral-wt-i19-applied-stale-2026-11-15`, `mistral-wt-ridge-applied-stale-2026-11-13` and the older ones of 2026-11-12 |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P22** (committed and pushed): `sweep!` with a `PlateSchedule` checks once per sweep that the store numbers
  its block rows as `MessageStore(g)` does (`_graph_layout`), and otherwise runs the groups without
  their plans; before, a hand-built store with renumbered rows got $F$ = 2.68 against 14.54, silently
  ([[plates]] "A store with another block numbering"). Test: Lenticulum `test/regime_plate.jl`, "PlateSchedule
  on a store with another block numbering (P22)", which failed before the fix (4 failures, 2 errors) and passes after (14/14).
- Verified: all eight suites pass (`~/.cache/lenticulum/p22/suite_*.log`). Not run: `docs/make.jl`,
  `check_notes.py`, device tests (no device code).
- Review: Mistral, no defects (mostly code reading, no probe outputs recorded; archive
  `../Lenticulum.jl-Review-2026-11-15/p22/`). No problem note, as there are no findings.
- I21 (before): committed and pushed; see [[Feature Backlog]] "Done".
- Next: P20 (Zygote through `sum(...; init)` in `_row_scalar`, probably
  small), or P21 (a design task), or another backlog item.

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
  defect, then also stopped at 100 turns ($2.35) before its summary. On I19 (Mistral Large, $0) it wrote four
  probes but no answer file at all, and three of the probes had errors of their own (a misparsed
  tolerance, a parenthesis, a keyword read for a NamedTuple): run and repair its probes before
  believing a failure. On I20 it kept the answer file but was stopped by `--max-price 1` after 71
  turns: `cost_usd` counts list prices even for the free Mistral Large (3.9 M input tokens, $1.01), so
  give a free-model review `--max-price 3` to reach its turn limit.
