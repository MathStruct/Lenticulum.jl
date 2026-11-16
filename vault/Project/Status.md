#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P20 (Zygote through the free energy on plate block rows), not committed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at `f748da5f` with P20 uncommitted: `lib/Mycelium.jl/src/free_energy.jl`, `free_energy.md`, `passing.md`, `ext/MyceliumChainRulesCoreExt.jl`, Lenticulum `test/plates.jl`, [[Feature Backlog]]. Verified: see below |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `f748da5f` with the P20 diff applied; Mistral's (unparsable) probes in `scratch/p20/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-p22-applied-stale-2026-11-16`, `mistral-wt-i21-…`, `-i20-…`, `-i19-…` of 2026-11-15, `mistral-wt-ridge-applied-stale-2026-11-13` and older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P20** (not committed): `Zygote.gradient` of `scalar_free_energy` on a store with block rows.
  Four blockers, not one: `_row_scalar`'s `sum(...; init)`, the covered-node masks (`_mark!`,
  `@non_differentiable`), the variable rows' term (`@non_differentiable`, reads only blocks), and
  the factors' batched code (an `rrule` on `_row_scalar_energy` that differentiates the same row
  replica by replica through `local_free_energy`). It gives the derivative at fixed messages, as
  ForwardDiff does; an envelope-gradient first version was dropped because it differed from
  ForwardDiff by 0.78 (LP-020; [[free_energy]] "Zygote on block rows").
- Verified: Lenticulum `test/plates.jl` passes (P20 testset 12/12; the switching testset errored before the
  fix); all eight suites pass (`~/.cache/lenticulum/p20/suite_*.log`;
  in ImplicitLayers Mooncake's `MooncakeLogExpFunctionsExt` failed to precompile, a version mismatch
  in that environment, and the suite passed). Not run: `docs/make.jl`, device tests (no device code).
  `check_notes.py`: no new hits.
- Found: P23 (backlog): Zygote cannot take `LinearConstraintFactor`, EP-switching or cluster-node
  free energies even per edge; on a clustered regime plate ForwardDiff fails too (`Float64(::Dual)`).
  `@test_broken` for the first two.
- Review: Mistral, two runs, no probe executed: the first prefixed every command with `cd … &&`,
  which the review agent's allowlist refuses; the second, told so, still never ran `julia`, and its
  three probes do not parse. Static reading: no defect. Archive `../Lenticulum.jl-Review-2026-11-16/p20/`.
  No problem note (no findings).
- Next: commit P20 (when asked); then P21 (a design task), P23, or another backlog item.

## Open

- **P21:** ForwardDiff through a relation factor's flat-input path, and `NonlinearFactor` at a dual
  expansion point (`@test_broken` in Lenticulum `test/forms_crossing.jl`).
- **P23:** Zygote through constraint, EP-switching and cluster-node free energies (backlog).
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
  give a free-model review `--max-price 3` to reach its turn limit. On P20 (2026-11-16) both runs lost
  every command to a `cd … &&` prefix (not on the allowlist): consider allowing it in
  `~/.vibe/agents/review.toml` or starting the brief with an example command line.
