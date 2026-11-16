#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P21 (ForwardDiff through the flat path with only η dual), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the P21 commit `a22157e1` (pushed): LenticulumCore `extended_gaussian.jl`/`.md` (`_flat_split`); Lenticulum `gaussian.jl`/`.md`, `nonlinear.jl`/`.md`, `constraint.jl` (docstring); `test/forms_crossing.jl`; [[Feature Backlog]] (P21 done, P25 new), [[LP-038 Findings of the review of P21]], Problem Log. Verified: see below |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `5eac9e5e`, with the P21 diff applied (before the switching testset); Mistral's probes in `scratch/p21/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-p23-applied-stale-2026-11-16`, `-p20-…`, `-p22-…`, `-i21-…`, `-i20-…`, `-i19-…`, older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P21** (committed and pushed, `a22157e1`): ForwardDiff through a relation factor's flat path when only η is dual (a
  derivative by `b`, `c` or an observation). A flagged input of duals is split on its precision's
  values (`_improper_split_dual` → `LenticulumCore._flat_split`, which `extended` now uses too); a
  precision of zero-partial duals is accepted, one with nonzero partials throws an `ArgumentError`
  (P25, `@test_broken`). `_jacobian` is generic in the element type (`NonlinearFactor` at a dual
  expansion point). Found on the way: `norm` of duals whose value is exactly 0 is `NaN`, which made
  the split's range check fail; it reads values now. [[gaussian]] "Dual inputs on the flat path".
- Verified: `test/forms_crossing.jl` run alone after the last edit, all testsets pass (P21: 614 + 271 +
  30 + 160 checks, one `@test_broken`), `~/.cache/lenticulum/p21/fc.out`. The eight suites
  (`~/.cache/lenticulum/p21/suite_*.log`) all pass; they ran before the switching testset (test
  only) and the note and docstring edits were added, so the Lenticulum suite has not run with the
  final `forms_crossing.jl` as part of `Pkg.test` (the file alone has). `check_notes.py`: no new
  hits. Not run: `docs/make.jl`, device tests (no device code).
- Review: Mistral, $1.12, ran **no** probe (its julia commands were refused by the approval
  callback, although the brief had the exact command line that worked on P23). Its seven probes
  were run by hand: every failure was the probe's own error; the gaps it named passed in a repair
  probe and two became tests ([[LP-038 Findings of the review of P21]]). Archive
  `../Lenticulum.jl-Review-2026-11-16/p21/`.
- Next: P24 (cluster nodes under ForwardDiff/Zygote, a design task) or
  P25, or another backlog item.

## Open

- **P25:** ForwardDiff through the flat path when the flat directions move (a flat input whose precision is dual; backlog).
- **P24:** ForwardDiff and Zygote through cluster-node free energies at fixed messages (backlog).
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
  every command to a `cd … &&` prefix (not on the allowlist). On P23 the brief gave the exact
  command line to copy (`julia -t auto --project=<env> scratch/p23/probe1.jl > ….out 2>&1`) and it
  ran six probes: keep doing that.
