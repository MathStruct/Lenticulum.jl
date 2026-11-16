#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P23 (Zygote through the constraint and switching cavity free energies), not committed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at `b184cc82` (pushed) with P23 uncommitted: LenticulumCore `gaussian_belief.jl`/`.md` and its ChainRulesCore ext; Lenticulum `constraint`, `switching`, `gaussian`, `exposure`, `hard_constraint`, `likelihood`, `nonlinear`, `Lenticulum.jl`, its ChainRulesCore ext; notes `constraint.md`, `switching.md`, Mycelium `free_energy.md`, `cluster.md`; tests `zygote_cavity.jl` (new), `plates.jl`, `regime_plate.jl`, `runtests.jl`; [[Feature Backlog]], [[LP-037 Findings of the review of P23]]. Verified: see below |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `b184cc82`, with the P23 diff applied (without the `likelihood.jl` fix); Mistral's probes in `scratch/p23/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-p20-applied-stale-2026-11-16`, `-p22-…`, `-i21-…`, `-i20-…`, `-i19-…`, older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P23** (uncommitted): `Zygote.gradient` in `ps` of the cavity free energies of
  `LinearConstraintFactor` and `SwitchingConstraintFactor` (EP sites), per node and through
  `scalar_free_energy`, per edge and on plate blocks. Five blockers: in-place `_blockdiag` (now
  `cat`), `inv(::Cholesky)` (now `LenticulumCore._spd_inv` with an rrule, symmetric part; also in
  `Gaussian(μ, Σ)`), `push!` in `_reduced_relation` (tuples) and `switch_statistics`
  (`_free_channels`, non-differentiable), and the SVD in `_complement` (non-differentiable; messages
  are constants in `ps`). [[constraint]] "Zygote through the cavity free energy".
- Verified: all eight suites pass (`~/.cache/lenticulum/p23/suite_*.log`); Lenticulum rerun after the
  review fix: `suite_Lenticulum_2.log`. New `test/zygote_cavity.jl` (random factors and messages,
  Zygote = ForwardDiff); the two P23 `@test_broken` in `plates.jl` are tests now. Not run:
  `docs/make.jl`, device tests (no device code), Mooncake/Enzyme on the new rrule. `check_notes.py`:
  no new hits.
- Split off: **P24**, ForwardDiff and Zygote through a cluster node's free energy at fixed messages
  (`Float64` arrays and mutation in `cluster.jl`/`cluster_batch.jl`); a design task,
  `@test_broken` in `regime_plate.jl`.
- Review: Mistral, one run, $1.28, six probes **executed** (the brief's exact command line worked).
  One finding, an `inv(_chol(...))` my regex missed in `likelihood.jl`; replaced, its impact not
  reproduced ([[LP-037 Findings of the review of P23]]). Archive `../Lenticulum.jl-Review-2026-11-16/p23/`.
- Next: commit P23 when asked; then P21 or P24 (both design tasks), or another backlog item.

## Open

- **P21:** ForwardDiff through a relation factor's flat-input path, and `NonlinearFactor` at a dual
  expansion point (`@test_broken` in Lenticulum `test/forms_crossing.jl`).
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
