#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P24 (ForwardDiff and Zygote through a cluster node's free energy at fixed messages), committed and pushed (`26eede51`); AGENTS.md now has agents commit at the end of each task (`5ce11bc6`).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at `5ce11bc6` (pushed); P24 in `26eede51`: Mycelium `Project.toml` (ForwardDiff weakdep, new extension), `ext/MyceliumForwardDiffExt.jl` (new), `ext/MyceliumChainRulesCoreExt.jl`, `src/cluster.jl`/`.md`, `src/cluster_batch.jl`; Lenticulum `src/regime.jl`, `test/regime_plate.jl`; [[Feature Backlog]], [[LP-040 Findings of the review of P24]], Problem Log, this note |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `c2284852`, with the P24 diff applied; Mistral's probes in `scratch/p24/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-p13-applied-stale-2026-11-16`, `-p21-…`, `-p23-…`, `-p20-…`, `-p22-…`, `-i21-…`, `-i20-…`, `-i19-…`, older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P24** (committed and pushed, `26eede51`): the cluster joint is generic in its number type (`_ClusterJoint{T}`), so
  ForwardDiff takes a cluster node's free energy at fixed messages, the belief's dependence on `ps`
  included; the plate's stacked joint stays `Float64` and dual parameters go replica by replica
  (`_plain`). Zygote: an rrule on `_cluster_local_energy` (used by `ClusterFactor` and `RegimeFactor`)
  that runs ForwardDiff over the flattened parameters, in a new extension `MyceliumForwardDiffExt`
  (ChainRulesCore + ForwardDiff). On a plate row Zygote then stopped at the state write-back
  (`push!`); `_group_writeback` is `@non_differentiable`. A first fix that skipped writing empty
  states broke `state_tree.jl` and `lp020.jl` (callers read the key) and was reverted.
  [[lib/Mycelium.jl/src/cluster|cluster]] "Derivatives at fixed messages".
- Verified: the P24 testset of Lenticulum `test/regime_plate.jl` (36 pass, 1 broken: flat
  directions, P26; ForwardDiff against central differences, Zygote against ForwardDiff, the
  fixed-belief gradient shown to differ) and the MFA plate testset (the two former `@test_broken`).
  Suites (`~/.cache/lenticulum/p24/`): Mycelium, MyceliumLearning, Lenticulum pass on the final code
  (`suite3_*`); LenticulumCore, VariationalDiffusion, AlgebraicFactors, ImplicitLayers, Adversarial
  pass on the code before the `_group_writeback` rule (`suite_*`, `suite2_*`), which they do not
  reach through Zygote on plates. Not run: `docs/make.jl`, the device tests.
- Review: Mistral, $0.41, six probes, no defect ([[LP-040 Findings of the review of P24]]). Archive
  `../Lenticulum.jl-Review-2026-11-16/p24/`.
- Next: P25/P26 (derivatives on flat paths) or another backlog item.

## Open

- **P25:** ForwardDiff through the flat path when the flat directions move (a flat input whose precision is dual; backlog).
- **P26:** ForwardDiff and Zygote through a cluster node whose joint precision has flat directions (backlog).
- P24's Zygote rule gives the incoming messages no tangent (`@not_implemented`); differentiating
  through inference with cluster nodes stays closed (their messages are built by mutation).
- AlgebraicFactors, ImplicitLayers and Adversarial run in CI only weekly: a change to the core
  should run their suites locally too (that is how the AlgebraicFactors break was found).
- Mistral review agent, refused commands: on P13 its six refusals all began with `cd … &&`, and its
  `julia` commands without output redirection ran; on P21 the refused `julia` command had `> … 2>&1`
  (which ran on P23). Next brief: give the command without redirection and say "never prefix `cd`".
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
  ran six probes; on P24 the command line without redirection and "never prefix `cd`": every probe
  ran and it finished with an answer file ($0.41). Keep doing that.
