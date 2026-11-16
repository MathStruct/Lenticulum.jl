#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P13 (the exactness of a tree sweep checked at run time), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the P13 commit `e7ddaf70` (pushed): Mycelium `passing.jl`/`.md`, `factors.jl`, `cluster.jl`, `Mycelium.jl` (export), `test/runtests.jl`; Lenticulum `Lenticulum.jl` (declarations), `test/exact_report.jl` (new), `test/gaussian_oracle.jl`, `test/runtests.jl`; [[Bethe Free Energy]], [[Feature Backlog]], [[LP-039 Findings of the review of P13]], Problem Log |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `86da7a51`, with the P13 diff applied; Mistral's probes in `scratch/p13/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-p21-applied-stale-2026-11-16`, `-p23-…`, `-p20-…`, `-p22-…`, `-i21-…`, `-i20-…`, `-i19-…`, older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P13** (committed and pushed, `e7ddaf70`): `ConvergenceReport` has a field `exact`. After the one sweep of an undamped
  tree schedule it is set only when every node that sends a message declares
  `Mycelium.exact_messages` (opt-in; declared for Mycelium's data, prior, relay, table and cluster
  factors and Lenticulum's Gaussian, prior, linear/hard constraint and switching factors), none is a
  `MixtureReduction` node, and no factor received a mixture on one edge while sending on another;
  `reason` names each. Before, a switching chain under belief propagation was reported "exact" while
  its marginals were up to 0.011 off enumeration. [[passing]] "Exactness checked at run time".
- Verified: `test/exact_report.jl` (99 checks, each claim against enumeration or a closed form) and
  `gaussian_oracle.jl` (`rep.exact` on 40 random trees) pass run alone (`~/.cache/lenticulum/p13/run.out`);
  all eight suites pass (`~/.cache/lenticulum/p13/suite_*.log`; the Lenticulum suite ran the P13
  testset, 99 checks). Not run: `docs/make.jl`.
- Review: Mistral, $0.98, four probes ran; one finding (wrappers not declared exact), intended, no
  change ([[LP-039 Findings of the review of P13]]). Archive `../Lenticulum.jl-Review-2026-11-16/p13/`.
- P21 before it: committed and pushed (`a22157e1`), [[LP-038 Findings of the review of P21]].
- Next: P24 (cluster nodes under ForwardDiff/Zygote, design: make the
  cluster joint generic in its element type and non-mutating) or another backlog item.

## Open

- **P25:** ForwardDiff through the flat path when the flat directions move (a flat input whose precision is dual; backlog).
- **P24:** ForwardDiff and Zygote through cluster-node free energies at fixed messages (backlog).
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
  ran six probes: keep doing that.
