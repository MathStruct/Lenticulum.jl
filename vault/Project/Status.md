#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P25 (ForwardDiff through the relation factors' flat path when its flat directions move, at constant rank), committed on `dev-danielb` on top of P26, neither pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P26 (`8a6a9f41`) and P25 committed, not pushed. P25: Lenticulum `src/gaussian.jl`/`.md`, `src/constraint.jl`, `src/switching.jl`, `ext/LenticulumForwardDiffExt.jl`/`.md`, `test/forms_crossing.jl`; [[Feature Backlog]], [[LP-042 Findings of the review of P25]], Problem Log, this note |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `8a6a9f41`, with the first P25 diff applied (before the LP-042 fix); Mistral's probes in `scratch/p25/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-p26-applied-stale-2026-11-16`, `-p24-…`, `-p13-…`, `-p21-…`, `-p23-…`, `-p20-…`, `-p22-…`, `-i21-…`, `-i20-…`, `-i19-…`, older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P25** (committed): a flagged input whose precision is dual (a derivative by $A$ upstream), and
  dual coefficients or covariances on the flat path, under ForwardDiff at constant rank. P26's
  construction applied twice: the split of Λ (`_moving_flat_split`) and the complement of range(B)
  in the residual metric with its log pseudo-determinant (`_range_complement`, also used by the
  switching factor's `_residual_metric_logdet`), dual methods in `LenticulumForwardDiffExt`. A
  first-order rank change and nested duals are an `ArgumentError`. [[lib/Lenticulum.jl/src/gaussian|gaussian]]
  "Flat directions that move".
- Verified: `test/forms_crossing.jl` in full (the two P25 testsets: 281 checks against central
  differences of the value path, 200 refusals and identities; they error on the code before P25);
  the Lenticulum suite (`~/.cache/lenticulum/p25/suite_Lenticulum.log`). No other package changed
  (only `docs` and `bench` depend on Lenticulum). Not run: `docs/make.jl`, the device tests.
- Review: Mistral, $1.18, stopped at 100 turns with one entry; its probes were run here: one
  finding (nested duals in `_range_complement` gave a `MethodError`), fixed with a regression test
  ([[LP-042 Findings of the review of P25]]). Archive `../Lenticulum.jl-Review-2026-11-16/p25/`.
- Next: another backlog item; the derivative work on the flat path is complete up to the limits below.

## Open

- **P25/P26 limits:** a parameter that changes a rank only at second order (a backward message's
  $A^\top R^{-1}A$ as $A$ leaves a rank-deficient value; $A_x$ leaving 0 in a cluster) gets the
  constant-rank derivative while the value is discontinuous there; Zygote does not go through the
  flat path's moving directions; the plate methods (`factor_messages` with a shared metric) and an
  inference with a dual $A$ upstream of a flat path are untested under moving flat directions;
  `cluster_marginal` of a flat class under ForwardDiff is untested.
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
  ran and it finished with an answer file ($0.41). Keep doing that. On P25 the commands ran but it
  spent its 100 turns on 26 debug variants and wrote one entry: run its last probes yourself.
