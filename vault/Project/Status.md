#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P26 (ForwardDiff and Zygote through flat directions of a cluster's joint precision, at constant rank), committed on `dev-danielb`, not pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P26 committed on top of `73277da7` (not pushed): Mycelium `Project.toml` (extension `MyceliumForwardDiffFlatExt`, ForwardDiff alone), `ext/MyceliumForwardDiffFlatExt.jl` (new), `src/cluster.jl`/`.md`; Lenticulum `test/regime_plate.jl`; [[Feature Backlog]], [[LP-041 Findings of the review of P26]], Problem Log, this note |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `73277da7`, with the P26 diff applied; Mistral's probes in `scratch/p26/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout); both re-resolved for the new extension (an environment resolved before it does not load it: `Pkg.resolve()`). Stashes, all droppable: `mistral-wt-p24-applied-stale-2026-11-16`, `-p13-…`, `-p21-…`, `-p23-…`, `-p20-…`, `-p22-…`, `-i21-…`, `-i20-…`, `-i19-…`, older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P26** (committed): a cluster's joint precision with flat directions under ForwardDiff (and so
  Zygote, whose rule runs ForwardDiff), at constant rank. The range and null space at the value move
  only into each other ($U + \varepsilon NN^\top DU\Lambda^{-1}$, $N - \varepsilon U\Lambda^{-1}U^\top DN$) and $K = U^\top JU$ is
  factorised in duals: no eigenvector is differentiated. `_Flat` holds `K` (a `Diagonal` for floats) instead
  of `λ`; `_psd_factor` hands a singular J to `_flat_factor`, whose dual method is in the new extension
  `MyceliumForwardDiffFlatExt`. A tangent with $N^\top DN \ne 0$ (rank change to first order) is refused;
  nested duals are refused. [[lib/Mycelium.jl/src/cluster|cluster]] "Flat directions under a derivative".
- Verified: the P26 testset of Lenticulum `test/regime_plate.jl` (58 pass: twenty random constant-rank
  J(θ), half with repeated eigenvalues, against central differences; a rank change refused; five regime
  factors with flat x, ForwardDiff against central differences and Zygote against ForwardDiff; P24's
  former `@test_broken`) and the P24 testset (35 pass). Suites (`~/.cache/lenticulum/p26/suite_*.log`):
  Mycelium, MyceliumLearning, Lenticulum, VariationalDiffusion, AlgebraicFactors, ImplicitLayers,
  Adversarial pass (LenticulumCore is unchanged and was not run). Not run: `docs/make.jl`, the device tests.
- Review: Mistral, $0.66, eight probes, no defect; probe 1 asserted nothing ([[LP-041 Findings of the review of P26]]).
  Archive `../Lenticulum.jl-Review-2026-11-16/p26/`.
- Next: P25 (the relation factor's flat path when the flat directions move), which can reuse P26's
  constant-rank construction for its eigendecomposition of Λ; or another backlog item.

## Open

- **P25:** ForwardDiff through the flat path when the flat directions move (a flat input whose precision is dual; backlog).
- **P26 limits:** a parameter that makes a flat direction determined only at second order (A_x
  leaving 0) gets the constant-rank derivative while the value is discontinuous there (`cluster.md`);
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
  ran and it finished with an answer file ($0.41). Keep doing that.
