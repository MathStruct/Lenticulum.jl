#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-17, after I22 (`:marginals` needs two quiet sweeps), committed on `dev-danielb`, not pushed (everything up to P28 is pushed; P29, P9 and I22 are not).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | pushed up to P28 (`27770961`); P29, P9 and I22 committed on top, not pushed |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `5505123b` (P9) with the I22 diff applied; the reviewer's probes in `scratch/i22/` (copied to `../Lenticulum.jl-Review-2026-11-17/i22/probes/`); its environment `scratch/p6env` (packages developed from the worktree, plus ForwardDiff, Zygote, Test) serves any review. Before the next review: `git reset --hard` to the commit under review, then apply its diff |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, ADTypes,
DifferentiationInterface, then include the first ten lines of `runtests.jl` and the file (one exists: `~/.cache/lenticulum/env-p29`).

## Done last

- **I22** (committed): `propagate!(...; criterion = :marginals)` stops only after two sweeps in a row
  within the bound (`:messages` still after one). Flooding's second sweep from an empty store changes
  only messages to factors, so one quiet sweep stopped the loopy divider with every mean at the prior.
  Argument that two suffice under flooding: Mycelium `passing.md` "Convergence on marginals". The former
  `@test_broken` in `test/p9_relative_tol.jl` is its regression test. Cost: one more sweep per
  `:marginals` run.
- Reviews: [[LP-052 Findings of the review of I22]]. Mistral Large (\$0): nine probes and a finished
  answer file with the P6 brief form. Codex (`codex exec review --uncommitted`, premium again): no
  finding, no tests run. One defect reproduced that predates I22: **I24**, partial residual sweeps
  (`ResidualSchedule(maxtasks)` below the task count) still stop `:marginals` after 2 sweeps with
  trivial marginals; `@test_broken`, row only.
- Verified: all seven suites (Mycelium, MyceliumLearning, Lenticulum, VariationalDiffusion,
  ImplicitLayers, AdversarialFactors, AlgebraicFactors; logs in `~/.cache/lenticulum/i22/`), and the
  P9 file after the `@test_broken` was added (183 pass, 2 broken). Not run: device tests, the vault
  build and `docs/make.jl`.
- Not mine, left uncommitted: an edit to `AGENTS.md` (a Codex worktree `../Lenticulum.jl-codex`,
  branches `codex/<topic>`) appeared in the checkout during the session.

## Next

No row is set; pick one with the maintainer. Small: **I23** (an iterated `infer!` with dual
parameters throws in `ConvergenceReport`: report the primal value), **I24** (decide the rule for
partial residual sweeps). Close to P5: a learned noise with hard messages, a residual-space
formulation of the positive definite path (LP-047 finding 4; no row yet). Left open by P27 (no rows
yet): the Bethe free energy with a mixture marginal under BP, derivatives through the new rules, two
extended components with hard directions.

## Open

- **P4 limits:** Zygote has no rank check (a rank-changing tangent gives the constant-rank
  derivative silently); second derivatives (nested duals) through extended Gaussians are refused;
  `_moving_flat_split` (P25) stays ForwardDiff-only.
- **P25/P26 limits:** a parameter that changes a rank only at second order (a backward message's
  $A^\top R^{-1}A$ as $A$ leaves a rank-deficient value; $A_x$ leaving 0 in a cluster) gets the
  constant-rank derivative while the value is discontinuous there; Zygote does not go through the
  flat path's moving directions; the plate methods (`factor_messages` with a shared metric) and an
  inference with a dual $A$ upstream of a flat path are untested under moving flat directions;
  `cluster_marginal` of a flat class under ForwardDiff is untested.
- P24's Zygote rule gives the incoming messages no tangent (`@not_implemented`); differentiating
  through inference with cluster nodes stays closed (their messages are built by mutation).
- **Backlog section G** (decisions, control and games, G1–G10; 2026-11-16, from Daniel's question on
  reinforcement learning and open games): rows only, from a reading of code and notes. Nothing is
  designed or run; G5–G8 need P16. Priorities are proposals.
- Mycelium's `GaussianBlock` constructor and `belief_block` are not Zygote-safe (LP-044); a
  Zygote pass that builds blocks inside it fails there.
- AlgebraicFactors, ImplicitLayers and AdversarialFactors run in CI only weekly: a change to the core
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
  defect, then also stopped at 100 turns (\$2.35) before its summary. On I19 (Mistral Large, \$0) it wrote four
  probes but no answer file at all, and three of the probes had errors of their own (a misparsed
  tolerance, a parenthesis, a keyword read for a NamedTuple): run and repair its probes before
  believing a failure. On I20 it kept the answer file but was stopped by `--max-price 1` after 71
  turns: `cost_usd` counts list prices even for the free Mistral Large (3.9 M input tokens, \$1.01), so
  give a free-model review `--max-price 3` to reach its turn limit. On P20 (2026-11-16) both runs lost
  every command to a `cd … &&` prefix (not on the allowlist). On P23 the brief gave the exact
  command line to copy (`julia -t auto --project=<env> scratch/p23/probe1.jl > ….out 2>&1`) and it
  ran six probes; on P24 the command line without redirection and "never prefix `cd`": every probe
  ran and it finished with an answer file (\$0.41). Keep doing that. On P25 the commands ran but it
  spent its 100 turns on 26 debug variants and wrote one entry: run its last probes yourself. On P4
  (AD) no answer file at all; its last probe (needing imports added) found the one defect. Its
  probes need the package's test environment (`--project=lib/<Pkg>/test`); say so in the brief.
  A watcher `while pgrep -f "vibe -p …"` matches its own command line and never ends: match `vibe-rs` with `pgrep -x` or wait on the PID.
  On P6 (2026-11-17) the brief gave the exact command line, a ready environment (`scratch/p6env`),
  "never prefix `cd`" and the answer-file rule: ten probes, a finished answer file, three right
  findings, \$0. Keep that brief as the model (`../Lenticulum.jl-Review-2026-11-17/p6/review-brief.md`). On P27 the same brief form gave five probes and an answer
  file with four of them at the 100-turn limit (\$0): one right finding; its fifth probe's reference lost
  digits at 1e6 (a false report): check a reviewer's reference before its claim. On P5 the same brief: four probes, the answer file
  records only the first, no summary (\$0); the maintainer ran the rest (one probe needed a one-line repair)
  and probe 2's properness check found F1. On P3 the first run crashed inside vibe on its first command (a validation error of its own); the rerun reached 100 turns with only probe 1 in the answer file: run its later probes yourself. On P28 the same: four probes recorded, three more written but not recorded (all passed when run). On P9 (vibe 3.1.2, after it had failed on P29): five probes recorded, the sixth (the one with the findings) written and not recorded. On I22 (P6 brief form): nine probes, all recorded, a finished answer with "Not run" and "Unverified" sections, one right finding, \$0. Another user's `vibe-rs --continue` runs on this machine: wait on the PID of your own run, not `pgrep -x vibe-rs`.
