#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-17, after P2 (hard inputs in the switching free energy, the frozen beliefs of the gradient and plates), committed on `dev-danielb`, not pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P2 committed (Lenticulum `switching.jl`/`.md`, `gaussian.jl`/`.md`, `test/hard_inputs.jl`; LenticulumCore `extended_gaussian.md`). Not pushed |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `01b0adea` with the P2 diff applied; Mistral's probes in `scratch/p2/` (copied to `../Lenticulum.jl-Review-2026-11-17/p2/`); its environment `scratch/p6env` (packages developed from the worktree, plus ForwardDiff, Zygote, Test) serves any review |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, then
`julia -t auto --project=<env> -e 'include("lib/Lenticulum.jl/test/<file>.jl")'`.

## Done last

- **P2** (committed): `switch_statistics` built on `_reduced_relation`, so `SwitchingConstraintFactor`'s
  term takes hard inputs; `_affine_gaussian_belief` substitutes hard directions and lifts the belief
  back, so `factor_belief` of `GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor` and the
  switching factor takes them, and the envelope gradient and the `DiagonalCoupling` replay run on
  graphs with hard relations (they refused, "no canonical form"). Plates needed only tests. Details in
  [[Backlog Done 2026-11]] (2026-11-17), [[switching]] "Hard inputs in the free energy",
  [[lib/Lenticulum.jl/src/gaussian|gaussian]] "Free energy with hard inputs".
- Review: [[LP-045 Findings of the review of P2]] (Mistral; no defect; a Zygote testset added for its gap).
- Verified: the suites of Lenticulum, AlgebraicFactors and ImplicitLayers pass (logs in
  `~/.cache/lenticulum/p2/`); the Zygote testset was added after the Lenticulum suite ran and passed
  on its own (`test/hard_inputs.jl` alone). LenticulumCore, Mycelium, MyceliumLearning and
  VariationalDiffusion not rerun (no code change there; one note in LenticulumCore). Vault build and
  `docs/make.jl` not run.

## Next

No row is set; pick one with the maintainer. Close to P2: P27 (mixture × extended, which would let a
switching factor with a latent outcome share a variable with a hard relation), P5 (a singular nonzero
noise in `HardConstraintFactor`'s free energy), a learned noise with hard messages (refused in
`GaussianFactor`/`GaussianPrior`), P3 (`TrivialBelief()` on the exact flat path).

## Open

- **P4 limits:** Zygote has no rank check (a rank-changing tangent gives the constant-rank
  derivative silently); second derivatives (nested duals) through extended Gaussians are refused;
  `_moving_flat_split` (P25) stays ForwardDiff-only. **P27:** mixture × extended product.
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
  findings, \$0. Keep that brief as the model (`../Lenticulum.jl-Review-2026-11-17/p6/review-brief.md`).
