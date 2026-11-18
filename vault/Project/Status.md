#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-18, after the review of PR #23 (the Mistral cloud agent's GH-15–GH-22 commits): all accepted, one amendment (LP-055). I23 and I24 were rebased onto `9d94e548`, which the agent had pushed after them, and committed on `dev-danielb` and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | I23, I24 and the LP-055 amendment on top of the cloud agent's `9d94e548`; pushed. PR #23 (`dev-danielb` → `dev`) is open |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `3655c0c1` (I23) with the first I24 diff applied (without the LP-054 fix); the reviewer's probes in `scratch/i24/` (copied to `../Lenticulum.jl-Review-2026-11-17/i24/probes/`); its environment `scratch/p6env` serves any review. Before the next review: `git reset --hard` to the commit under review, then apply its diff |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, ADTypes,
DifferentiationInterface, then include the first ten lines of `runtests.jl` and the file (one exists: `~/.cache/lenticulum/env-p29`).

## Done last

- **CI** (2026-11-18): Julia 1.12, the oldest supported, no longer runs on every push; the core packages run on it weekly (`Weekly.yml`, job `oldest`: Mondays and on demand). `CI.yml` keeps the current release and the device tests.
- **Review of PR #23** ([[LP-055 Findings of the review of PR 23]]): the cloud agent's three commits
  (`74274922`, `407a4196`, `9d94e548`) were read and accepted. Amendment: the GH-21 `SampleBelief`
  moments refused tuple weights (AdversarialFactors LP-018.9 errored); they are now collected, and
  points are stacked with `reshape`/`reduce(hcat, …)`. The GH-15 test also checks the fixed point for root
  `:data`. Device tests (Mycelium) pass. GH-20 sends device and view inputs to the per-replica
  path too; this has not been benchmarked.
- **GitHub issues GH-15–GH-22** (a Mistral cloud agent, commits `74274922` and `407a4196` on `dev-danielb`; GH-n is GitHub issue #n; the backlog's I rows are a separate series): fixes for the eight open GitHub issues, each with a
  regression test and a note beside its source file. **GH-15** the tree shortcut of `propagate!` now
  requires `!message_uses_cavity(factor)` (new trait, declared by `NonlinearFactor`), so a nonlinear
  tree is iterated and its residual checked instead of reporting a one-sweep fixed point.
  **GH-16** the fixed noise is part of `m_step_signature`, so tied factors with different fixed noise
  are refused instead of pooled into a wrong M-step. **GH-17** rank tolerances use the
  decomposition's element type (`_eps_eltype`) in `_rank_deficient_svd`, `_range_svd`,
  `_projected_rank` and the hard-constraint rank test; Float32 maps are no longer misclassified.
  **GH-18** damping decides the improper flag from the combined support (`_damped_improper`), on CPU
  and on device; complementary nullspaces damp to a proper belief. **GH-19** the device diagonal
  moment methods go through `_improper_check` like the CPU ones. **GH-20** the batched per-replica
  constraint kernel runs only on plain host float arrays; dual coefficients take the per-replica
  path. **GH-21** `SampleBelief` moments are pure reductions (cat/matmul/broadcast), differentiable
  under ForwardDiff and Zygote, scalar-type preserving. **GH-22** `_has_m_step(f, ps, frozen)` queries
  `which` with the concrete call-site arguments, so `m_step` methods specialised on the parameter
  type are seen. Tests added: `sample_moments.jl` (GH-21), Mycelium `runtests.jl` + `device_improper.jl`
  (GH-18/GH-19), `m_step_detection.jl` (GH-22), Lenticulum `gh15_nonlinear_tree.jl`, `p29_rank_tolerance.jl`
  (GH-17), `plate_block_duals.jl` (GH-20), `em.jl` (GH-16). Not run here (no Julia): the suites of
  LenticulumCore, Mycelium, MyceliumLearning, Lenticulum and the weekly packages, plus the device
  tests. Follow-up noted in `passing.md`: `DEQFactor`/`DiffusionFactor` also read the prior in
  their messages and should declare `message_uses_cavity` when those packages are reviewed.
- **I24** (committed): under `criterion = :marginals` a `ResidualSchedule` whose `maxtasks` is below
  the number of tasks also needs every pending message residual within the bound when it stops
  (the sweep's residual filters, one scan of every task decides: `_max_pending`). Before, it stopped
  after 2 sweeps with trivial marginals (LP-052 F1). Rule chosen with the maintainer. Test: the I24
  lines of `test/p9_relative_tol.jl` (the former `@test_broken`), against the dense solve and a scan
  of the pending residuals after the report, warm starts, damping, relative criterion. Review:
  [[LP-054 Findings of the review of I24]] (Mistral Large, \$0: twelve probes, a finished answer; one
  defect, the filter alone left residuals up to 2.7 × `tol` at the report; reproduced and fixed).
- **I23** (committed, `3655c0c1`): iterated inference with ForwardDiff dual parameters; the report holds
  the primal residual, convergence and the residual schedule's choice of task are decided on it.
  [[LP-053 Findings of the review of I23]].
- Verified for I24: Mycelium, Lenticulum and AlgebraicFactors on the final code; MyceliumLearning,
  VariationalDiffusion, ImplicitLayers and AdversarialFactors on the first I24 version (the LP-054
  change is reached only by an explicit `:marginals` on a partial residual schedule, which no code
  outside `test/p9_relative_tol.jl` uses). Logs in `~/.cache/lenticulum/i24/`. Not run: device tests,
  the vault build and `docs/make.jl`.

## Next

No row is set; pick one with the maintainer. **I3** (the choice of spanning tree, alternating
trees; marked high). Close to P5: a learned noise with hard messages, a residual-space formulation
of the positive definite path (LP-047 finding 4; no row yet). Left open by P27 (no rows yet): the
Bethe free energy with a mixture marginal under BP, derivatives through the new rules, two extended
components with hard directions. Small: `ImplicitLayers`' `SolveReport` also has
`residual::Float64` (not checked whether duals reach it; that package is unreviewed);
`AndersonSchedule` runs without its memory under duals (LP-053).

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
  and probe 2's properness check found F1. On P3 the first run crashed inside vibe on its first command (a validation error of its own); the rerun reached 100 turns with only probe 1 in the answer file: run its later probes yourself. On P28 the same: four probes recorded, three more written but not recorded (all passed when run). On P9 (vibe 3.1.2, after it had failed on P29): five probes recorded, the sixth (the one with the findings) written and not recorded. On I22 (P6 brief form): nine probes, all recorded, a finished answer with "Not run" and "Unverified" sections, one right finding, \$0. On I23 (same form, `--max-price 3`): three probes, all recorded, one right finding, \$0; vibe exited 3 at its turn limit after writing the answer. On I24 (same form): twelve probes, a finished answer with tables, one right finding, \$0. Another user's `vibe-rs --continue` runs on this machine: wait on the PID of your own run, not `pgrep -x vibe-rs`.
