#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-19, after P10 (noise-free nodes against nodes with noise on the calibration benchmark: `noisefree_nodes(n; levels)` and `examples/nodes_calibration.md`; `noisefree_nodes(n)` is now the recommended node set for root queries), committed on `dev-danielb`, not pushed. Before it P36 (each package's own suite reaches its own files), P35 (`MixtureProx` takes a modified Newton step where its curvature is indefinite), P15, P34, P12 and P33 (a Lenticulum graph is Float32 end to end).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | I23, I24 and the LP-055 amendment on top of the cloud agent's `9d94e548`; pushed. PR #23 (`dev-danielb` → `dev`) closed unmerged on 2026-11-18; `dev` is 22 commits behind and can be fast-forwarded |
| `~/Code/CategoryTheory/Lenticulum.jl-codex` | `codex/p36` | at `9f933eb7` with the P36 diff uncommitted (merged into `dev-danielb` with one more testset; diff in `~/.cache/lenticulum/p36/p36.diff`); reset it before the next Codex package |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `9f933eb7` with the P36 diff applied (without the session's last testset); the test environments of LenticulumCore, Mycelium and MyceliumLearning instantiated (`--project=lib/<Pkg>.jl/test` serves probes); probes in `scratch/p36/` (copied to `../Lenticulum.jl-Review-2026-11-19/p36/mistral/`); `scratch/p6env` serves any review, and `~/.cache/lenticulum/env-p29` serves as the old code for a bit-for-bit comparison. Before the next review: reset to the commit under review, then apply its diff |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, ADTypes,
DifferentiationInterface, then include the first ten lines of `runtests.jl` and the file (one exists: `~/.cache/lenticulum/env-p29`).

## Done last

- **P10** (2026-11-19, chosen by the maintainer from the three Section P rows): `noisefree_nodes(n; levels)`
  (no noise, one node per level, `field_nodes`' levels by default; argument checks) and
  `lib/VariationalDiffusion.jl/examples/nodes_calibration.jl` (the models of `calibration.jl`, whose setup
  is now `calibration_setup.jl`, skipped by `bench/smoke.jl`). Measured in `nodes_calibration.md`:
  noise-free recall as high or higher in eight of nine settings, less bias on the circle (43–46 % at
  $t\le0.05$), larger on the two circles at $t\le0.05$, 4–25× faster. Recommended for root queries in the
  docstrings and `implicit.md`; no constructor default changed (`ProxDM`'s free-energy nodes stay
  `field_nodes`, not measured). Review [[LP-068 Findings of the review of P10]] (Codex, read-only, ten
  findings, all right; argument checks, `missing` for skipped verdicts, the test's scope, the notes'
  overstatements). VariationalDiffusion suite passed on the final code
  (`~/.cache/lenticulum/p10/vd-suite-final.log`); both calibration scripts pass at smoke size; benchmark log
  `~/.cache/lenticulum/p10/nodes-full.log`. Not run: the docs builds (docstrings changed), the full
  `calibration.jl` after the refactor (smoke only; Codex compared the smoke outputs before and after:
  the same up to timings), one training seed per relation.
- **P36** (2026-11-19, chosen as the only Section P row ready to start): tests in the owning package for
  the files only other packages' tests reached: Mycelium `test/p36_exposure.jl`, `test/p36_overlay_ad.jl`
  (exposure, overlay, the cluster Zygote rule, ForwardDiff through flat precisions), MyceliumLearning
  `test/sample_gradient.jl`, LenticulumCore `test/statistical_game.jl`, `test/interop.jl` (the
  ExponentialFamily checks moved there from Lenticulum's `runtests.jl`; new test dependencies). Weekly job
  `reactant` in `Weekly.yml` (**not yet run on GitHub**; start it by hand after the next push, with
  `Examples.yml` and `coverage`). Coverage of those files by their own suite now 89–100 %. Codex
  implemented (one run); review [[LP-067 Findings of the review of P36]] (Mistral, \$0, by mutation: three
  missing overlay checks, closed by a testset of the session that fails under each mutation). No source
  defect. **Stale local Manifests**: the main checkout's untracked Manifests predated the newest
  extensions, so Mycelium's local suite had never loaded its ForwardDiff extensions; `Pkg.resolve()` in
  every package fixed six of eight (LP-067). Suites passed on the final code after that: LenticulumCore,
  Mycelium, MyceliumLearning, Lenticulum (`~/.cache/lenticulum/p36/gates2/`; Mycelium's final run
  `mycelium-final.log`). A first run of Mycelium beside two other suites failed the timing check
  `state_tree.jl:131` (`s4 < 8s1`, 5.5 ms against 4.2 ms): load-sensitive, passes alone; no row. Not run:
  device tests, the docs builds, VariationalDiffusion and the weekly packages (no file of theirs changed
  apart from the CI job; their Manifests were resolved).

## Next

No row is set; pick one with the maintainer. Section P has P14 (shared solver layer, large: design first)
and P16 (owned scalarisations: design first). Open after P10: the tutorials and the README still use
`field_nodes` for root queries; whether to move them (their recorded outputs would change), and
`ProxDM`'s free-energy nodes with noise-free nodes (not measured). Close to P5: a learned noise with hard messages, a residual-space formulation of the
positive definite path (LP-047 finding 4; no row yet). Left open by P27 (no rows yet): the Bethe free energy
with a mixture marginal under BP, derivatives through the new rules, two extended components with hard
directions. Small: `ImplicitLayers`' `SolveReport` also has `residual::Float64` (not checked whether duals
reach it; that package is unreviewed); `AndersonSchedule` runs without its memory under duals (LP-053). Not
benchmarked yet: GH-20's per-replica path for device and view inputs to the hard-constraint plate kernel.

## Open

- **Local Manifests** are untracked and go stale: after adding a weak dependency, an extension or a test
  dependency, run `Pkg.resolve()` in the package (and `--project=lib/<Pkg>.jl/test`), or its suite runs
  without the new extension (LP-067). Other lower coverage left by the first report (no row): Mycelium
  `anderson.jl` 29 %, `blocks.jl` 55 %, `plates.jl` 61 %; the device extensions run only with the device tests.
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
  and probe 2's properness check found F1. On P3 the first run crashed inside vibe on its first command (a validation error of its own); the rerun reached 100 turns with only probe 1 in the answer file: run its later probes yourself. On P28 the same: four probes recorded, three more written but not recorded (all passed when run). On P9 (vibe 3.1.2, after it had failed on P29): five probes recorded, the sixth (the one with the findings) written and not recorded. On I22 (P6 brief form): nine probes, all recorded, a finished answer with "Not run" and "Unverified" sections, one right finding, \$0. On I23 (same form, `--max-price 3`): three probes, all recorded, one right finding, \$0; vibe exited 3 at its turn limit after writing the answer. On I24 (same form): twelve probes, a finished answer with tables, one right finding, \$0. Another user's `vibe-rs --continue` runs on this machine: wait on the PID of your own run, not `pgrep -x vibe-rs`. On P33 (P6 brief form, `--max-price 3`): four probes, all recorded, no summary (exit 3 at the turn limit), \$0; the session ran the open edge itself. On P30 (P6 brief form, `--max-price 3`): five probes, three recorded, no summary at the turn limit (\$0); of the two unrecorded, one needed an import repaired (`SpecialFunctions` is not in `p6env`: use `Lenticulum.SpecialFunctions`) and showed the one defect. Its old-versus-new bitwise comparison (`probe2_f64_same.jl` with `scratch/p30/oldenv`) is reusable.
