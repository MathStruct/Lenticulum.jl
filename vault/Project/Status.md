#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-19, after P34 (the default output coefficient `I` takes the parameters' number type at use), committed on `dev-danielb`, not pushed. Before it the P33 benchmark, P12 and P33, which finished the Float32 work: a Lenticulum graph is Float32 end to end (stopping tolerances and comparison scores stay Float64: `~/.cache/lenticulum/p33/remaining-float64.md`); since P34 also with the default `noise` of a `RegimeFactor` with `learn_noise = true`.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | I23, I24 and the LP-055 amendment on top of the cloud agent's `9d94e548`; pushed. PR #23 (`dev-danielb` → `dev`) closed unmerged on 2026-11-18; `dev` is 22 commits behind and can be fast-forwarded |
| `~/Code/CategoryTheory/Lenticulum.jl-codex` | `codex/p33` | at `d3adbd6e` with the P33 diff uncommitted (merged into `dev-danielb` as a diff; the P32 worktree diff saved to `~/.cache/lenticulum/p32/codex-worktree-final.diff`); reset it before the next Codex package |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `d3adbd6e` with the P33 diff applied; probes in `scratch/p33/` (copied to `../Lenticulum.jl-Review-2026-11-19/p33/probes/`); `scratch/p6env` serves any review, and `~/.cache/lenticulum/env-p29` (the main checkout's packages) serves as the old code for a bit-for-bit comparison before a commit. Before the next review: reset to the commit under review, then apply its diff |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, ADTypes,
DifferentiationInterface, then include the first ten lines of `runtests.jl` and the file (one exists: `~/.cache/lenticulum/env-p29`).

## Done last

- **P34** (2026-11-19, the next row; benchmarks allowed by the maintainer): the default `output_coeff = I`
  of the three constraint factors is stored as `Matrix{Bool}` and converted at use (`_coeffs(f, ps)`,
  `_fixed_at`) to the floating promotion of `ps.A` and the offsets; other integers and floating
  coefficients keep the rule before. Test `lib/Lenticulum.jl/test/p34_output_coeff.jl` (66 checks: types,
  Float64 bit for bit against an explicit Float64 identity, Zygote, ForwardDiff, Float32 EM on the
  benchmark's plate with the default noise). Review [[LP-064 Findings of the review of P34]] (Codex
  read-only; integers rounded in Float32 under a tie: reproduced, fixed by keeping only Booleans exact).
  Bench (`bench/float32_cluster_plates.jl`, new plates `reg1d`, `reg1w`; log `~/.cache/lenticulum/p34-bench/`):
  the default-noise plate is now Float32 with the free energy of the `noise = T(0.1)` plate bit for bit,
  same time as the widened one, 11 % less allocated. Lenticulum suite passed on the final code
  (`~/.cache/lenticulum/p34/lenticulum-suite-final.log`); no package depends on Lenticulum. Not run:
  device tests, the docs builds. Still widening, by design: a fixed noise `Q` (constraint and switching
  factors, `RegimeFactor` without `learn_noise`) keeps the constructor's type.
- **P33 benchmark** (2026-11-19, benchmarks allowed by the maintainer): `bench/float32_cluster_plates.jl`,
  `fit_em` (10 iterations) on two regime plates, Float64/Float32, stacked against forced per replica (the
  pre-P33 Float32 path). A Float32 plate is now 14–33 times faster than before P33 and as fast as Float64
  (0.94–1.01), 11–23 % less allocated; free energies agree (Float32/Float64 to 5e-7 relative). Recorded in
  Mycelium `cluster.md` "Float32 plates, measured" (logs `~/.cache/lenticulum/p33-bench/`). Found on the way:
  `RegimeFactor`'s default `noise = 0.1` makes its output coefficient Float64 and widens a Float32 plate
  even with `learn_noise = true`; new row **P34**. Bench and notes only, so no suite was run and there was no review.
- **P12** (2026-11-19, chosen by the maintainer): `ImplicitProx(nodes)` leaves `λ = nothing`; the
  `DiffusionFactor` constructor sets `density_lambda(schedule, nodes)` for `:gaussian`/`:mixture`
  messages and 1 for `:point` (point answers unchanged); a factor built field by field resolves it at
  use (`_resolve_lambda`). Explicit λ kept, still warned when off by more than 2. Test
  `lib/VariationalDiffusion.jl/test/p12_default_lambda.jl` (36 checks; closed-form precision on the
  Gaussian oracle). The Krylov test of `runtests.jl` compared a `:point` with a `:gaussian` factor under
  soft evidence; both are `:gaussian` now (their default λ differs otherwise). Review
  [[LP-063 Findings of the review of P12]] (Codex read-only; one type instability, fixed with an
  `@inferred` test). VariationalDiffusion suite passed on the final code
  (`~/.cache/lenticulum/p12/vd-suite-final.log`); no other package depends on it. Not run: the docs
  builds (the `ImplicitProx` docstring changed).
- **P33** (2026-11-19): the stacked path of cluster plates (`cluster_batch.jl`) serves any plain
  floating type (`FrozenClusterBeliefs{T}`), so Float32 plates are stacked; duals and replicas of
  differing joint types go replica by replica. `_flatten_active` drops empty parts, so a fixed factor's
  empty subtree no longer widens Float32 accelerated EM. Sample reweighting keeps its inputs' type and is
  now Zygote-differentiable. Codex implemented (one run, logs copied to `~/.cache/lenticulum/p33/`); the
  session made the block check read the first column only. Review [[LP-062 Findings of the review of P33]]
  (Mistral, \$0, four probes, no defect; one more probe by the session). Float64 bit for bit unchanged
  against `d3adbd6e` (Codex's script, rerun on the final code: 52 lines identical; the reviewer's: 128).
  Suites passed on the final code: Mycelium, MyceliumLearning, Lenticulum, VariationalDiffusion,
  AlgebraicFactors, ImplicitLayers, AdversarialFactors (`~/.cache/lenticulum/p33/gates/`); LenticulumCore
  and the Mycelium/MyceliumLearning device tests passed on Codex's version, before the block-check change
  (which touches neither). Not run: the docs builds; the speed of the stacked Float32 path (benchmarks
  deferred).
- **P32** (2026-11-18): Mycelium's cluster joints (number type = floating promotion of forms and
  messages, no Float64 floor), the Gaussian overlaps and mixture products, and MyceliumLearning's
  `MomentStats{T}`/`fit_em` history keep the element type of their inputs; a Lenticulum graph, regime
  and EM included, is Float32 end to end (the 128 `@test_broken` of `p31_element_type.jl` pass). Codex
  implemented (one run; its sandbox could not write `~/.cache`, its logs went to `/tmp/lenticulum-p32`,
  copied to `~/.cache/lenticulum/p32/`). Defect found by the session and the review alike: the flat/range
  thresholds of cluster joints were Float64's; now scaled with eps(T) (`_flat_rtol`; Float32 joints of
  condition above ≈ 3.5e4 are flat: `cluster.md` "Thresholds"). Review [[LP-061 Findings of the review of P32]]
  (Mistral, \$0; its EM "divergence" was a degenerate 12-point problem; with 60 points Float32 tracks
  Float64 to 7e-7, now a test). Float64 bit for bit unchanged against `cb289789` (Codex's script and the
  flat probe). Suites passed on the final code: Mycelium, MyceliumLearning, Lenticulum, VariationalDiffusion,
  AlgebraicFactors, ImplicitLayers, AdversarialFactors, Mycelium and MyceliumLearning device tests (logs
  `~/.cache/lenticulum/p32/gates/`). The Lenticulum suite run may have started before the last test
  ("P32: Float32 EM on a regime plate") was appended; that testset was run alone and passed. New row
  **P33** (cluster batch path, accelerated EM, sample reweighting still Float64). Not run: the docs builds.
- **P31** (2026-11-18): the rest of Lenticulum (switching, regime, nonlinear, likelihoods, M-steps, the
  hard-constraint plate kernel, exposure, coupling) keeps the element type of its parameters and data;
  constants cast at use, nothing else converted. Codex implemented (one run); test
  `lib/Lenticulum.jl/test/p31_element_type.jl` (1347 pass, 128 broken, all on Mycelium/MyceliumLearning
  sources: new row **P32**). Review [[LP-060 Findings of the review of P31]] (Mistral, \$0; recorded only its
  baseline, the session ran its two probes): no defect; Float64 results bit for bit unchanged against
  `6157f273`; ForwardDiff now goes through `LikelihoodFactor` (it threw before). Lenticulum suite passed
  (Codex's run on the same diff, log `~/.cache/lenticulum/p31/lenticulum-suite.log`); no other package
  changed and none depends on Lenticulum. Not run: device tests, the docs builds.
- **P30** (2026-11-18): with Float32 parameters, noise, priors and data, Lenticulum's
  `GaussianFactor`, `GaussianPrior`, `LinearConstraintFactor` and `HardConstraintFactor` send Float32
  messages and give Float32 (or `false`) energies, also extended messages, plates, mean field and
  learnable noise; promotion decides, nothing is converted. Codex implemented the Lenticulum part
  (one run); its test left 218 checks broken on Float64 sources in LenticulumCore's extended-Gaussian
  algebra and Mycelium's `variable_corrections` (`-false === 0`), which the session fixed. Test
  `lib/Lenticulum.jl/test/p30_element_type.jl` (2124 checks). Review [[LP-059 Findings of the review of P30]]
  (Mistral, \$0; one defect, an Int Wishart `df`, fixed). Float64 results bit for bit unchanged
  against `3d0bc449`. All eight suites and the Mycelium device tests passed (logs
  `~/.cache/lenticulum/p30/gates/`; Lenticulum rerun on the final code, `Lenticulum-final.log`). New
  row **P31**: the rest of Lenticulum (nonlinear, switching, regime, likelihoods, M-steps) is still Float64.
- **P8** (2026-11-18, chosen by the maintainer): Mycelium's graph free energy keeps the element type
  of its terms (sums from `false`, exact zeros as `false`, `log(2π)` via `LenticulumCore._log2π(T)`,
  `scalar_free_energy` returns `float(total)`). Test `lib/Mycelium.jl/test/element_type.jl`. Review
  [[LP-058 Findings of the review of P8]] (Codex, read-only in the main checkout; two defects fixed).
  All eight suites passed on the final code plus the Mycelium device tests (logs in
  `~/.cache/lenticulum/p8/`). New row **P30**: Lenticulum's Gaussian factors still send Float64
  messages for Float32 parameters, so a Lenticulum graph is not yet Float32 end to end.
- **I25 follow-up** (2026-11-18, chosen by the maintainer): `bench/bp_trees_scale.jl`, the I3 grids in
  mixed units, unit-free stopping (log `~/.cache/lenticulum/i25b/`). Normalised weights keep their tree
  across scales (up to rounding ties on the isotropic grid); raw ones need up to 51 % more plain
  evaluations, 2.5 times with Anderson; normalised loses at most 17 % on a shared scale. Default kept;
  `coupling.md` §3. Side finding: Anderson(3) is not scale-equivariant (no row). Bench and notes only,
  so no suite was run and there was no review.
- **I25** (2026-11-18): `coupling_weights(g, ps; normalize = true, default, hard)` in Lenticulum
  (`coupling.jl`, `coupling.md`), with `coupling_precision(f, ps)` per linear-Gaussian factor type.
  Weight = largest $\lVert D_u^{-1/2} J_{ab} D_v^{-1/2}\rVert$ over the factor's variable pairs (raw
  $\lVert J_{ab}\rVert$ with `normalize = false`). Bench (`bench/bp_trees.jl`, log in
  `~/.cache/lenticulum/i25/`): raw = the hand-made `1/q` exactly; normalised equal on anisotropic grids,
  8–16 % more evaluations plain on isotropic/random (±3 % with Anderson). Default kept normalised for
  unit invariance (measured by the follow-up above). Review [[LP-057 Findings of the review of I25]]. Lenticulum suite passed;
  only notes changed in Mycelium.
- **I3** (2026-11-18): `spanning_tree_schedule(g; weights, ntrees)`. `weights` (callable on factor
  indices or Dict by name) gives a maximum-weight spanning forest (Kruskal); `ntrees = K`/`:cover`
  appends trees preferring unused edges. Default unchanged (checked against the old function on
  6 484 graph/root pairs by the reviewer). `bench/bp_trees.jl`: weights `1/q` 1.7–5.4 times fewer
  evaluations than BFS where BFS holds the weak direction, never worse; alternating trees lose in
  evaluations everywhere. Recorded in `schedules.md`, [[Composing Optimisers]] §4.4,
  [[LP-056 Findings of the review of I3]]. Codex implemented (`codex exec -C ../Lenticulum.jl-codex`,
  one run, about 1 h with suites and benchmark; its tree oracle could not tell weighted from BFS
  order, so the maintainer added a loopy test). Follow-up row I25 (default weights from factors).
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

No row is set; pick one with the maintainer. Section P has P10 (needs F2), P14 (shared solver layer, large: design first), P15 (coverage, examples job), P16. Close to P5:
a learned noise with hard messages, a residual-space formulation of the positive definite path (LP-047
finding 4; no row yet). Left open by P27 (no rows yet): the Bethe free energy with a mixture marginal
under BP, derivatives through the new rules, two extended components with hard directions. Small:
`ImplicitLayers`' `SolveReport` also has `residual::Float64` (not checked whether duals reach it; that
package is unreviewed); `AndersonSchedule` runs without its memory under duals (LP-053). Not benchmarked yet: GH-20's per-replica path for
device and view inputs to the hard-constraint plate kernel.

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
  and probe 2's properness check found F1. On P3 the first run crashed inside vibe on its first command (a validation error of its own); the rerun reached 100 turns with only probe 1 in the answer file: run its later probes yourself. On P28 the same: four probes recorded, three more written but not recorded (all passed when run). On P9 (vibe 3.1.2, after it had failed on P29): five probes recorded, the sixth (the one with the findings) written and not recorded. On I22 (P6 brief form): nine probes, all recorded, a finished answer with "Not run" and "Unverified" sections, one right finding, \$0. On I23 (same form, `--max-price 3`): three probes, all recorded, one right finding, \$0; vibe exited 3 at its turn limit after writing the answer. On I24 (same form): twelve probes, a finished answer with tables, one right finding, \$0. Another user's `vibe-rs --continue` runs on this machine: wait on the PID of your own run, not `pgrep -x vibe-rs`. On P33 (P6 brief form, `--max-price 3`): four probes, all recorded, no summary (exit 3 at the turn limit), \$0; the session ran the open edge itself. On P30 (P6 brief form, `--max-price 3`): five probes, three recorded, no summary at the turn limit (\$0); of the two unrecorded, one needed an import repaired (`SpecialFunctions` is not in `p6env`: use `Lenticulum.SpecialFunctions`) and showed the one defect. Its old-versus-new bitwise comparison (`probe2_f64_same.jl` with `scratch/p30/oldenv`) is reusable.
