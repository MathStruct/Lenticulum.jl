#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-22 evening, four Codex packages merged, three more queued (see Next). Before it I30 (the graph-wide trees boxed in the graph drivers; review [[LP-077 Findings of the review of I30]]), pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | LP-076 finding 2, I30's design and I30 pushed after it (`75b92efa`). PR #23 (`dev-danielb` → `dev`) closed unmerged on 2026-11-18; `dev` is behind and can be fast-forwarded |
| `~/Code/CategoryTheory/Lenticulum.jl-codex` | `codex/*` of the queue (see Next) | the queue runs there (script: the session's scratch `codex-queue.sh`); Codex's I30 notes are in `git stash` (top) and `../Lenticulum.jl-Review-2026-11-22/i30/`. Codex's quota: a spare reset in use, expiring 2026-12-01; the weekly one resets 2026-11-29 (`agent-allowance` still showed it used up while Codex answered). Run Julia there with `JULIA_DEPOT_PATH=$PWD/scratch/depot:$HOME/.julia:`. Older states in `git stash` (shared by all worktrees): the I17 states on top, I16's, I4's, I26's, U3's and U2's below. Codex's sandbox cannot stash or check out: the session does that |
| `~/Code/CategoryTheory/Lenticulum.jl-glm` | detached at `79c15aa0` | made for GLM's review of P14 W1, with the package untracked (`scratch/p14env`, `scratch/p14w1/` probes); reset (`git clean` of `lib/ImplicitSolvers.jl`) before the next review |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/i30` | at `956a5e2f` with I30 applied uncommitted (the same diff is committed on `dev-danielb`); `scratch/i30/` (baseline recorder and the 39 old records, `snoop.jl`, the benchmark logs) and `scratch/i30r/` (GLM's probes). Its Manifests were stale (no `LenticulumCoreForwardDiffDIExt`, LP-067) and are resolved now. Reset before the next review: `git stash` (without `-u`), then check out the commit under review |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, ADTypes,
DifferentiationInterface, then include the first ten lines of `runtests.jl` and the file (one exists: `~/.cache/lenticulum/env-p29`).

## Done last

- **I30** (2026-11-22, Daniel: "do all that"). Inside the graph drivers' differentiated region `ps`, `st` and the
  offset tree travel in a non-parametric box (`LenticulumCore._Erased`), read only through rules whose pullbacks are
  boxed too (`_ErasedPullback`); pinning is in flat coordinates (`_graph_pin`); the replay reads `ps` once.
  `bench/structure_compile.jl`, first gradient on a new structure at 121 factors: Zygote 13.75 → 1.61 s, ForwardDiff
  5.81 → 0.98 s; the implicit adjoint's compilation at 31 factors about 11 → 1.8 s; warm times unchanged. Codex
  stopped at the design's pinning step with a counterexample (design revised, a deliberate change of one value);
  Mistral implemented steps 1–6 (one defect: the boxed state's cotangent dropped under Zygote, found by comparing
  39 gradients recorded on the old code); a SnoopCompile profile led the session to box the pullbacks, which did
  most of the gain. GLM reviewed (\$2.96): no defect, one suspicion fixed with a test. Values bitwise equal but for
  a `-0.0` and one last bit (Zygote, a tied key). Tests `lib/Lenticulum.jl/test/erased_trees.jl` (slow tier).
  All eight suites passed, Lenticulum with both tiers (`~/.cache/lenticulum/i30/gates2/`). Not run: the docs
  builds, device tests (`_pin_coordinates_wide`'s device form is untested), the smoke runner.
- **LP-076 finding 2, and I30's design** (2026-11-22, Daniel: "do all that"). Under a caller's ForwardDiff a
  loss free of the coordinates left the gradient partly uninitialised: ForwardDiff's `extract_gradient!`
  loops over the caller's partials. `_gradient` for `AutoForwardDiff` wraps the loss (`_LiftToInner`,
  `LenticulumCoreForwardDiffDIExt`) so that such a value carries this call's zero partials. Codex's review
  of the fix found one more (right): `_graph_ad`'s guard missed a dual captured by `fun`, and ForwardDiff
  refused the call; the guard walks `fun` now. Tests `lib/LenticulumCore.jl/test/i1.jl`, "LP-076 finding 2"
  (39; six fail without the fix). Triage in [[LP-076 Findings of the review of I1]] (closed). I30's design is
  in `gradient_couplings.md`, "The graph-wide trees out of Zygote's signatures (I30)", with its gate
  `bench/structure_compile.jl` (first call per new structure; smoke run passed). All eight suites passed, Lenticulum with both tiers (`~/.cache/lenticulum/lp076/gates2/`). Not run: the docs builds, device tests (Codex's point 3, device coordinates under a caller's ForwardDiff, untested).
- **I1, the AD paths** (2026-11-22, chosen by Daniel from the list: the one medium row with a measured
  limit). Measured first: on a new structure the replay's first call compiled 10.7 s with ForwardDiff and
  7.1 s with Zygote at 31 factors (11.2 s and 13.7 s at 61), the implicit adjoint about 15 s; the envelope
  driver and EM were already cheap. ForwardDiff's time was the factors' own code compiled for a new dual
  type (DifferentiationInterface's tag holds the closure, which holds `ps` and `st`): the graph drivers now
  take `_graph_ad`'s fixed tag and chunk (`LenticulumCoreForwardDiffDIExt`), only when no dual is reachable
  from the contexts (nesting). Zygote's was its tuple adjoints through `_rebuild` and `offset_tree`'s
  recursion: a rule for `_unflatten`, and a loop for trees wider than 32 fields (narrower ones keep the
  recursion, which Enzyme needs: the suites found it). Now 0.81 s and 2.5 s at 31 factors, 1.5 s and 5.3 s
  at 61, 3.6 s and 13.4 s at 121 (`passing.md`, "The AD paths (I1)"); warm times unchanged. Review LP-076
  (Codex: two defects of the rule fixed with tests; two older nested-ForwardDiff defects open). Tests
  `lib/LenticulumCore.jl/test/i1.jl`, `lib/Lenticulum.jl/test/structure_compile.jl` (slow tier). All eight
  suites passed, Lenticulum with both tiers (`~/.cache/lenticulum/i1/gates/`). Not run: the docs builds,
  device tests (the rule's device form is untested), the smoke runner; the implicit adjoint and
  ForwardDiff's remainder were not profiled after the change.
- **I27, I28** (2026-11-22, chosen by Daniel: the two small rows left by LP-071). I27: `free_energy_gradient`
  evaluates `scalar_free_energy` once and refuses a value that is not finite (`_check_finite_free_energy`,
  `lib/MyceliumLearning.jl/src/gradient.jl`), after the gradient: at the store for the envelope driver, the
  replay's own loss, the free energy at the implicit adjoint's fixed point; before, two exact observations of one direction got finite
  gradients, and a constant `Inf` cost finite replay gradients. I28: `_product_belief` takes an extended
  Gaussian flat nowhere by its mean and singular covariance, so a `MeanField()` linear or switching factor next
  to hard directions gives the dense value. Tests `lib/Lenticulum.jl/test/hard_overlap.jl`, "I27, I28: review
  of I26" (57); review LP-075 (Codex: the coupled drivers had skipped the check, fixed; a switching test added).
  Suites: MyceliumLearning, ImplicitLayers and Lenticulum with both tiers passed (`~/.cache/lenticulum/i27/run3/`). Not run: the docs builds, device tests, the cost of the check (one forward pass of the free
  energy per gradient).
- **I17** (2026-11-21, chosen by Daniel from the list: the one medium row): reverse mode (Zygote) through the
  coupled drivers (`free_energy_gradient` with `schedule`), through `infer!` and through the implicit graph
  adjoint is linear in the graph. The replay is one loop over message versions with one `rrule` walking the
  tasks backwards through per-task pullbacks; each task reads a local store and its own parameter subtree
  (`lib/MyceliumLearning.jl/src/gradient_couplings.md`, "Reverse mode by message versions"). Measured first:
  all three paths were quadratic. Design by the session, implementation by Codex, tests by Codex (old code
  copied as the oracle) and by Mistral (from the specification, `coupled_spec.jl`, literals recorded on the old
  code; four test mistakes fixed by the session). Measuring Codex's version found three more superlinear
  forward costs of the Krylov product, fixed by the session; one of them was in I16's rule of `_subtrees` and
  so in the free energy's gradient too. `bench/coupled_scaling.jl` at 121 factors: the unrolled Zygote gradient 479 → 115 ms, 419 → 121 MB, its compilation 767 → 21 s; one Krylov product 747 → 76 ms; every column now about 2× per doubling. Review LP-074 (Mistral: no
  defect). All seven suites that depend on Mycelium passed (`~/.cache/lenticulum/i17/final/`). Not run: the
  docs builds, device tests after the session's fixes (Codex ran them on its version), the full smoke runner
  (the new benchmark's smoke run passed), I16's `bench/zygote_scaling.jl` after the `_subtrees` fix.
- **I16** (2026-11-20, chosen by the session: a medium row that asked for a measurement first): the graph
  free energy reads `ps`, the state and the message store once for all nodes, and ChainRulesCore rules add
  the nodes' cotangents per key and per edge (`_subtrees`, `_states`, `_local_stores`, `_sum_onto`;
  `lib/Mycelium.jl/src/passing.md`, "The backward pass by key"). Measured first: the reads of the store,
  not of `ps`, were half the backward time. `bench/zygote_scaling.jl` at 961 factors: gradient 3.11 s → 286 ms,
  1.56 GB → 203 MB, Zygote's compilation 45 min → 66 s; linear per doubling; the plain free energy no
  slower. Tests by Codex from the specification (`lib/Lenticulum.jl/test/zygote_keys.jl`); review LP-073
  (Codex: four defects fixed with regression tests, one limit). All seven suites that depend on Mycelium
  passed (`~/.cache/lenticulum/i16/final4/`). Not run: the docs builds, device tests, the full smoke runner.

## Next

**Codex queue (2026-11-22, Daniel: use Codex's spare reset aggressively before it expires on 2026-12-01).** Merged today: the Mycelium coverage tests (`79c15aa0`), I30's open ends (`4342f879`), the Anderson and block AD fixes with two defects of the coverage tests (`2112c49e`), P14 W1, the package `lib/ImplicitSolvers.jl`; reviews triaged in [[LP-078 Findings of the reviews of the Mycelium coverage tests and the Anderson and block AD fixes|LP-078]]. Running in `../Lenticulum.jl-codex`, detached (`setsid`), logs and copies of the briefs and scripts in `~/.cache/lenticulum/codex-queue/` (`heap.log`, `queue3.log`): `codex/residual-heap` (I31, keep the residual schedule's heap between sweeps, `heap-brief.md`; starts 22:45 on 2026-11-22 after Codex's 5-hour limit), then `codex/p14-w2` (ImplicitLayers onto ImplicitSolvers, `w2-brief.md`) and `codex/p14-w3` (VariationalDiffusion, `w3-brief.md`); the queue stops itself on a usage limit. **I31's first run (22:45–22:50) stopped at its determinism gate, rightly** (checked: RED-Diff draws from an RNG in its prox config, `lib/VariationalDiffusion.jl/src/reddiff.jl:90–101`, so a repeated probe differs; its answer `../Lenticulum.jl-Review-2026-11-22/residual-heap/IMPL-heap-run1.md`, a note-only WIP commit on the branch). Decided: a trait `repeatable_messages(f)`, `false` by default, declared `true` for the deterministic factors; reuse only on graphs whose factors are all repeatable. Queued as a second run on `codex/residual-heap` after W3 (`codex-queue5.sh`, `heap2-brief.md`, `queue5.log`; queue3/queue4 stopped themselves falsely, their `grep "usage limit"` matching the brief's own text in Codex's log; queue5 checks only a failed run's tail)). **P14 W3 merged (2026-11-23)**, reviewed by Mistral, triaged in [[LP-079 Findings of the reviews of P14 W2 and W3|LP-079]]. **P14 W2** (`codex/p14-w2`, with three old assertions changed to the shared API, in `../Lenticulum.jl-mistral` on `mistral/review`) under Mistral's review; its suites pass only with Enzyme ≥ 0.13.221 (0.13.205 aborts in Enzyme's compiler at `test/parameter_ad.jl:22`) (`grep "usage limit"` in the logs). Each job leaves a WIP commit on its branch and its answer in `../Lenticulum.jl-Review-2026-11-22/<topic>/IMPL-*.md`. **To do next:** for each branch, read the diff, have another model review it (Mistral in `../Lenticulum.jl-mistral`, GLM in `../Lenticulum.jl-glm`; the review briefs in `../Lenticulum.jl-Review-2026-11-22/small-fixes/` and `p14-w1/` are the model), triage in a problem note, run the suites of the touched packages and their dependents, merge into `dev-danielb` without Codex's own `Status.md` edits. Left for Codex after them: P14 W4 (MyceliumLearning's graph adjoint) and W5 (Krylov.jl out). Open after I30: the implicit adjoint reads `ps` twice (its Zygote compile at 61 factors not profiled); Zygote through `infer!` (`_functional_infer`) not measured. Open after I17: ForwardDiff and reference differences through a replay cost one pass per parameter coordinate (quadratic by nature; 6.7 s for reference differences at 241 factors); the I17 tests, trimmed (2026-11-21), take 7 min 51 s of Lenticulum's suite instead of 10 min 21 s: Zygote through the old code (the oracle) is replaced by ForwardDiff on the new, whose values the old ones equal; what is left is compilation per graph type, schedule and backend, not the number of cases. The whole suite took 48 min 37 s (`~/.cache/lenticulum/trim/Lenticulum.log`), so it now has two tiers (`LENTICULUM_SLOW_TESTS`, `lib/Lenticulum.jl/test/runtests.jl`): the fast tier (28 min 37 s run beside the slow one) on every push, the slow tier (the tests that differentiate through replays, plates and element types, 28 min 20 s alone) in the new `Slow.yml`, weekly, on pull requests and on pushes to the AD paths. `Slow.yml` ran for the first time on the push of I29 (2026-11-21, run 44962044734): passed in 1 h 1 min, beside CI's 1 h 0 min. `ScaleShiftFactor` (3 min) stayed in the fast tier. I29 (found while trimming) is done: ForwardDiff through the coupled replay on a frozen key, a `Vector{Real}` from `_replace_coordinates`. Open after I16: Zygote's compilation still grows with the keys (66 s at 961 factors, I1); the block-row path's nodes outside rows still read `ps` per node; O(degree²) at a hub (LP-073). Open after I4: keep the residual heap between sweeps (its
first probe of every task is a third of a sweep's cost; needs to know when the store changed outside a
sweep); I24's `_max_pending` still scans every task; the residual schedule's means are further from the
dense solve than flooding's at the same message tolerance (4e-9 against 1e-10 on the grids; the schedule's,
not examined). Section P has P14 (shared solver layer, large: design first) and P16 (owned scalarisations:
design first). U5 (permutation, sum, clamp factors; a permutation is now a `SplitFactor`). Open after I26: mixtures
with hard components (not counted by the overlap term), a redundant clamp whose zero variance rounds to $-3\cdot10^{-17}$
and is refused during inference (`@test_broken` in `test/hard_overlap.jl`, no row). Open after U3:
the implicit adjoint for extended-Gaussian messages; a $O(d)$ free energy for `ScaleShiftFactor`. Open after U2: whether `SplitFactor` is cheaper than a
selection-matrix `HardConstraintFactor` (not measured), a batched plate kernel for it. Open after P10: the
tutorials and the README still use `field_nodes` for root queries; `ProxDM`'s free-energy nodes with
noise-free nodes (not measured). Close to P5: a residual-space formulation of the positive definite path
(LP-047 finding 4; no row yet). Left open by P27 (no rows yet): the Bethe free energy with a mixture marginal
under BP, derivatives through the new rules, two extended components with hard directions. Small:
`ImplicitLayers`' `SolveReport` also has `residual::Float64`; `AndersonSchedule` runs without its memory under
duals (LP-053). Not benchmarked yet: GH-20's per-replica path for device and view inputs to the hard-constraint
plate kernel.

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
  probes need the package's test environment (`--project=lib/<Pkg>/test`); say so in the brief. On I26 (2026-11-20, P6 brief form, `--max-price 3`): five probes recorded with an independent derivation of the
  formula, a sixth written and not recorded (the session ran it), no defect, \$0, exit 3 at the turn limit. On U3 (2026-11-20, the
  brief with the exact command line and "never prefix `cd`") 20 of its commands were refused, it hit 100 turns
  without an answer file, and three of its seven probes had wrong oracles (missing normalisers, a wrong dense
  marginal, a sign convention): run its probes and check their oracles before believing a failure.
  A watcher `while pgrep -f "vibe -p …"` matches its own command line and never ends: match `vibe-rs` with `pgrep -x` or wait on the PID.
  On P6 (2026-11-17) the brief gave the exact command line, a ready environment (`scratch/p6env`),
  "never prefix `cd`" and the answer-file rule: ten probes, a finished answer file, three right
  findings, \$0. Keep that brief as the model (`../Lenticulum.jl-Review-2026-11-17/p6/review-brief.md`). On P27 the same brief form gave five probes and an answer
  file with four of them at the 100-turn limit (\$0): one right finding; its fifth probe's reference lost
  digits at 1e6 (a false report): check a reviewer's reference before its claim. On P5 the same brief: four probes, the answer file
  records only the first, no summary (\$0); the maintainer ran the rest (one probe needed a one-line repair)
  and probe 2's properness check found F1. On P3 the first run crashed inside vibe on its first command (a validation error of its own); the rerun reached 100 turns with only probe 1 in the answer file: run its later probes yourself. On P28 the same: four probes recorded, three more written but not recorded (all passed when run). On P9 (vibe 3.1.2, after it had failed on P29): five probes recorded, the sixth (the one with the findings) written and not recorded. On I22 (P6 brief form): nine probes, all recorded, a finished answer with "Not run" and "Unverified" sections, one right finding, \$0. On I23 (same form, `--max-price 3`): three probes, all recorded, one right finding, \$0; vibe exited 3 at its turn limit after writing the answer. On I24 (same form): twelve probes, a finished answer with tables, one right finding, \$0. Another user's `vibe-rs --continue` runs on this machine: wait on the PID of your own run, not `pgrep -x vibe-rs`. On P33 (P6 brief form, `--max-price 3`): four probes, all recorded, no summary (exit 3 at the turn limit), \$0; the session ran the open edge itself. On P30 (P6 brief form, `--max-price 3`): five probes, three recorded, no summary at the turn limit (\$0); of the two unrecorded, one needed an import repaired (`SpecialFunctions` is not in `p6env`: use `Lenticulum.SpecialFunctions`) and showed the one defect. Its old-versus-new bitwise comparison (`probe2_f64_same.jl` with `scratch/p30/oldenv`) is reusable.
