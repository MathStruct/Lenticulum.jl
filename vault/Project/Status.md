#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-21, after I17 (reverse mode through the coupled drivers, `infer!` and the graph adjoint linear in the graph; review [[LP-074 Findings of the review of I17]]), committed on `dev-danielb`, not pushed. Before it I16 (Zygote's gradient of the free energy linear in the keys), I4, I26, U3, U2 (pushed with P10, P36 and P35), P15, P34, P12 and P33.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | I23, I24 and the LP-055 amendment on top of the cloud agent's `9d94e548`; pushed. PR #23 (`dev-danielb` → `dev`) closed unmerged on 2026-11-18; `dev` is 22 commits behind and can be fast-forwarded |
| `~/Code/CategoryTheory/Lenticulum.jl-codex` | `codex/i17` | at `3ff93526` with Codex's first I17 diff applied uncommitted (`IMPL-I17.md`, logs in `scratch/i17/`; the session's three fixes are not in it); run Julia there with `JULIA_DEPOT_PATH=$PWD/scratch/depot:$HOME/.julia:`. The I16 review state is in `git stash` ("I16 review state (before I17)"; its untracked files in `scratch/i16-untracked/`), I4's, I26's, U3's and U2's below it. Codex's sandbox cannot stash or check out (read-only git index): the session does that. Reset it before the next Codex package |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/i17-tests` | at `3ff93526` with Codex's first I17 diff applied (the review's probes in `scratch/i17r/`, the specification tests and their runs in `scratch/i17/`, all copied to `../Lenticulum.jl-Review-2026-11-21/i17/`); the I26 review state in `git stash` ("I26 review state (before I17 tests)"), older states below it. `scratch/p6env` serves any review and now has Krylov. Before the next review: reset to the commit under review, then apply its diff. `git stash -u` takes `scratch/` with it: stash without `-u` |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, ADTypes,
DifferentiationInterface, then include the first ten lines of `runtests.jl` and the file (one exists: `~/.cache/lenticulum/env-p29`).

## Done last

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
- **I4** (2026-11-20, chosen by the session: a medium row that wanted a benchmark): `ResidualSchedule`'s
  sweep keeps the residuals in an indexed heap and, after a commit, probes only the tasks that read the
  committed message (`_dependents!`, `lib/Mycelium.jl/src/passing.jl`; `passing.md`, "The residual
  schedule's queue"). The same commits as the old full scan, bit for bit (kept as `_residual_sweep_scan!`;
  `lib/Lenticulum.jl/test/residual_queue.jl` compares them on random loopy graphs, and fails when any
  dependency rule is removed). A factor is taken not to read the target's cavity only if it declares
  `exact_messages` (`message_uses_cavity` defaults to `false` and `DEQFactor`/`DiffusionFactor` read it
  undeclared). Measured, `bench/residual_queue.jl`: a sweep 19×–261× faster than the scan (3 × 3 to 10 × 10
  grids), $3.3\,T$ evaluations instead of $T^2$; to convergence 0.46–0.89 of flooding's evaluations, on the
  divider 0.23–0.30 against 2.1 (`bench/bp_schedules.jl`, its residual row now counted from the record).
  Review LP-072 (Codex, nine probes: three defects fixed with regression tests, one accepted, extra probes
  reduced). All seven suites that depend on Mycelium passed (`~/.cache/lenticulum/i4/final/`). Smoke runs of both
  benchmarks passed. Not run: the docs builds, device tests, the full smoke runner.
- **I26** (2026-11-20, the open high row): where $p \ge 2$ exact messages meet at a variable, the variable's
  term adds $-\tfrac12[(p-1)\log\det G - \sum_i\log\det G_{-i}]$, $G$ the Gram matrix of their hard bases
  (`_hard_overlap` in `lib/Mycelium.jl/src/free_energy.jl`; derivation in `free_energy.md`, "Exact messages that
  meet at a variable"). Two oblique exact projections of $z$ gave $-\log Z + \log\sin\theta$; the LP-070
  `@test_broken` passes. Codex wrote `test/hard_overlap.jl` from the specification (dense oracles, 2204
  tests); reviews LP-071 (Codex: four findings, two fixed with regression tests, two older and now rows I27,
  I28; Mistral: no defect). **A contract now written down:** a factor that sends exact messages must include
  the Jacobian of its own exact directions on its cavity's support, as `HardConstraintFactor` does; the test
  fixture `LinePrior` in `test/split.jl` did not and was corrected. All eight suites passed (`~/.cache/lenticulum/i26/`;
  Lenticulum rerun after the fixture fix, `Lenticulum-final.log`). Not run: the docs builds, device tests; cost not measured.

## Next

No row is set; pick one with the maintainer. Open after I17: ForwardDiff and reference differences through a replay cost one pass per parameter coordinate (quadratic by nature; 6.7 s for reference differences at 241 factors); the I17 tests add about 10 min to Lenticulum's suite (`coupled_keys.jl` 6 min, `coupled_spec.jl` 4 min: candidates for trimming). Open after I16: Zygote's compilation still grows with the keys (66 s at 961 factors, I1); the block-row path's nodes outside rows still read `ps` per node; O(degree²) at a hub (LP-073). Open after I4: keep the residual heap between sweeps (its
first probe of every task is a third of a sweep's cost; needs to know when the store changed outside a
sweep); I24's `_max_pending` still scans every task; the residual schedule's means are further from the
dense solve than flooding's at the same message tolerance (4e-9 against 1e-10 on the grids; the schedule's,
not examined). Section P has P14 (shared solver layer, large: design first) and P16 (owned scalarisations:
design first). U5 (permutation, sum, clamp factors; a permutation is now a `SplitFactor`). Open after I26: I27 (the
gradient drivers do not check that the free energy is finite), I28 (`MeanField()` next to hard directions), mixtures
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
