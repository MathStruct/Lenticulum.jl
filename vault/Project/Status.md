#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-15, after I21 (mixture blocks with a precision per component, energy rows kept), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the I21 commit (pushed): `lib/Mycelium.jl/src/blocks.jl`, `blocks.md`, `cluster_batch.jl`, `cluster.md`, `graph.jl`, `free_energy.jl`, `free_energy.md`, `Mycelium.jl` (export); `lib/Lenticulum.jl/test/regime_plate.jl`; the vault notes below and the new LP-036. verified (below) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `4ffd955c` with the I21 diff applied (before the fix of LP-036); Mistral's probes in `scratch/i21/`, older in `scratch/i20/` etc. (untracked). Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-i20-applied-stale-2026-11-15`, `mistral-wt-i19-applied-stale-2026-11-15`, `mistral-wt-ridge-applied-stale-2026-11-13` and the older ones of 2026-11-12 |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **I21** (committed and pushed): plates of tied cluster nodes send a continuous channel a
  `ComponentMixtureBlock` (a precision and a flag per component) once the outcomes' precisions
  differ, instead of a `VectorBlock` of mixtures (`blocks.md` "Mixtures with a precision per
  component"); the plate's energy rows are kept in the graph per store layout (`FactorGraph.energy_rows`,
  `free_energy.md`). `fit_em` on the mixture plate 0.025 → 0.018 s and 0.219 → 0.132 s
  (`~/.cache/lenticulum/i20/bench_c.txt`), same iterates; [[Benchmark against RxInfer]] 19 ms and
  0.15 s against 125 ms and 3.4 s (`~/.cache/lenticulum/i21/bench_mix.log`).
- Verified: all seven suites pass (`~/.cache/lenticulum/i21/suite_*.log`; Mycelium, MyceliumLearning
  and Lenticulum again after the LP-036 fix, `suite2_*.log`; the other four ran before that fix, which
  only adds a refusal); four mutations fail 2–57 tests (`i20/mut21_*.log`). `docs/make.jl` passes;
  `check_notes.py` as before. Not run: device tests (no device code).
- Review: Mistral, one finding (point × mixture blocks of different sizes not refused), right, fixed
  with a test ([[LP-036 Findings of the review of I21]]; archive `../Lenticulum.jl-Review-2026-11-15/i21/`).
- Found while testing: **P22**, a plate schedule on a store with another block numbering writes to
  the wrong rows silently (predates I21; [[Feature Backlog]]).
- Next: P22 (small, a defect), or the open defects P20/P21, or another item of the backlog. The
  mixture plate has no single hot spot left (25× closed-form EM).

## Open

- **P21:** ForwardDiff through a relation factor's flat-input path, and `NonlinearFactor` at a dual
  expansion point (`@test_broken` in Lenticulum `test/forms_crossing.jl`).
- **P20:** Zygote through a plated free energy fails on `sum(...; init)` (found by Mistral; the
  same before P7).
- AlgebraicFactors, ImplicitLayers and Adversarial run in CI only weekly: a change to the core
  should run their suites locally too (that is how the AlgebraicFactors break was found).
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
  give a free-model review `--max-price 3` to reach its turn limit.
