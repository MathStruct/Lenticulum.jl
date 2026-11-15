#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-15, after I20 (untied data factors grouped, block inputs to cluster nodes), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the I20 commit (`661f0bbb`, pushed): `lib/Mycelium.jl/src/plates.jl`, `free_energy.jl`, `cluster.jl`, `cluster_batch.jl`, `plates.md`, `cluster.md`; `lib/MyceliumLearning.jl/src/em.jl`, `gradient.jl`; `lib/Lenticulum.jl/src/regime.jl`, `test/regime_plate.jl`; the vault notes below and the new LP-035. verified (below) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `2d554500` with the I20 diff applied; Mistral's probes in `scratch/i20/` (untracked; older ones in `scratch/i19/` etc.). Test environment loading this worktree: `~/.cache/lenticulum/i19/env-mistral`; one loading the main checkout: `~/.cache/lenticulum/i20/env`. Stashes, all droppable: `mistral-wt-i19-applied-stale-2026-11-15`, `mistral-wt-ridge-applied-stale-2026-11-13` and the older ones of 2026-11-12 |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **I20** (committed and pushed, `661f0bbb`): (1) data factors that `as_plate` leaves untied (one empty key per replica)
  are grouped in plate sweeps and free-energy rows when none is learnable and every key's
  parameters and state are empty, checked each sweep; edge by edge otherwise (`plates.md` "Untied
  factors without parameters"; `_GroupPlan.keys`, `_EnergyRow.keys`). (2) Cluster nodes and
  `RegimeFactor` take a plate sweep's inputs as blocks and return `CategoricalBlock`/`MixtureBlock`/
  `GaussianBlock` directly where shapes allow, bit-identical to the column path; `_continuous_message`
  groups components by a linear scan (`cluster.md` "Block inputs"). (3) `fit_em` filters keys with
  `haskey` once per key, not once per node. `fit_em` on the mixture plate 0.057 → 0.025 s ($N = 300$),
  0.615 → 0.219 s ($N = 3000$), same iterates (`~/.cache/lenticulum/i20/bench_base.txt`, `bench_b.txt`);
  [[Benchmark against RxInfer]] 30 ms and 0.24 s against RxInfer's 126 ms and 3.2 s (`i20/bench_mix.log`).
- Verified: all seven suites pass (`~/.cache/lenticulum/i20/suite_*.log`; ImplicitLayers' log has a
  precompile `LoadError` about `LogExpFunctions._log1pexp_thresholds`, also in I19's log, tests pass);
  `regime_plate.jl` again after the last test was added (`t2_regime_plate.log`); four mutations of
  the block path fail 18–69 tests (`mut_*.log`); `docs/make.jl` passes; `check_notes.py` as before.
  Not run: device tests (no device code).
- Review: Mistral, one finding (mislabeled categorical blocks), not reproduced; a test added
  ([[LP-035 Findings of the review of I20]]; archive `../Lenticulum.jl-Review-2026-11-15/`).
- Next: I21 (a block for mixtures with per-component precisions; cache
  `_energy_rows`), or another item of the [[Feature Backlog]]; the rest of F3, an energy threshold,
  needs an energy network trained on the benchmark's relations.

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
