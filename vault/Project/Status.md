#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-13, after I19 (the batched `RegimeFactor` plate profiled and sped up), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the ridge residual commit (`1770cc13`, pushed; `master` fast-forwarded to it on 2026-11-13), then the I19 commit, pushed: `lib/Mycelium.jl/src/cluster_batch.jl`, `cluster.jl`, `cluster.md`, `passing.md`, both ChainRulesCore extensions, `lib/Lenticulum.jl/src/regime.jl`, tests in `regime_plate.jl` and `cluster_flat.jl`, `bench/regime_plate_profile.jl` (new), the notes below |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `1770cc13` with the I19 diff applied (before the two new tests and the fix to Lenticulum's extension); Mistral's probes in `scratch/i19/`, older ones in `scratch/ridge/`, `scratch/i6/` (untracked). A test environment loading this worktree: `~/.cache/lenticulum/i19/env-mistral`. Stashes, all droppable: `mistral-wt-ridge-applied-stale-2026-11-13` (committed), `mistral-wt-i6-applied-stale-2026-11-12`, `mistral-wt-p11-applied-2026-11-12` and the four older ones of 2026-11-12 |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **I19** (committed and pushed): a profile of `fit_em` on the mixture of regressions as a plate showed 45%
  of the time in per-replica joints, 25% in per-replica messages, 15% in per-replica clusters. Now
  one stacked joint for all replicas (`_batch_joint`), `_joint_message`/`_joint_belief` split into
  parts on all columns and per-replica assembly, one shared cluster for equal factors
  (`_as_clusters`, moved to Mycelium), an unrolled `_same_struct`. `fit_em` 0.338 → 0.059 s
  ($N = 300$) and 3.611 → 0.625 s ($N = 3000$), same iterates (`bench/regime_plate_profile.jl`, run
  before at `1770cc13` and after); against RxInfer 68 ms and 0.68 s versus 126 ms and 3.4 s
  ([[Benchmark against RxInfer]], log `~/.cache/lenticulum/i19/bench_mix.log`). Notes: `cluster.md`
  "Batching in one stacked joint", [[Feature Backlog]] (I19 done, I20 new), [[LP-034 Findings of the review of I19]].
- Review: Mistral stopped at 100 turns without an answer; its four probes, run by the maintainer's
  session after fixing their own errors, found no defect; two became tests. Archived in
  `../Lenticulum.jl-Review-2026-11-13/`.
- Verified: all seven suites pass (`~/.cache/lenticulum/i19/suite_*.log`); Lenticulum again after
  removing the duplicate `@non_differentiable` of `_as_clusters` from its extension and adding the
  tests (`suite2_Lenticulum.log`); three mutations of the batched data term fail 90, 54 and 60 tests;
  `docs/make.jl` passes (`docs.log`); `check_notes.py` as before. Not run: device tests (no device
  code); the other suites after the extension fix and the two tests (both touch only Lenticulum).
- Next: I20 (batch the tasks a plate steps one at a time) or another
  item of the [[Feature Backlog]]; the rest of F3, an energy threshold, needs an energy network
  trained on the benchmark's relations.

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
  believing a failure.
