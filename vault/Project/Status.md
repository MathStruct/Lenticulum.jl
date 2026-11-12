#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after I6 (batched per-outcome cluster nodes over a plate), committed and pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | at the I6 commit, pushed; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | reset to `7277fe4f` with the I6 diff applied (before the LP-032 fix), Mistral's probes in `scratch/i6/` (untracked). Stale stashes, all droppable: `mistral-wt-p11-applied-2026-11-12` (P11 is committed) and the four older ones of 2026-11-12 |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done today

- Committed and pushed earlier today: P1, P17/P18, I18, F3 in part (`round_trip`), P11 (`7277fe4f`).
- **I6** (committed and pushed): tied cluster nodes whose factors' coefficients or noise depend on their
  outcome are batched over a plate (Mycelium `cluster_batch.jl`: one precision per class, the data
  term per assignment; `FrozenClusterBeliefs` with per-class `Σ`, `H`, `class`); EM's stacked
  statistics read them (MyceliumLearning `em.jl`), and a plate row of `RegimeFactor`s outside
  clusters is batched as one-factor clusters (Lenticulum `regime.jl`). Tests:
  `lib/Lenticulum.jl/test/regime_plate.jl` (210). Notes: `cluster.md`, `regime.md`, L-SPEC-009,
  [[Benchmark against RxInfer]] (table remeasured; the plate is 2.2 and 2.6 times faster than the
  unplated graph, 0.36 s and 3.8 s), [[Feature Backlog]] (I6 done, I19 new).
- Review: Mistral's probes, run by this session after Mistral stopped three times at its turn limit
  without an answer ([[LP-032 Findings of the review of I6]]): one defect, older than I6 (labels of a
  categorical input not in the batching shape), fixed with a regression test; a second review by
  GLM-5.3 found nothing more.
- Verified: before the LP-032 fix all seven suites passed (Mycelium, MyceliumLearning, Lenticulum,
  AlgebraicFactors, ImplicitLayers, Adversarial, VariationalDiffusion; `~/.cache/lenticulum/i6/suite_*.log`),
  the docs build (`docs.log`); after it the Mycelium, MyceliumLearning and Lenticulum suites
  (`suite3_*.log`). Not run: device tests (no device code), the four other suites after the
  one-line fix.
- Next: the rest of F3 (a ridge residual) or I19 (profile the batched
  `RegimeFactor` plate).

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
  defect, then also stopped at 100 turns ($2.35) before its summary.
