#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after the rest of I18 (planned factor -> variable messages; committed).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | the rest of I18 committed (below), not pushed; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | fast-forwarded to `10dfa23a`, with the rest of I18 applied for Mistral's review, and its scratch tests in `scratch/` (untracked). Two stale stashes: `mistral-wt-polarity-cache-draft-2026-11-12` and `mistral-wt-i18-predfix-2026-11-12` (the pre-fix I18 diff, superseded by `9fbba63f`); both can be dropped |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done: the rest of I18

- `_step!` (Mycelium `passing.jl`) keeps an `_EmitPlan` per (edge, bit mask of the informative
  inputs) in `g.emit_plans` (`graph.jl`): the checked polarity, the input edges, and the factor,
  rule and input names in its type. Mean-field nodes and factors with more than 64 edges keep the
  old path (`polarity_resolution.md`, difficulty 3, "Planned per graph").
- Measured: the sweep is about 19% faster at every size (9.96 instead of 12.43 ms at 2 401 factors;
  `passing.md`, "After the plans").
- Test: Mycelium `test/emit_plan.jl` compares it with the old path on random graphs, and runs 8 tasks at once.
- Review: Mistral, triaged in [[LP-027 Findings of the review of the I18 plans]]. No defect was found.
  The concurrency test was added because of the review.
- Verified: all eight suites passed (`Pkg.test()`, logs in `~/.cache/lenticulum/i18b/`), and so did
  the device tests of Mycelium and MyceliumLearning. Mycelium passed again after the concurrency test was added.
- Not run: the docs build (no docstring changed).
- Next: push when Daniel asks; then I16 and I17, or P1, by the proposed order in [[Feature Backlog]].

## Open

- **P1:** tests where the canonical form (an `improper` `GaussianBelief`) meets `ExtendedGaussian`.
  One crossing already broke a test unnoticed (AlgebraicFactors, fixed 2026-11-11).
- **P20:** Zygote through a plated free energy fails on `sum(...; init)` (found by Mistral; the
  same before P7).
- AlgebraicFactors, ImplicitLayers and Adversarial run in CI only weekly: a change to the core
  should run their suites locally too (that is how the AlgebraicFactors break was found).
- Mistral review agent: `vibe ... --agent review --trust --workdir ../Lenticulum.jl-mistral`
  (`~/.claude/CLAUDE.md`). A review needed about 140 turns; give it 100 and continue the session
  with `vibe -c` if it stops.
