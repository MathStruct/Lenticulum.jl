#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after I18 in part (the rank memo and one factorisation; **not committed**).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | the I18 change uncommitted (below); `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | the helper worktree for Mistral (`AGENTS.md`, "Helper agents and worktrees"); at `15db235d` with the I18 diff applied for Mistral's review (before the fixes of LP-026), and Mistral's scratch tests in `scratch/` (untracked). An old draft of the polarity-cache notes is in the stash `mistral-wt-polarity-cache-draft-2026-11-12` (superseded by the commit; drop it). Bring the worktree to `dev-danielb` and apply the change before a review |

Pushing: the `mathstruct` SSH key is often rejected; push over HTTPS with `gh`'s credentials
(`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done, not committed: I18 in part

What was done, in `dev-danielb` and uncommitted:
- `_rank_deficient` (Lenticulum `gaussian.jl`) keeps its answer per weight array: a global memo keyed
  by `objectid`, checked with a weak reference and a copy (`gaussian.md`, "What the checks cost").
- `_moments` factorises the input precision once: `LenticulumCore._mean_cov`, with an rrule that
  calls the two old rules.
- `_nearly_symmetric` has a loop method, and `_symmetric_part` is one broadcast (`gaussian_belief.md`,
  "Per-message costs").
- The tests are `test/i18.jl` in LenticulumCore and in Lenticulum. They compare each fast path
  with the old code on generated inputs.

- Measured: the sweep is about 15% faster at every size (12.4 instead of 14.6 ms at 2 401 factors),
  with 14.6 MB allocated per sweep instead of 22.7. A backward message at d=2 takes 1.06 μs
  instead of 2.4 (tables in `lib/Mycelium.jl/src/passing.md`, "After I18").
- Review: Mistral, triaged in [[LP-026 Findings of the review of I18]]. Two defects were fixed with
  tests (a non-square input, an error message), the memo tests were added, and one race was not
  reproduced.
- Verified: all eight suites passed (`Pkg.test()`, logs in `~/.cache/lenticulum/i18/`). After the
  LP-026 fixes, LenticulumCore and Lenticulum passed again, and so did the device tests of Mycelium
  and MyceliumLearning (`LENTICULUM_DEVICE_TESTS=true`).
- Not run: Enzyme through `_mean_cov` (no test drives it).
- Next: commit when asked. Then comes the rest of I18, the dynamic dispatch on the polarity type in
  `_rule_message`, about 17% of a sweep (`polarity_resolution.md`, difficulty 3). After that come
  I16 and I17, or P1 by the proposed order in [[Feature Backlog]].

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
