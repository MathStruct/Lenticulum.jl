#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, the polarity cache (not committed).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | the polarity cache uncommitted on top of I5 (pushed); `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | the helper worktree for Mistral (`AGENTS.md`, "Helper agents and worktrees"); at `3cd2c355` (I5) with the polarity-cache diff applied for Mistral's review, and Mistral's scratch tests in `scratch/` (untracked); environments instantiated. Bring it to `dev-danielb` (`git merge --ff-only dev-danielb`) and apply the change before a review |

Pushing: the `mathstruct` SSH key is often rejected; push over HTTPS with `gh`'s credentials
(`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done, not committed: polarities cached per graph

The item left after I5 was `belief_distance`. A new profile (`bench/sweep_profile.jl`, 2 401
factors) puts it under 4% of a sweep, so it was left alone. `resolve_polarity` took about 25%: it
built a `NamedTuple` type from run-time channel names on every message. `FactorGraph` now caches
the polarity per factor by (target, available), behind a lock (`graph.jl`,
`polarity_resolution.jl`; note `polarity_resolution.md`, difficulty 3). The chain builder of
`bench/state_scaling.jl` moved to `bench/kalman_chain.jl`, shared with the profile script.

- Verified: all eight suites passed (logs in `~/.cache/lenticulum/bd/`). The sweep is 24% faster at
  every size (29.8 ms instead of 39.1 at 4 801 factors; table in `lib/Mycelium.jl/src/passing.md`,
  "Where a sweep spends its time after I5").
- Not run: the device tests (the change is not device-facing).
- Review: Mistral found no defects (`../Lenticulum.jl-Review-2026-11-12/polarity-cache-mistral.md`).
  It ran an exhaustive comparison against the uncached function, the graph-edit paths
  (`replace_factor!` after the cache was filled), an 8-thread stress test, and Zygote and ForwardDiff
  through `step!`. Its scratch test is `../Lenticulum.jl-mistral/scratch/polarity_cache_review.jl`.
  A stress test is not a proof of thread safety.
- Next: I18. These are per-message costs that depend only on the graph and the parameters:
  `_rank_deficient` (an SVD of the weight matrix on every backward message, about 7.6% of a sweep)
  and `_improper_check` (about 6.8%). I16 and I17 are still open.

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
