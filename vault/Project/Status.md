#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-11, after P7 (commits `5cbc4c40`, `5a41803b`).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | clean, pushed at `5a41803b`; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | the helper worktree for Mistral (`AGENTS.md`, "Helper agents and worktrees"); fast-forwarded to `5a41803b`, environments instantiated. Bring it to `dev-danielb` (`git merge --ff-only dev-danielb`) and apply the change before a review |

Pushing: the `mathstruct` SSH key is often rejected; push over HTTPS with `gh`'s credentials
(`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Next: I5, parameters and state by integer slot

The proposed order in the [[Feature Backlog]] puts I5 first. Facts already established:

- A sweep and the free energy are quadratic in the number of factors. `bench/state_scaling.jl`
  at 2 401 factors: sweep 74.8 ms, free energy 42.9 ms, each ×3.3–3.5 per doubling.
- Profile at 2 401 factors (2026-11-11): `_subtree` and `_stored` take 437 of 725 samples of a
  sweep (60%) and 533 of 655 of the free energy (81%), almost all in `haskey` on the
  `NamedTuple`: a lookup by a run-time `Symbol` scans the names. Another 106 sweep samples sit on
  `passing.jl:325` (`belief_distance`), probably dynamic dispatch; look at it after I5.
- Lookup sites: `_subtree`/`_stored` in Mycelium (passing, free_energy, plates, cluster,
  cluster_batch, sampling), MyceliumLearning (gradient, gradient_couplings, em, graph_adjoint) and
  Lenticulum's `switching.jl`. Write-backs go through `_writeback` (`passing.jl`); MyceliumLearning's
  `_candidate` calls `_setsubtree` directly.
- `ps` and `st` are user-facing LuxCore-style `NamedTuple`s keyed by `parameter_keys(g)`; Zygote and
  ForwardDiff differentiate through `ps`. Callers read `st[name]` after `infer!`
  (`lib/Lenticulum.jl/test/lp020.jl`), so every key must be present in a returned state.
- Daniel asked for the design (where the slot index lives, how a `NamedTuple` maps to slots
  without an O(keys) cost per lookup) to be proposed before a large edit. Done means: the
  benchmark linear (table in `lib/Mycelium.jl/src/passing.md`), all seven suites pass, a test that
  fails on the quadratic version, a Mistral review in the helper worktree.

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
