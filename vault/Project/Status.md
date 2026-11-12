#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-12, after I5 (committed and pushed).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | clean, pushed with the I5 commit; `master` is behind (merge on request) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | the helper worktree for Mistral (`AGENTS.md`, "Helper agents and worktrees"); at `660ffe5a` with the I5 diff applied and Mistral's scratch tests in `scratch/` (untracked); environments instantiated. Bring it to `dev-danielb` (`git merge --ff-only dev-danielb`) and apply the change before a review |

Pushing: the `mathstruct` SSH key is often rejected; push over HTTPS with `gh`'s credentials
(`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done: I5, the trees read by position

Committed on `dev-danielb`: Mycelium (`passing.jl`, `free_energy.jl`,
`plates.jl`, `graph.jl`, the ChainRulesCore extension, `test/state_tree.jl`), MyceliumLearning
(`gradient.jl`, the import list, `test/gradient_couplings.jl`), `bench/state_scaling.jl`, notes.
The design, the measurements and what is left are in `lib/Mycelium.jl/src/passing.md`, "The trees
by position"; the review is in [[LP-025 Findings of the review of I5]].

- Verified: all eight suites passed on 2026-11-12 (logs in `~/.cache/lenticulum/i5/`). After the
  review fix, Mycelium, MyceliumLearning and Lenticulum were run again. The new timing test fails
  on the old code. The benchmark grows ×1.8–2.2 per doubling: sweep, free energy, gradient,
  `parameter_step`.
- Not run: the device tests (`LENTICULUM_DEVICE_TESTS=true`); the change is not device-facing.
- Next: the item the earlier profile left for after I5: `passing.jl` `belief_distance`, 106 of 725
  sweep samples, probably dynamic dispatch; profile again first, since the sweep is now 4 times
  faster. Rows added: I16 (Zygote backward quadratic, expected), I17 (coupled drivers and graph
  adjoint still read by name).

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
