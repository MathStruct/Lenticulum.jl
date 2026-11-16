#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P4's plate half (extended-Gaussian inputs in plates: tested, no code change), committed on `dev-danielb`; not pushed. P4's AD half is open and next.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P4 plates: `lib/Lenticulum.jl/test/extended_plates.jl` (new, in `runtests.jl`), `src/gaussian.md` "Extended Gaussian inputs" (In plates), [[Feature Backlog]] (P4 partial, P27 new), this note. P25/P26 pushed (`3101cf20`) |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `8a6a9f41` with the first P25 diff applied; Mistral's probes in `scratch/p25/`. Test environments: `~/.cache/lenticulum/i19/env-mistral` (this worktree), `~/.cache/lenticulum/i20/env` (the main checkout). Stashes, all droppable: `mistral-wt-p26-applied-stale-2026-11-16` and older |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P4, plates** (committed): extended messages in a plate are `VectorBlock`s and take the
  per-replica path; it was already right. `test/extended_plates.jl`: a plate of `GaussianFactor`s
  under hard relations against the dense conditioned Gaussian (10 instances × 4 replicas), and
  `LinearConstraintFactor`'s block method with an extended input in every replica against the
  sequential tasks and each replica alone (8 × 5). Known limits as `@test_broken`: mixture × extended
  (new row P27) and the coupled replay through a hard relation under ForwardDiff and Zygote.
- Verified: the new file (849 pass, 3 broken) and the Lenticulum suite
  (`~/.cache/lenticulum/p4/suite_Lenticulum_plates.log`). Test-only change, no external review.
- Found, not in P4: with `DiagonalCoupling` the fixed-belief gradient refuses a graph whose
  variables have hard directions ("no canonical form"); that is P2.

## Next: P4's AD half

The coupled replay (`free_energy_gradient(...; schedule, ad)`, `ExactCoupling`) through a
`HardConstraintFactor` fails: ForwardDiff at `ExtendedGaussian`'s `T<:AbstractFloat` (inside
`_as_extended` of the hard factor's dual inputs), Zygote at `_independent_join` of a tuple.
Reference differences agree with the dense $-\log Z$ to $10^{-9}$ (probes in the session scratch,
re-creatable from the broken testset). What it needs (to be designed, LenticulumCore's core type):
`ExtendedGaussian{T<:Real}` with every subspace basis (`_orth`, `_complement`, `hard_directions`,
the kept eigenvectors of `_canonical_part` and `_from_potential`) taken on the values and the duals
carried through the reduced inverses, a refusal when a derivative moves a null space (as P21 before
P25); `extended(::GaussianBelief)` with dual Λ; a non-mutating `_independent_join`; and the replay's
loss, the Bethe free energy, through the hard factor's term (an SVD) and `support_entropy`
(`eigvals`). Oracle: $-\log Z$ of the dense model (the broken testset's tree), ForwardDiff and Zygote
against it and against reference differences.

## Open

- **P4 AD half:** above. **P27:** mixture × extended product.
- **P25/P26 limits:** a parameter that changes a rank only at second order (a backward message's
  $A^\top R^{-1}A$ as $A$ leaves a rank-deficient value; $A_x$ leaving 0 in a cluster) gets the
  constant-rank derivative while the value is discontinuous there; Zygote does not go through the
  flat path's moving directions; the plate methods (`factor_messages` with a shared metric) and an
  inference with a dual $A$ upstream of a flat path are untested under moving flat directions;
  `cluster_marginal` of a flat class under ForwardDiff is untested.
- P24's Zygote rule gives the incoming messages no tangent (`@not_implemented`); differentiating
  through inference with cluster nodes stays closed (their messages are built by mutation).
- AlgebraicFactors, ImplicitLayers and Adversarial run in CI only weekly: a change to the core
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
  defect, then also stopped at 100 turns ($2.35) before its summary. On I19 (Mistral Large, $0) it wrote four
  probes but no answer file at all, and three of the probes had errors of their own (a misparsed
  tolerance, a parenthesis, a keyword read for a NamedTuple): run and repair its probes before
  believing a failure. On I20 it kept the answer file but was stopped by `--max-price 1` after 71
  turns: `cost_usd` counts list prices even for the free Mistral Large (3.9 M input tokens, $1.01), so
  give a free-model review `--max-price 3` to reach its turn limit. On P20 (2026-11-16) both runs lost
  every command to a `cd … &&` prefix (not on the allowlist). On P23 the brief gave the exact
  command line to copy (`julia -t auto --project=<env> scratch/p23/probe1.jl > ….out 2>&1`) and it
  ran six probes; on P24 the command line without redirection and "never prefix `cd`": every probe
  ran and it finished with an answer file ($0.41). Keep doing that. On P25 the commands ran but it
  spent its 100 turns on 26 debug variants and wrote one entry: run its last probes yourself.
