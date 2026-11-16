#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-16, after P4's AD half (the coupled replay through a `HardConstraintFactor` under ForwardDiff and Zygote), committed on `dev-danielb`; not pushed. P4 is done. The backlog got section G the same day (notes only).

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P4 AD committed (LenticulumCore `extended_gaussian.jl` and its extensions, Mycelium `gaussian.jl`, Lenticulum `hard_constraint.jl`, `constraint.jl`; tests `extended_gaussian_ad.jl`, `hard_gradient.jl`). Not pushed |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `60903bc4` with the P4 AD diff applied (without the LP-043 fix); Mistral's probes in `scratch/p4ad/`. Stashes, all droppable: `mistral-wt-p25-applied-stale-2026-11-16` and older. LenticulumCore's test environment there is instantiated (`--project=lib/LenticulumCore.jl/test` runs its probes) |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

## Done last

- **P4, AD half** (committed): `ExtendedGaussian{T<:Real}`; decompositions on the values, the
  derivative carried by $\Delta = X - \operatorname{stop\_gradient}(X)$ (one formula for ForwardDiff
  and Zygote); a first-order rank change refused under ForwardDiff; non-mutating `_independent_join`;
  the hard factor's free energy through $\sum\log\sigma + \operatorname{tr}(G^+\Delta)$. Details in
  [[Feature Backlog]] (Done 2026-11-16) and [[lib/LenticulumCore.jl/src/extended_gaussian|extended_gaussian]]
  "Derivatives (P4, 2026-11-16)".
- Verified: LenticulumCore, Mycelium, MyceliumLearning, Lenticulum suites after the review fix;
  AlgebraicFactors, Adversarial, ImplicitLayers, VariationalDiffusion before it (the fix touches only
  the empty-eigenvalue case of `_psd_split`). Logs in `~/.cache/lenticulum/p4ad/`.
- Review: [[LP-043 Findings of the review of P4]] (Mistral; Zygote threw on a belief with no Gaussian
  direction; fixed, regression test).

## Next

P4 unblocked P2 (hard inputs in the free energies of `SwitchingConstraintFactor`, the plate paths
and the frozen beliefs of the gradient; with `DiagonalCoupling` the fixed-belief gradient refuses a
graph whose variables have hard directions, "no canonical form") and P6 (`HardConstraintFactor` in
plates and in `MeanField()` nodes). Neither is designed yet; pick one with the maintainer.

## Open

- **P4 limits:** Zygote has no rank check (a rank-changing tangent gives the constant-rank
  derivative silently); second derivatives (nested duals) through extended Gaussians are refused;
  `_moving_flat_split` (P25) stays ForwardDiff-only. **P27:** mixture × extended product.
- **P25/P26 limits:** a parameter that changes a rank only at second order (a backward message's
  $A^\top R^{-1}A$ as $A$ leaves a rank-deficient value; $A_x$ leaving 0 in a cluster) gets the
  constant-rank derivative while the value is discontinuous there; Zygote does not go through the
  flat path's moving directions; the plate methods (`factor_messages` with a shared metric) and an
  inference with a dual $A$ upstream of a flat path are untested under moving flat directions;
  `cluster_marginal` of a flat class under ForwardDiff is untested.
- P24's Zygote rule gives the incoming messages no tangent (`@not_implemented`); differentiating
  through inference with cluster nodes stays closed (their messages are built by mutation).
- **Backlog section G** (decisions, control and games, G1–G10; 2026-11-16, from Daniel's question on
  reinforcement learning and open games): rows only, from a reading of code and notes. Nothing is
  designed or run; G5–G8 need P16. Priorities are proposals.
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
  defect, then also stopped at 100 turns (\$2.35) before its summary. On I19 (Mistral Large, \$0) it wrote four
  probes but no answer file at all, and three of the probes had errors of their own (a misparsed
  tolerance, a parenthesis, a keyword read for a NamedTuple): run and repair its probes before
  believing a failure. On I20 it kept the answer file but was stopped by `--max-price 1` after 71
  turns: `cost_usd` counts list prices even for the free Mistral Large (3.9 M input tokens, \$1.01), so
  give a free-model review `--max-price 3` to reach its turn limit. On P20 (2026-11-16) both runs lost
  every command to a `cd … &&` prefix (not on the allowlist). On P23 the brief gave the exact
  command line to copy (`julia -t auto --project=<env> scratch/p23/probe1.jl > ….out 2>&1`) and it
  ran six probes; on P24 the command line without redirection and "never prefix `cd`": every probe
  ran and it finished with an answer file (\$0.41). Keep doing that. On P25 the commands ran but it
  spent its 100 turns on 26 debug variants and wrote one entry: run its last probes yourself. On P4
  (AD) no answer file at all; its last probe (needing imports added) found the one defect. Its
  probes need the package's test environment (`--project=lib/<Pkg>/test`); say so in the brief.
  A watcher `while pgrep -f "vibe -p …"` matches its own command line and never ends: match `vibe-rs` with `pgrep -x` or wait on the PID.
