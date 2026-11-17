#overview

> Where the work stands, for the next session to start from: the branch, what was done last,
> what is next with the facts it needs, and what is open. A session that ends updates it; a
> session that starts reads it after `AGENTS.md`. History belongs in git and the notes it points
> to, not here: replace what is stale rather than appending.

> Sources: original to this repository; [[Feature Backlog]], [[Refactor 2026-11]], `AGENTS.md`

**Last updated:** 2026-11-17, after P27 (a Gaussian mixture times an extended Gaussian), committed on `dev-danielb`, not pushed.

## Branches and worktrees

| checkout | branch | state |
|---|---|---|
| `~/Code/CategoryTheory/Lenticulum.jl` | `dev-danielb` | P2, P6 and P27 committed. Not pushed |
| `~/Code/CategoryTheory/Lenticulum.jl-mistral` | `mistral/review` | at `7d1014f8` with the P27 diff applied; Mistral's probes in `scratch/p27/` (copied to `../Lenticulum.jl-Review-2026-11-17/p27/`); its environment `scratch/p6env` (packages developed from the worktree, plus ForwardDiff, Zygote, Test) serves any review. Before the next review: `git reset --hard` to the commit under review, then apply its diff |

Pushing: a plain `git push origin dev-danielb` works from every shell (`~/.ssh/config` maps
`github.com` to the `mathstruct` key). If GitHub itself rejects the key, push over HTTPS with `gh`'s
credentials (`git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/MathStruct/Lenticulum.jl.git dev-danielb`).

Running one Lenticulum test file alone: the package has no test project (test dependencies are
`[extras]`), so make a scratch environment that develops `lib/LenticulumCore.jl`, `lib/Mycelium.jl`,
`lib/MyceliumLearning.jl`, `lib/Lenticulum.jl` and adds Test, LuxCore, ForwardDiff, Zygote, ADTypes,
DifferentiationInterface, then `julia -t auto --project=<env> -e 'include("lib/Lenticulum.jl/test/<file>.jl")'`.

## Done last

- **P27** (committed): a `SwitchingConstraintFactor` (BP or EP rule) on a variable a
  `HardConstraintFactor` reaches. Mycelium: `combine_with_overlap` of a Gaussian and an
  `ExtendedGaussian` (`_extended_overlap`), `ep_project`/`ep_divide` of extended Gaussians, and an
  indefinite Gaussian times an extended one (`_potential_conjoin`). Lenticulum: the relation factors
  collapse mixture inputs, also of extended components, to their moments (`HardConstraintFactor`
  refused them). Details in Mycelium [[lib/Mycelium.jl/src/gaussian|gaussian]] "Mixtures meeting
  extended Gaussians" and [[Backlog Done 2026-11]].
- Review: [[LP-046 Findings of the review of P27]] (Mistral; `ep_divide` did not check the hard values,
  fixed; a far-from-origin report not reproduced).
- Verified: the suites of Mycelium, MyceliumLearning, Lenticulum, VariationalDiffusion,
  AlgebraicFactors, AdversarialFactors and ImplicitLayers pass (logs in `~/.cache/lenticulum/p27/`);
  the five dependents ran before the F1 fix (one `return nothing` in `ep_divide` of extended
  Gaussians, which none of them calls); Mycelium and Lenticulum ran after it. LenticulumCore not
  rerun (no change there). Vault build and `docs/make.jl` not run.

## Next

No row is set; pick one with the maintainer. Close to P27 and P2: P5 (a singular nonzero noise in
`HardConstraintFactor`'s free energy), a learned noise with hard messages (refused in
`GaussianFactor`/`GaussianPrior`), P3 (`TrivialBelief()` on the exact flat path). Left open by P27
(no rows yet): the Bethe free energy with a mixture marginal under BP (no `variable_entropy` for
mixtures, Gaussian ones included), derivatives through the new rules, two extended components with
hard directions.

## Open

- **P4 limits:** Zygote has no rank check (a rank-changing tangent gives the constant-rank
  derivative silently); second derivatives (nested duals) through extended Gaussians are refused;
  `_moving_flat_split` (P25) stays ForwardDiff-only.
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
- Mycelium's `GaussianBlock` constructor and `belief_block` are not Zygote-safe (LP-044); a
  Zygote pass that builds blocks inside it fails there.
- AlgebraicFactors, ImplicitLayers and AdversarialFactors run in CI only weekly: a change to the core
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
  On P6 (2026-11-17) the brief gave the exact command line, a ready environment (`scratch/p6env`),
  "never prefix `cd`" and the answer-file rule: ten probes, a finished answer file, three right
  findings, \$0. Keep that brief as the model (`../Lenticulum.jl-Review-2026-11-17/p6/review-brief.md`). On P27 the same brief form gave five probes and an answer
  file with four of them at the 100-turn limit (\$0): one right finding; its fifth probe's reference lost
  digits at 1e6 (a false report): check a reviewer's reference before its claim.
