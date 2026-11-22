#problem

> **Status: closed (2026-11-22).** I30 (the graph-wide trees boxed in the graph drivers' differentiated region)
> was checked three ways: Codex's first attempt stopped at the design's pinning step with a counterexample (the
> design was revised); the session compared 39 gradients recorded on the old code and found a defect of the
> implementation (the boxed state's cotangent dropped under Zygote), fixed; GLM reviewed the result with seven
> probes against the old code and found no defect, one suspicion of which pointed at wrong code (fixed with a
> test). Related: [[Feature Backlog]] (I30), [[LP-076 Findings of the review of I1]].

> Sources: Codex (GPT, `codex exec -s workspace-write`, the implementation brief; it stopped at step 3 and then
> hit its usage limit), the session's baseline comparison and profile, GLM-5.3 (`vibe --agent review --model glm-5-3`,
> \$2.96, with the requirement and the diff, not the reasoning). Briefs, diff, answers and probes:
> `../Lenticulum.jl-Review-2026-11-22/i30/`; the probes in `scratch/i30/` and `scratch/i30r/` of the Mistral worktree.
> code: `lib/MyceliumLearning.jl/src/gradient_couplings.jl` (`_graph_pin`, `_replay`, `_coupled_gradient`),
> `src/graph_adjoint.jl`, `ext/MyceliumLearningChainRulesCoreExt.jl` (`_part_erased`, the rule of `_replay_run`),
> `lib/Mycelium.jl/ext/MyceliumChainRulesCoreExt.jl` (the rules on `_Erased`), `lib/LenticulumCore.jl/src/parameter_tree.jl`
> (`_Erased`, `_ErasedPullback`); note `gradient_couplings.md`, "The graph-wide trees out of Zygote's signatures (I30)".
> tests: `lib/Lenticulum.jl/test/erased_trees.jl`.

## Codex: the design's step 3

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | pinning with `ifelse.(mask, θ, θref)` gives a `Float64` frozen leaf where the old per-key pinning gave a dual with zero partials, and a `Vector{Real}` | right (its probe) | the design's step 3 takes `_replace_coordinates`' promoted element type; test "I30: pinning in flat coordinates" |
| 2 | the old per-key flatten rounds a frozen value to its key's promoted type (`16_777_217` beside an active `Float32` became `16_777_216`); the graph-wide θ keeps it, so the gradient of the active coordinate changes from 0 to 1 | right (its probe) | taken as a deliberate change: the old value is a defect (a frozen value changed by its neighbour's type); the same test records the new value under the three backends |

## The session: values against the old code

| # | finding | right? | what was done |
|---|---|---|---|
| 3 | Zygote handed the replay rule the boxed state's cotangent as a `Tangent{Any}` or `(tree = Δ,)`, which `_part_erased` did not unbox: the state's cotangent was dropped (a stateful factor 13.80 for 66.36, a seeded score estimate 0.49 for 6.33) | right (`scratch/i30/compare_baseline.jl`: two of 39 records) | `_part_erased` takes every form; the 39 records then equal the old ones except a `-0.0` and one last-bit change of a tied key |
| 4 | after steps 1–6 Zygote still compiled 7.0 s at 121 factors: rule pullbacks captured the unboxed tree, two reads of `ps` accumulated whole-tree cotangents, `map`/`only` over the draws' results (SnoopCompile, `scratch/i30/snoop.jl`) | right (the profile) | `_ErasedPullback`, one read in `_replay`, a loop over the draws: 1.6 s; test "I30: the replay's Pullback type does not depend on the structure" |

## GLM

| # | finding | right? | what was done |
|---|---|---|---|
| 5 | no defect; probes 2 and 4 compare the unrolled and the implicit drivers with the old code (frozen and tied keys, a stateful factor, damping, several sweeps): bitwise for ForwardDiff and reference differences, the documented last bits for Zygote | — | none |
| 6 | suspicion: the rules of `_states` on a box of a tree that is not a `NamedTuple` read `_subtrees` in their fallback, right only for `absent = NamedTuple()` | right as code (unreachable through the drivers, whose `st` is a `NamedTuple` or `nothing`) | the fallback takes the caller's read; test in "I30: the box" |
| 7 | suspicion: `_part_erased` peels one level from any one-field `tree` cotangent | not a defect: the box adds exactly one such level, so a state tree with a key `tree` stays consistent | none |
| 8 | the 39-record baseline lives in scratch, not in the repository | right | the committed oracles are `coupled_keys.jl` and `coupled_spec.jl` (old functions); GLM's probes 2 and 4 re-checked frozen keys through the implicit adjoint, which those do not |

GLM also saw Zygote through `infer!` give 13.78 against ForwardDiff's 65.78 on a stateful factor and traced it
to I17's deliberate `stop_gradient` of the evolving state in reverse mode: the old code gives the same 13.78
bitwise. Not a change of I30.

**Open.** None of the findings. Not run: the device tests (the device form of `_pin_coordinates_wide` is
untested), the docs builds; the implicit adjoint still reads `ps` twice (`gradient_couplings.md`, "Not done").
