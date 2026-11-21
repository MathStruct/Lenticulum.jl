#open-problem

> **Status: resolved (2026-11-21).** A review of I17, which makes reverse-mode AD (Zygote) through the
> coupled gradient drivers, through `infer!` and through the implicit graph adjoint linear in the graph:
> the replay is one function with one `rrule` over message versions, and each task reads a local store and
> its own parameter subtree. Its requirement is that no value or derivative changes. Mistral found no
> defect in six probes; three of its probes failed on mistakes of their own, which the session fixed and
> ran again (all passed), and one finding was not reproduced end to end. The defects that were found came
> from measuring, not from the review: the first version's Krylov product was still superlinear in its
> forward pass, for three reasons, fixed with a tighter regression test.
> Related: [[Feature Backlog]] (I17), [[Backlog Done 2026-11]], [[LP-073 Findings of the review of I16]].

> Sources: review by Mistral (Mistral Large, `vibe --agent review`, 100 turns, probes 0–4b), with the
> requirement and the diff, not the reasoning. Brief, diff and answer: `../Lenticulum.jl-Review-2026-11-21/i17/`;
> probes in `../Lenticulum.jl-mistral/scratch/i17r/` (copied to the review folder).
> code: `lib/MyceliumLearning.jl/src/gradient_couplings.jl` (`_replay_forward`, `_replay_run`, `_task_step`,
> `_task_reads`, `_graph_scalar`, `_pinned_parameters`), `graph_adjoint.jl` (`_coordinate_store`,
> `_graph_update`, `_update_part`), `ext/MyceliumLearningChainRulesCoreExt.jl`;
> `lib/Mycelium.jl/ext/MyceliumChainRulesCoreExt.jl` (`_read_slots`); `gradient_couplings.md` ("Reverse mode
> by message versions"). tests: `lib/Lenticulum.jl/test/coupled_keys.jl`, `coupled_spec.jl`.

**Problem.** The change replaces three folds that Zygote differentiated with hand-written rules; a wrong
rule gives a wrong gradient silently. The reviewer compared the new code with the old (copied as `_old_*`)
under Zygote, ForwardDiff and reference differences.

## Mistral

| # | probe | result | what was done |
|---|---|---|---|
| 0 | every schedule × `nsweeps` 1–3 × damping 0, 0.3 on a small tree, Zygote against ForwardDiff | passed, except a flooding sweep of one: Zygote 2.1e14 against ForwardDiff 0.24 | probe 1b: the old code gives the same values and the same disagreement bit for bit; a single flooding sweep leaves the Gaussian factor's ports uninformative (improper messages). Not a regression; no row (an improper message under Zygote is outside I17) |
| 2 | five schedules × `nsweeps` × damping × four graphs (tree, loop, mixed diagonal/detached/frozen, Float32), old against new | passed (tapes `==`, values `==`, gradients ≈ the old under Zygote and ForwardDiff) | — |
| 3 | state versions: a tied stateful key, a key absent from `st`, `st = nothing` and `(;)`, a task repeated in one sweep | passed | — |
| 4 | the rules called directly with hand-made cotangents | crashed: a belief cotangent `(; η, Λ)` without `improper` reached Zygote's pullback of the task, which read the field | **not reproduced end to end**: `belief_cov`'s rule does return such a `Tangent`, but a loss through it on the final messages (probe 4b, case E, by the session) gives the old gradient, as Zygote fills the field in before the rule sees it. With the field given, the probe's remaining checks pass (its own flatten helper needed a fix for `Tangent`s) |
| 4b | losses of the final messages' η only, of the final state only, of the score only | failed for two probe mistakes: the old tape was built inside AD (`push!` refused), and the old score tape was drawn from the `rng` the new tape had already used | fixed by the session (`probe4b_fixed.jl`): all five losses equal the old ones under Zygote and ForwardDiff |

Not checked by the reviewer: the package suites, device arrays, the implicit adjoint's Krylov path beyond the
supplied tests.

## Found by measuring (the session)

The first version (Codex) made the unrolled gradient linear (27 → 58 → 124 MB from 31 to 121 factors), but
one Krylov product of the implicit adjoint still grew 2.5–3.1× per doubling (29 → 80 → 261 → 1 073 ms from
31 to 241 factors). Its backward pass was linear; the forward pass of Zygote's pullback was not. Three
causes, each located with an allocation profile at 61 and 241 factors:

| # | cause | fix | evidence |
|---|---|---|---|
| 1 | the I16 rule of `_subtrees` read `getfield(ps, i)` in code specialised on `ps` (a field per factor): O(keys) per read, O(nodes × keys) per call. Affects I16's free energy too | `_read_slots`, `@noinline @nospecialize` (a `@nospecialize` helper alone, or a read through `_IndexedTree`, was inlined back) | the rule's forward pass at 241 keys, 722 nodes: 10.7 → 0.13 MB |
| 2 | `_same_message_layout`, a check, traced by Zygote, generated a pullback per task at run time | it and `_check_pathwise` are `@non_differentiable` | 45.6 of 139.8 MB of the update's forward pass at 241 factors |
| 3 | the update's `map` captured `ctx`, whose type holds the type of `ps`; every dynamically dispatched task call cost O(keys) | the body is `_update_part` of small arguments; the closure captures `g`, the damping and the backend | the update's forward pass 20 → 53 → 157 MB before, 12.1 → 24.2 → 48.0 MB after (61, 121, 241 factors) |

**Regression test.** Codex's allocation test (6 and 24 steps, bound 8×) passed on the first version. It now
runs at 10 and 40 steps with a bound of 5×: the final code gives 4.05× and 4.05×, the first version 4.11× and
5.14× (fails on the product), the code before I17 8.9× for the unrolled gradient
(`~/.cache/lenticulum/i17/ck-{final,codex}.log`).
