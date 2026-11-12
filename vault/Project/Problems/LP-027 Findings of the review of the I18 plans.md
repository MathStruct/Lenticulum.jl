#open-problem

> **Status: resolved (2026-11-12).** No defect was found. One claim of a compile-time regression was
> not reproduced, a gap in the tests (concurrency, several stores) was filled, and one note was not
> yet written when the review ran. Related: [[Feature Backlog]] (I18),
> [[LP-026 Findings of the review of I18]].

> Sources: a review of the uncommitted second part of I18 (the planned factor -> variable messages)
> by Mistral (`vibe --agent review`), run with Julia in the helper worktree `Lenticulum.jl-mistral`
> (2026-11-12). Its answer is in `../Lenticulum.jl-Review-2026-11-12/REVIEW-i18b-from-mistral-2.md`
> (the first run stopped at its turn limit), and its scratch tests are `scratch/i18b_equiv.jl`,
> `scratch/i18b_equiv2.jl`, `scratch/i18b_race.jl` and `scratch/i18b_compile.jl` in that worktree.
> code: `lib/Mycelium.jl/src/passing.jl` (`_EmitPlan`, `_emit_plan`, `_emit`),
> `lib/Mycelium.jl/src/graph.jl` (`emit_plans`); note: `lib/Mycelium.jl/src/polarity_resolution.md`,
> difficulty 3, "Planned per graph".

**Problem.** The change was required to give the same results, errors and warnings as the code it
replaced, for every graph, store and rule. Mistral re-implemented the old branch from the diff and
compared it with the new one, task by task and sweep by sweep: random graphs with belief-propagation,
expectation-propagation and mixture-reduction rules, detached edges, pins, damping, plate graphs with
block rows, factors of 63, 64, 65 and 70 edges, and `reset!`. It also ran 16 threads on one graph
with the same keys. It found the same beliefs, residuals and errors in each case, and no race.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | Each new set of informative inputs compiles a new `_emit` -> `_rule_message` -> `factor_message` chain, while the old code "compiled `factor_message` once per factor type"; `emit_plans` is not bounded | not reproduced as a regression. The old code passed the inputs as a `NamedTuple{chans}` too, so it also compiled once per set. A scratch comparison on a 20-channel factor (30 new sets on each path, not kept in the repository) took 2.09 s on the old path and 2.10 s on the new one. `emit_plans` grows like the `polarities` cache, which has one entry per (target, available channels) | nothing changed; recorded in `polarity_resolution.md` | — |
| F2 | no test of concurrent tasks on one graph, nor of several stores sharing one graph's plans | right (a gap; Mistral's scratch test found the property holds) | test added | Mycelium `test/emit_plan.jl`, "plans under concurrent tasks and several stores": 8 tasks with their own stores, 20 rounds, compared with the old path, then every stored plan with a fresh one |
| F3 | no test that the warning of the `TrivialBelief()` case fires | right, but the warning is unchanged code with `maxlog = 1`, which a test sees only on the first call of a session | not added; the outcome (`TrivialBelief()`) is tested | — |
| F4 | `polarity_resolution.md` still said the dispatch "remains" | right when the review ran; the note was written after the diff was taken | updated with the change | — |

**Leaning / resolution.** Only F2 called for code, and it got a test. F1 is recorded as not
reproduced.

Not checked: the compile time per new input set on a schedule that sees many different sets
(a residual schedule on a loopy graph); it is the same on both paths by the argument above but was
measured only on one factor.
