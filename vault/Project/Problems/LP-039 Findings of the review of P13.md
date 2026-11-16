#open-problem

> **Status: resolved (2026-11-16).** The review of P13 (the exactness of a tree sweep checked at run
> time) ran four probes. Its one finding, that `ExposedFactor` and `OverlayFactor` do not declare
> `exact_messages` and so make a report say "not exact", is right as an observation and intended:
> the declaration is opt-in, and delegating to the wrapped factor would claim exactness for an
> exposed factor whose parameter channels carry uncertain beliefs. No code change. Related:
> [[Feature Backlog]] (P13), [[lib/Mycelium.jl/src/passing|passing]] "Exactness checked at run time".

> Sources: a review of the uncommitted P13 change by Mistral (`vibe --agent review`, `--max-price 3`,
> $0.98 at list price), in the helper worktree `Lenticulum.jl-mistral` at `86da7a51` with the diff
> applied; the brief, the diff, the answer file and the probes with their outputs, and the repaired
> probe `probe2_fixed.jl`, in `../Lenticulum.jl-Review-2026-11-16/p13/`.
> code: `lib/Mycelium.jl/src/passing.jl` (`_inexact_reasons`, `exact_messages`, `ConvergenceReport`),
> `lib/Lenticulum.jl/src/Lenticulum.jl` (the declarations); tests: `lib/Lenticulum.jl/test/exact_report.jl`.

**Problem.** The requirement given to the reviewer: `report.exact` true only when the marginals are
exact, false whenever a step approximated (a mixture collapsed, a `MixtureReduction` node, an
undeclared factor); the reason names each; iterated schedules never exact; the four-argument
constructor still works; declared graphs without mixtures exact, and their marginals then equal a
closed form. Asked especially for false claims of exactness.

| # | finding or probe | right? | what was done | test |
|---|---|---|---|---|
| F1 | `ExposedFactor` and `OverlayFactor` return `exact_messages == false` although they wrap exact factors; "add methods that delegate to the contained factors" | the observation is right (`probe4.out`); it is the intended cautious direction, documented in `Lenticulum.jl` and in [[lib/Mycelium.jl/src/passing\|passing]]. The proposed fix is wrong for `ExposedFactor`: its parameters are channels, and with a non-point belief on them the factor is bilinear in its inputs, its messages not exact; whether they are points is known only at run time. `OverlayFactor` was not examined | none | — |
| P1 | the test file and its helpers | ran and passed (`probe1.out`) | — | `test/exact_report.jl` |
| P2 | a `PriorFactor` emitting a mixture into a `LinearConstraintFactor` | the probe errored on its own construction (mixture components declared improper, and an earlier version a disconnected graph). Repaired (`probe2_fixed.jl`): the report says not exact, "node :obs receives a mixture on :y and collapses it" | — | `probe2_fixed.jl` only |
| P3 | an `ExposedFactor` in a tree | did not run past a constructor error of the probe; its conclusion is F1, by reading | — | — |

No false claim of exactness was found. Not probed: a cluster node receiving a mixture, plates
(`PlateSchedule` is not a `TreeSchedule`, so it never claims exactness), stale messages from an
earlier run.

**Reviewer's tooling.** Its six refused commands all began with `cd … &&`; its `julia` commands,
written without output redirection, ran. On P21 the refused `julia` command had a redirection
(`> … 2>&1`), which P23 ran; the cause is not settled.

**Leaning / resolution.** No change. A declaration for wrappers, if wanted, has to look at the run:
an exposed factor is exact when every parameter channel holds a point.
