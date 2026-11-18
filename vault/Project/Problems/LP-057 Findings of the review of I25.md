#open-problem

> **Status: resolved (2026-11-18).** The review of I25 (`coupling_weights`, default spanning-tree
> weights from the factors) found no defect against the requirement. One observation (vector-shaped
> coefficients throw) is not a defect: inference refuses the same parameters.
> Related: [[Feature Backlog]] (I25), [[lib/Lenticulum.jl/src/coupling|coupling]],
> [[lib/Mycelium.jl/src/schedules|schedules]] "Choice of tree and alternating trees".

> Sources: implemented by the maintainer's session (Claude); reviewed by Mistral Large (`vibe --agent
> review`, \$0, stopped at its 100-turn limit) in the helper worktree `Lenticulum.jl-mistral` at
> `6340896f` with the diff applied: three probes, two recorded in the answer file; the maintainer ran
> the third. The brief, the answer, the probes and probe 3's output are in
> `../Lenticulum.jl-Review-2026-11-18/i25/`. code: `lib/Lenticulum.jl/src/coupling.jl`; tests:
> `lib/Lenticulum.jl/test/i25_coupling_weights.jl`.

**Problem.** The requirement given to the reviewer: `coupling_precision(f, ps)` gives each
linear-Gaussian factor's precision over its continuous channels (formulas per factor type, current
noise, a declared output's fixed coefficient, the switch excluded; `:hard`, `nothing` otherwise);
`coupling_weights(g, ps)` gives per factor the largest $\lVert D_u^{-1/2} J_{ab} D_v^{-1/2} \rVert_2$
over channel pairs on different variables (or the raw $\lVert J_{ab} \rVert_2$), $D_v$ the summed
diagonal blocks, with the stated rules for hard, unknown and one-variable factors, non-finite values
refused, and the result accepted by `spanning_tree_schedule`.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | Raw and normalised weights and $D_v$ against an independently assembled dense precision: a 3-channel constraint, two channels on one variable, an unconnected channel, a `GaussianFactor` 2 ⇒ 3 | no defect | — | probe 1 |
| F2 | Learned noise (`GaussianFactor`, `GaussianPrior` diagonal; `RegimeFactor` with `L`, `logw` and a declared output) read at the current parameters; a discrete `PriorFactor` gets `default` | no defect | — | probe 2 |
| F3 | `SwitchingConstraintFactor` in a graph with its switch connected; parameters tied across two nodes (`params`); a frozen constraint (15 checks, run by the maintainer) | no defect | — | probe 3 |
| O1 | Coefficients given as vectors (`A = (a = [1.0], …)`) throw `MethodError` | not a defect: `infer!` refuses the same parameter tree | — | probe 1b |

Not probed by the review: plates, ForwardDiff dual and Float32 parameters (duals are refused by the
`Float64` conversion, as `coupling.md` says), `HardConstraintFactor` with a partly positive noise.
