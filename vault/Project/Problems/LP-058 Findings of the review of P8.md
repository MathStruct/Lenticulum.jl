#open-problem

> **Status: resolved (2026-11-18).** The review of P8 (the graph free energy in the element type of
> its messages) found two defects, both reproduced and fixed: a mixture of categoricals of different
> element types lost precision, and a graph of exact zeros returned an `Int`. A third finding,
> Lenticulum's Gaussian factors still widening to Float64, is right but outside P8: it is row P30.
> Related: [[Feature Backlog]] (P8, P30), [[LP-024 Findings of the review of 2026-11-09]] item 7,
> [[lib/Mycelium.jl/src/free_energy|free_energy]] "Element type in, element type out".

> Sources: implemented by the maintainer's session (Claude); reviewed by Codex (GPT, `codex exec -s
> read-only` in the main checkout with the diff uncommitted; the helper worktree was not reset). The
> brief, the diff and the answer are in `../Lenticulum.jl-Review-2026-11-18/p8/`. code:
> `lib/Mycelium.jl/src/free_energy.jl`, `lib/Mycelium.jl/src/factors.jl`,
> `lib/Mycelium.jl/src/gaussian.jl`, `lib/LenticulumCore.jl/src/gaussian_belief.jl`,
> `lib/LenticulumCore.jl/src/extended_gaussian.jl`; tests: `lib/Mycelium.jl/test/element_type.jl`.

**Problem.** The requirement given to the reviewer: a graph whose messages are all Float32 gives a
Float32 free energy (`scalar_free_energy`, `factor_free_energies`, `variable_corrections`, node by
node and on a plate's rows); a Float64 graph gives what it gave before; ForwardDiff and Zygote keep
working. Constant zeros are `false`, and `log(2π)` is computed in the value's type.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `variable_entropy` of a `MixtureBelief` of categoricals took its accumulator type from the first component only: a Float32 and a Float64 component gave `-0f0` instead of $2.3\cdot10^{-98}$ | yes, reproduced | the type is promoted over every component | `element_type.jl`, the mixture line |
| F2 | A graph whose every term is an exact zero (one `DataFactor`) returned `0::Int` from `scalar_free_energy`, where it gave `0.0` before | yes, reproduced | `scalar_free_energy` returns `float(total)` | `element_type.jl`, the zero-only graph |
| F3 | `GaussianPrior` (and `GaussianFactor`) energies in Lenticulum add a Float64 normaliser to Float32 data, so a Lenticulum Gaussian graph is still Float64 | yes (the session had measured the same for the messages) | outside P8, which is Mycelium's sum: row P30, with these places named | — |
| F4 | No ForwardDiff or Zygote test of the changed Gaussian entropy and log-partition | yes, a missing test; no defect | added: derivatives in the precision against the closed forms, both AD systems; they passed before and after | `element_type.jl`, the AD lines |

Not reviewed: the review fixes themselves (F1, F2) were not sent to a second review.
