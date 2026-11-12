#open-problem

> **Status: resolved (2026-11-12).** One defect found and fixed with a test; one suspicion not
> reproduced. Related: [[Feature Backlog]] (I5), [[LP-024 Findings of the review of 2026-11-09]].

> Sources: a review of the uncommitted I5 change (the parameter and state trees read by position)
> by Mistral, run with Julia in the helper worktree `Lenticulum.jl-mistral` (2026-11-12). It
> stopped twice at its turn limit before writing a final list, so the findings below come from its
> scratch scripts there (`scratch/i5_review.jl`, `scratch/i5_ad.jl`, `scratch/i5_plate_scaling.jl`),
> which the maintainer's session ran again. code: `lib/Mycelium.jl/src/passing.jl`,
> `lib/MyceliumLearning.jl/src/gradient.jl`; note: `lib/Mycelium.jl/src/passing.md`, "The trees by position".

**Problem.** Mistral checked the I5 change against its requirement: every schedule returns a
`NamedTuple`, and the very tree it was given when there is no state; missing keys are appended in
the order of their first write; the tied rule (the last write wins); plates with block rows and a
stateful factor; residual probes do not write; `step!` on a `NamedTuple`; Zygote and ForwardDiff
through the free energy against the reference; `parameter_step` and `free_energy_gradient`
against finite differences. Rerun by the maintainer's session: 195 checks pass in
`i5_review.jl` and 26 in `i5_ad.jl`. The one error in `i5_review.jl` is a defect of the scratch
test itself (`MaybeCount` has no `parameterlength`), which `i5_ad.jl` corrects.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | `parameter_step` with an optimiser state given as a `Dict` throws `MethodError: no method matching _stored(::Dict…)`; the old code read it with `haskey` and `getindex`, so a `Dict` worked | right, reproduced (`i5_ad.jl`). The docstring asks for a `NamedTuple`, but the old code accepted a `Dict` and nothing announced a change | `_stored(d::AbstractDict, name)` = `get(d, name, nothing)` | MyceliumLearning `test/gradient_couplings.jl`, "parameter_step: an optimiser state given as a Dict": one step with a `Dict` equals one with the `NamedTuple`; it fails without the method |
| 2 | `_row_items` scans all energy rows for each key in `_envelope_gradient`: O(keys × rows) | not reproduced: Mistral's `i5_plate_scaling.jl` found the cost linear at the sizes it tried. The rows are few (one per plate and channel set), so it would matter only for graphs with many plates | nothing changed | — |

**Leaning / resolution.** Fixed with a test where reproduced; the second is recorded only.
