#open-problem

> **Status: resolved (2026-11-17).** The review of P6 (`HardConstraintFactor` in plates and on
> `MeanField()` nodes) found two defects and one minor one, all fixed with regression tests: the
> per-replica path from message blocks failed under Zygote; the mean-field refusal also refused a
> positive definite noise, where the factor is a `LinearConstraintFactor`; mismatched argument
> lengths were not refused. A fourth point (missing note sections) came from the reviewer's copy,
> made before the note was written.
> Related: [[Feature Backlog]] (P6), [[hard_constraint]] "Plates (P6)" and "Mean-field nodes".

> Sources: a review of the uncommitted P6 change by Mistral (`vibe --agent review`, Mistral Large 4,
> `--max-price 3`, $0 against the budget), in the helper worktree `Lenticulum.jl-mistral` at
> `9038f2c0` with the diff applied. The brief, the answer, the probes, the maintainer's reproducer
> (`revert_check.jl`), the instrumented count (`count_shared.jl`) and the timing script are in
> `../Lenticulum.jl-Review-2026-11-17/p6/`.
> code: `lib/Lenticulum.jl/src/hard_constraint.jl`; tests: `lib/Lenticulum.jl/test/hard_plates.jl`.

**Problem.** The requirement given to the reviewer: `factor_messages` of a `HardConstraintFactor`
gives, for vector and block inputs, what `factor_message` gives replica by replica (messages up to
rounding, errors alike) for every kind of input and noise; the derivative of the per-replica path
under ForwardDiff and Zygote; the mean-field refusal mathematically right; every use of the new
trusted `ExtendedGaussian` constructor passes a normalised representative.

The run kept its answer file, entry by entry, and finished it (ten probes, all of them ran).

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | Under Zygote the block method throws (`Mutating arrays is not supported`): its per-replica fallback turned the messages into a block with `Mycelium.belief_block`, whose `Any[...]` copy Zygote cannot take (`probe8b.jl`) | **right**: `revert_check.jl` gives the error with the pre-fix line. Not a regression (Mycelium's default block method goes through the same `belief_block`), but the requirement asked for it | The fallback builds `VectorBlock(Vector{Any}(beliefs))` directly; the messages are always `ExtendedGaussian`s, for which `belief_block` gives a `VectorBlock` too. Building a `GaussianBlock` inside a Zygote pass still fails, in Mycelium's constructor (`collect(Bool, improper)`): not this change's, not fixed | "P6: block inputs under Zygote, argument lengths (LP-044)": Zygote by `c` through the block method against ForwardDiff of the per-replica loss |
| 2 | The mean-field refusal is unconditional, but with a positive definite noise the factor is a `LinearConstraintFactor`, whose mean-field message exists and whose average energy is finite (`probe4.jl`: fit 0.515, complexity 1.266) | **right** | With a positive definite noise `mean_field_message` and `mean_field_energy` are those of `LinearConstraintFactor(dims, m; noise = Q)` (same parameter layout); zero and singular noise are refused with the reason | "P6: MeanField() nodes, LinearConstraintFactor's with a positive definite noise (LP-044)": 20 random factors, message and energy equal to the `LinearConstraintFactor`'s; the refusal testset now includes a singular noise |
| 3 | Mismatched lengths of factors, inputs and priors returned `length(inputs)` messages without an error (`probe10.jl`) | right (the old default threw a `DimensionMismatch` from `eachindex`) | A `DimensionMismatch` on the shared and the per-replica path | in the LP-044 testset |
| 4 | The docstrings cite the note sections "Plates" and "Mean-field nodes", which the note did not have | not a defect of the change: the diff was copied to the worktree before the sections were written; they are in [[hard_constraint]] | — | — |

Also checked by the reviewer and passed: the shared kernel's algebra against `preimage` and
`_from_potential`, 150 random and controlled instances (rank-deficient and wide `A_t`, a `Σ` shared
with a different `D`, an improper `η` outside the range, contradictions in replica 1 and 3), block
inputs of every kind with mixed improper flags, ForwardDiff with dual parameters and dual inputs,
Zygote by `c` and `A` including losses on `Σ` and the flat projector, `_within_gradient`, error
parity for mixtures, categoricals, non-PSD precisions and differing factors, ambiguities
(`Test.detect_ambiguities`), `st` threading, and the one use of `ExtendedGaussian(Val(:normalised), …)`.

Not probed: `MixtureBlock`, `ComponentMixtureBlock` and `CategoricalBlock` inputs beyond one case
each (they take the per-replica path); device arrays.

**Leaning / resolution.** Fixed (findings 1 to 3).
