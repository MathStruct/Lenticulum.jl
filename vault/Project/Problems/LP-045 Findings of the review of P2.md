#open-problem

> **Status: resolved (2026-11-17).** The review of P2 (hard inputs in the free energy of
> `SwitchingConstraintFactor`, in the frozen factor beliefs of the gradient, and in plates) found no
> defect. One gap it left (Zygote through the switching term with hard inputs) was probed by the
> maintainer and is now a testset; one remark is recorded as unverified.
> Related: [[Feature Backlog]] (P2), [[switching]] "Hard inputs in the free energy",
> [[lib/Lenticulum.jl/src/gaussian|gaussian]] "Free energy with hard inputs".

> Sources: a review of the uncommitted P2 change by Mistral (`vibe --agent review`, Mistral Large 4,
> `--max-price 3`, $0 against the budget; list price \$0.83), in the helper worktree
> `Lenticulum.jl-mistral` at `01b0adea` with the diff applied. The brief, the answer, the probes and
> the maintainer's scripts are in `../Lenticulum.jl-Review-2026-11-17/p2/`.
> code: `lib/Lenticulum.jl/src/switching.jl`, `lib/Lenticulum.jl/src/gaussian.jl`; tests:
> `lib/Lenticulum.jl/test/hard_inputs.jl`.

**Problem.** The requirement given to the reviewer: the switching factor's term takes hard inputs
(the softened limit plus $\tfrac h2\log 2\pi e\varepsilon$; `LinearConstraintFactor`'s at $K = 1$);
`belief_free_energy ∘ factor_belief` equals `local_free_energy` for the four Gaussian-family
factors with hard inputs and its derivative is the free-energy gradient; a learned noise keeps
refusing hard messages; a plate's free energy and gradient are the sums over its replicas.

The run kept its answer file and finished it (five probes). Its first probe reported 61 of 120
mismatches, which it traced to its own probe (a precision used as a covariance, flat directions
counted as hard); the corrected probe (`probe1d.jl`) agrees to $10^{-9}$.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | No defect: the softened limit holds, all-point inputs give finite terms, the shipped tests pass | — | — | `test/hard_inputs.jl` |
| 2 | Not covered by the review: Zygote and ForwardDiff through the switching term in `ps` with hard inputs | a gap, not a defect: the maintainer's probe (`maintainer_zygote.jl`) gives Zygote against ForwardDiff to $5.6\cdot10^{-16}$ (27 random factors) | kept as a testset | "P2: Zygote through the switching term with hard inputs, against ForwardDiff" |
| 3 | `factor_belief` may refuse (Cholesky of the belief precision) where `local_free_energy` returns a value, on nearly singular instances | **unverified**: no script shows it. Both paths factorise the same precision for the switching factor; for `GaussianFactor` the local path inverts through `GaussianBelief`. A refusal, not a wrong number | none | none |
| 4 | Not covered: the switching factor's batched plate methods with extended blocks, and a hard channel ordered before a point channel | the batched methods return `nothing` for `VectorBlock`s, so plates go per replica (the plate testset passes through that path); the ordering is exercised by the random channel kinds of the LP-020 testset | none | — |
