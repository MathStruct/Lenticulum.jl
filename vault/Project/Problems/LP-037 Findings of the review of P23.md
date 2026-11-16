#open-problem

> **Status: resolved (2026-11-16).** One finding: the change meant to route every `inv(_chol(M, what))`
> in Lenticulum through `_spd_inv` (which has a Zygote rule) and missed one, in `likelihood.jl`, whose
> argument had nested parentheses. The miss is real; the impact claimed for it (Zygote failing or
> giving wrong gradients through EP site messages) was not reproduced. The call now uses
> `_spd_inv` (the same function, so the primal is unchanged). Related: [[Feature Backlog]] (P23, P24),
> [[constraint]] "Zygote through the cavity free energy".

> Sources: a review of the uncommitted P23 change by Mistral (`vibe --agent review`, `--max-price 3`,
> $1.28 at list price), run with Julia in the helper worktree `Lenticulum.jl-mistral` at `b184cc82`
> with the diff applied; the brief, the diff, the answer file and the probes in
> `../Lenticulum.jl-Review-2026-11-16/p23/`.
> code: `lib/LenticulumCore.jl/src/gaussian_belief.jl` (`_spd_inv`),
> `lib/LenticulumCore.jl/ext/LenticulumCoreChainRulesCoreExt.jl`, `lib/Lenticulum.jl/src/constraint.jl`,
> `lib/Lenticulum.jl/src/switching.jl`, `lib/Lenticulum.jl/src/likelihood.jl`;
> tests: `lib/Lenticulum.jl/test/zygote_cavity.jl`, `lib/Lenticulum.jl/test/plates.jl`.

**Problem.** P23 made Zygote take the cavity free energies of `LinearConstraintFactor` and
`SwitchingConstraintFactor` in `ps`. The requirement given to the reviewer: Zygote works for every
message kind these factors accept and equals ForwardDiff; primal values, errors and element types
are unchanged; the rrule of `_spd_inv` is the exact pullback of `inv(_chol(M, name))`; marking
`_complement` non-differentiable drops no tangent a caller needs.

| # | finding or probe | right? | what was done | test |
|---|---|---|---|---|
| F1 | `likelihood.jl:568` (`_site_message`) still calls `inv(_chol(Matrix(Symmetric(Σζ)), ...))`; "any code path through `_site_message` will fail under Zygote or give wrong gradients" | the miss: yes (`grep`). The impact: not reproduced; the reviewer ran no probe through it, and it is a message path, not part of a free energy in `ps`. "Wrong gradients" cannot follow (Zygote errors on `inv(::Cholesky)`, it does not differentiate it wrongly) | replaced by `_spd_inv`, so the note's "every" holds; primal identical | none specific (the Lenticulum suite covers the primal) |
| P1–P6 | `_spd_inv` primal and rrule with symmetric and non-symmetric cotangents; primal values and `Float32`; both factors, 20 random instances each, Zygote against ForwardDiff; `_complement` and `_free_channels`; `_blockdiag` empty, `Float32`, under both ADs | probes ran and passed (`probe1_spd_inv.jl` to `probe6_blockdiag.jl`) | — | `test/zygote_cavity.jl` covers the same |

Not checked by the review: the test file itself (the reviewer's run stopped at a 120 s timeout of
its own), extended Gaussians with flat directions, Mooncake and Enzyme.

**Leaning / resolution.** F1 applied as above, with the impact recorded as not reproduced.
