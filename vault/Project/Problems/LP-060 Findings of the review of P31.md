#open-problem

> **Status: resolved (2026-11-18).** The review of P31 (the rest of Lenticulum in the element type
> of its parameters) found no defect. The reviewer recorded only its baseline before its turn limit;
> its two probes, run by the maintainer's session (one after a one-line repair), show Float64 results
> bit for bit unchanged and Float32 results throughout the factors in scope.
> Related: [[Feature Backlog]] (P31, P32), [[LP-059 Findings of the review of P30]],
> [[lib/Lenticulum.jl/src/switching|switching]] "Element type".

> Sources: implemented by Codex (GPT, `codex exec -C ../Lenticulum.jl-codex`, branch `codex/p31`, one
> run); reviewed by Mistral Large 4 (`vibe --agent review` in `../Lenticulum.jl-mistral`, \$0, stopped
> at its 100-turn limit). Brief, probes, outputs and Codex's inventory of what is left
> (`remaining-float64.md`): `../Lenticulum.jl-Review-2026-11-18/p31/`. code:
> `lib/Lenticulum.jl/src/switching.jl`, `regime.jl`, `nonlinear.jl`, `likelihood.jl`, `m_steps.jl`,
> `hard_constraint.jl`, `exposure.jl`, `coupling.jl`, `constraint.jl`; tests:
> `lib/Lenticulum.jl/test/p31_element_type.jl`.

**Problem.** The requirement given to the reviewer: with Float32 parameters, noise and data, every
message, marginal and free-energy term of `SwitchingConstraintFactor`, `RegimeFactor`,
`NonlinearFactor`, `LikelihoodFactor` (Laplace, EP), the M-steps, the shared plate kernel of
`HardConstraintFactor`, exposed parameters and `coupling_weights` is Float32 or `false`; no conversion
of data or parameters (constants are cast at use); Float64 results bit for bit unchanged;
ForwardDiff, Zygote and Int inputs keep working.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| R1 | baseline: `p31_element_type.jl` 1347 pass, 128 broken, 0 fail | — | — | `p31_element_type.jl` |
| R2 | probe 2 (run by the session; its last block needed `let;` before a tuple assignment): Float64 switching (latent and observed outcome, plate), regime (and plate), nonlinear (finite differences, user Jacobian, `Unscented()`, a rank-0 Jacobian), Poisson / bivariate Poisson / ordinal-probit likelihoods (Laplace, EP, large counts), hard-constraint shared kernel, exposed parameters, one EM iteration, `coupling_weights`, `partial_sums`, printed with `repr` under the old code (`6157f273`) and the new | no defect: every number identical; only exact zeros `0.0` became `false`, and `LikelihoodFactor`'s default `A`/`b` are now `Matrix{Bool}`/`Vector{Bool}` (same products) | — | `probe2_f64_same.jl`, `out_old.txt`, `out_new.txt` |
| R3 | probe 2: ForwardDiff through `log_evidence` and the message of a `LikelihoodFactor` | improvement, not a defect: the old code threw `Float64(::Dual)`; the new gives a derivative | — | `probe2_f64_same.jl` ("lik dev", "lik dmsg") |
| R4 | probe 2: Zygote through the same | not a regression: fails in old and new code alike (`rrule(maximum, …; init)` missing; now `init::Bool`, before `init::Float64`) | none; Zygote through `LikelihoodFactor` was never supported | — |
| R5 | probe 1 (run by the session): 225 Float32 type and agreement checks over the factors in scope, including the shared hard kernel with Float32 Gaussian and point inputs, mixed Float32/Float64 channel inputs (promotion), default and explicit `partial_sums` noise | no defect | — | `probe1_f32_types.jl`, `probe1.log` |

Left open (rows, not defects): regime inference goes through Mycelium's cluster joints, whose
`_numtype` promotes with `Float64` (`lib/Mycelium.jl/src/cluster.jl`, 124 broken checks);
MyceliumLearning's `MomentStats` and `fit_em`'s history are Float64 (`em.jl`, 3 broken); Mycelium's
Gaussian overlap uses a Float64 `log(2π)` (`gaussian.jl`, 1 broken). Row P32.

Not reviewed by a second model; device tests (JLArrays) were not run for these factors.
