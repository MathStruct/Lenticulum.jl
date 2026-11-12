#open-problem

> **Status: resolved (2026-11-12).** One defect found and fixed, a ForwardDiff crossing the first
> version of P1 left open. Also: one test that did not build what its comment said, one header that
> overclaimed (a pre-existing limit, now `@test_broken`) and stale note text. One finding is not a
> defect. Related: [[Feature Backlog]] (P1, P21), [[LP-027 Findings of the review of the I18 plans]].

> Sources: a review of the uncommitted P1 change by Mistral (`vibe --agent review`), run with Julia in
> the helper worktree `Lenticulum.jl-mistral` (2026-11-12). Its answer is in
> `../Lenticulum.jl-Review-2026-11-12/P1-from-mistral-2.md` (the first run stopped at its turn limit),
> its scratch scripts are under `scratch/p1/` in that worktree (`probe_req1.jl`, `probe_req2.jl`,
> `probe_stack.jl`, `probe_nonlinear.jl`, `probe_jlarrays.jl`).
> code: `lib/Mycelium.jl/src/gaussian.jl` (`combine`, `belief_distance`, `can_damp`, `_damp` for an
> `ExtendedGaussian` and a `GaussianBelief`), `lib/Lenticulum.jl/src/gaussian.jl` (`_likelihood_message`);
> tests: `lib/Mycelium.jl/test/forms_crossing.jl`, `lib/Lenticulum.jl/test/forms_crossing.jl`;
> notes: `lib/Lenticulum.jl/src/gaussian.md` "Where the two forms meet", `lib/Mycelium.jl/src/gaussian.md`
> "Extended Gaussians".

**Problem.** P1 asked for tests where the two forms of a flat likelihood meet (a flagged
`GaussianBelief`, the form under ForwardDiff, and an `ExtendedGaussian`, the form on values). The change
reviewed fixed two defects those tests found (`_likelihood_message` built the extended form of a dual η;
`damp` was a silent no-op on a mixed pair). The brief gave Mistral the requirement, not the reasoning.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | A variable that hears a plain extended message (a rank-deficient factor with plain parameters) and a dual flagged one (the factor being differentiated) throws in `infer!` under ForwardDiff: `combine` and `belief_distance` converted the `GaussianBelief` side with `extended`, which has no method for duals | right. Reproduced by Mistral (`probe_req1.jl`, R1b) and by the extended test: against the code before the fix the two-factor inference errs in all 10 instances with `MethodError: no method matching svd!(::Matrix{ForwardDiff.Dual...})` | a mixed pair whose `GaussianBelief` holds non-float numbers is taken in canonical form when the extended side has no hard directions (`_canonical`); with hard ones `combine` and `belief_distance` throw an `ArgumentError` and `can_damp` is false. Plain-value pairs are unchanged | Lenticulum `test/forms_crossing.jl`, "ForwardDiff through an inference with flat backward messages" (two factors on `x`, derivative against the dense joint); Mycelium `test/forms_crossing.jl`, "a mixed pair with dual numbers goes through the canonical form" (60 random pairs, 1 080 checks) |
| F2 | The Mycelium test said "a mixed pair with dual numbers" but built `Real` arrays | right | replaced by the testset of F1 with ForwardDiff duals | as F1 |
| F3 | The Lenticulum test's header claimed every parameter dual for all three factors; `NonlinearFactor` with a dual expansion point throws earlier, in `_jacobian` (`Float64(::Dual)`) | right; the limit predates P1 | header corrected; the limit recorded as `@test_broken` and in P21 | Lenticulum `test/forms_crossing.jl`, "known limit, NonlinearFactor's backward message at a dual expansion point" |
| F4 | `gaussian.md` still said the crossings are untested and that `_likelihood_message` checks "the element type"; the backlog listed P1 as proposed | right for the worktree's copy, which did not have the note changes; "the element type" was still stale in the main checkout | corrected | — |
| F5 | With JLArrays, the new mixed `damp` returns a host `ExtendedGaussian`, where before P1 it returned the device message undamped | right as a description, not a defect: `ExtendedGaussian` is a host type (`Vector`, `Matrix`), and `combine` of the same pair already returned it on the host before P1 | nothing changed | — |

**Leaning / resolution.** F1 was a real defect of the same kind as the one P1 fixed in
`_likelihood_message`: a place that decided by representation and forgot that under ForwardDiff
only some arrays carry duals. F2–F4 corrected tests and notes.

Not checked: the distance near the rank threshold on a run (`lib/Mycelium.jl/src/gaussian.md`); a
mixed pair where Λ, not only η, holds duals is covered by the same `_extendable` test but was not
run separately.
