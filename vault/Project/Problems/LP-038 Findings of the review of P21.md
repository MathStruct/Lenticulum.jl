#open-problem

> **Status: resolved (2026-11-16).** The review found no defect in P21 (ForwardDiff through a relation
> factor's flat path with only η dual, and `NonlinearFactor` at a dual expansion point). It ran no
> probe: every Julia command was refused by its approval callback. Its seven probes were run
> afterwards; their failures were errors of the probes. The gaps it named were probed again by hand
> and passed, and two of them (the switching factor, two flat inputs) are tests now. One behaviour
> was found on the way and documented, not changed: a partial of η along null(Λ) is dropped.
> Related: [[Feature Backlog]] (P21, P25), [[lib/Lenticulum.jl/src/gaussian|gaussian]] "Dual inputs
> on the flat path".

> Sources: a review of the uncommitted P21 change by Mistral (`vibe --agent review`, `--max-price 3`,
> $1.12 at list price), in the helper worktree `Lenticulum.jl-mistral` at `5eac9e5e` with the diff
> applied; the brief, the diff, the answer file, its probes with their outputs, and the repair probe
> `probe_mine.jl` in `../Lenticulum.jl-Review-2026-11-16/p21/`.
> code: `lib/Lenticulum.jl/src/gaussian.jl` (`_improper_split_dual`), `lib/Lenticulum.jl/src/nonlinear.jl`
> (`_jacobian`), `lib/LenticulumCore.jl/src/extended_gaussian.jl` (`_flat_split`);
> tests: `lib/Lenticulum.jl/test/forms_crossing.jl`.

**Problem.** The requirement given to the reviewer: with a flagged input of dual η and a precision
without derivative, every relation factor's message works, equals the value path's and has its
central differences as partials; a precision with nonzero partials throws an `ArgumentError`;
`NonlinearFactor` at a dual expansion point likewise, both methods, with and without a `jacobian`;
the `Float64` path and `extended`'s refusals are unchanged.

| # | finding or probe | right? | what was done | test |
|---|---|---|---|---|
| R1 | `SwitchingConstraintFactor` not tested | a gap, not a defect: `probe_mine.jl` passes (40 instances, flat x to z and to k, 212 checks) | added as a testset | "P21: the switching factor and two flat inputs" |
| R2 | two flat inputs into one `LinearConstraintFactor`, several flat directions | a gap: passes (40 instances, 64 checks) | added (same testset) | same |
| R3 | flat input of dimension 1, η exactly 0, an `Int` Jacobian, Dual tags other than `Nothing` | gaps: η with value 0 and nonzero partials, the `Int` Jacobian and a tagged `ForwardDiff.derivative` pass in `probe_mine.jl`. Dimension 1: a flat input there is flat everywhere, the `_flat_part` path, not this change | none | `probe_mine.jl` only |
| P1–P3 | its switching probes | failed on their own errors: they expect the value path's source to be a `GaussianBelief` (it is an `ExtendedGaussian`) and call a `CategoricalBelief` constructor that does not exist | — | — |
| P4, P7 | "zero-partial duals accepted", "mixed partials", "extended refusals" | errored with "information vector outside the range": the probes draw η at random, outside range(Λ), which `extended` rightly refuses (requirement 5) | — | — |
| P5 | NonlinearFactor derivatives differ from central differences | not reproduced: the probe's `msg(t)` draws a new direction `randn(rng, nin)` on every call, so the four evaluations move along different directions | — | "P21: NonlinearFactor's message at a dual expansion point" (160 checks) |
| P6 | several flat directions: "both messages non-trivial" | the probe's expectation: with two flat directions both paths give `TrivialBelief()` here (agreeing); R2 checks the kinds agree | — | — |
| X1 | found while repairing: a partial of η along null(Λ) is dropped (derivative 0), where the value path refuses the input at every t ≠ 0 | right, as a behaviour; the factors' own messages keep η in range(Λ) for every t, so only a hand-built input reaches it | documented in [[lib/Lenticulum.jl/src/gaussian\|gaussian]] "What it does not do" | not tested |

**Leaning / resolution.** No code change from the review; two testsets' worth of coverage added
(R1, R2). The review agent's Julia commands were refused this time although the brief gave the exact
command line that worked for P23; why was not looked into.
