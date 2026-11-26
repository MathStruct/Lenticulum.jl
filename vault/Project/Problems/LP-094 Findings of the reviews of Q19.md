#problem

> **Status: closed (2026-11-26).** Q19 (element types on every message path) and what its reviews found.
> Codex extended the shared element-type test to mixture, mean-field, plate, frozen-belief and gradient
> messages, with fixtures for learned covariances and integer parameters, fixed the four defects that
> showed, and ran the BigFloat cases through GenericLinearAlgebra.jl, a test-only dependency, which cleared
> all 38 of Q19's `@test_broken`. The session found one defect in that change: a dual prior mean lost its
> derivative. Mistral's review found none. Gemini's ten findings: one was not reproduced, half of one is
> the intended conversion. Of the rest only the BigFloat error message was introduced by Q19; the others
> were already on `dev-danielb`, of the kind Q19 exists to catch (integer parameters widening a Float32
> call, a dual stored in the factor throwing, a plate rounding a replica's offset), and the test did not
> check derivatives at all. All fixed, each with a test; the matrix now compares every dual's partials with
> a central difference. Related: [[LP-091 Findings of the review of Q4|LP-091]],
> [[Backlog Q Quality and Engineering]] (Q4, Q19, Q20).

> Sources: Codex (`gpt-6.2-sol`, part 1, report in `../Lenticulum.jl-codex/scratch/q19/REPORT.md`); the
> session's probes; Mistral Large (`vibe --agent review`, \$0, 100 turns, turn limit after seven probes,
> answer file and probes in `scratch/q19r/` of `../Lenticulum.jl-mistral`); Gemini Hydrogen (`agy --mode
> plan`, `gemini-4-hydrogen-high`, read-only, ten findings); Mistral Large implementing part 2 (`vibe
> --agent accept-edits`, \$0, two runs, report in `scratch/q19p2/REPORT.md` there). code:
> `lib/LenticulumCore.jl/test/support/eltype_matrix.jl`, `lib/Lenticulum.jl/test/support/eltype_fixtures.jl`,
> `lib/Lenticulum.jl/test/lp094.jl`, `lib/AlgebraicFactors.jl/test/lp094.jl`,
> `lib/Lenticulum.jl/src/gaussian.jl`, `lib/Lenticulum.jl/src/constraint.jl`, `lib/Lenticulum.jl/src/switching.jl`,
> `lib/Lenticulum.jl/src/scale_shift.jl`, `lib/Lenticulum.jl/src/likelihood.jl`, `lib/AlgebraicFactors.jl/src/toric.jl`,
> `lib/Mycelium.jl/src/cluster.jl`.

## Found by the session

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | Codex's `_prior_relation` built x − μ in the covariance's type, which a prior's stored mean does not enter: with a dual μ, `ForwardDiff` through `factor_belief`/`belief_free_energy` (single and plate) threw `Float64(::Dual)`; `dev-danielb` gave the derivative of the central difference | yes (reproduced on Codex's tree; passes on `dev-danielb`) | the stored mean promotes the type when it carries a derivative (`_derivative_carrier`, see G4) | `lp094.jl` "finding 1" |

## Mistral's review

No defect. Seven probes (`scratch/q19r/probe_a.jl` … `probe_f.jl`): the prior's frozen paths bitwise
against the old code; every operation of six fixtures in duals against central differences (0
mismatches apart from the stopped `ScaleShiftBelief` energy, by design); the hard-constraint plates' shared
and fallback paths; the two BigFloat kernels with and without GenericLinearAlgebra; and that the test leaves
an operation out only where its method is the generic fallback (`probe_f.jl`, whose run the session did
after the turn limit: all seven checks right).

## Gemini's review

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| G1 | `HardConstraintFactor`'s `local_free_energy` with a dual offset throws without GenericLinearAlgebra | not reproduced: returns a dual | — | — |
| G2 | the BigFloat widening of `_range_complement` and `_flat_factor` skips their `ArgumentError`: without GenericLinearAlgebra a bare `MethodError` from `svd!`/`eigen!` | yes, introduced by Q19 | for a non-BLAS type the kernels check `hasmethod(svd!, …)` / `hasmethod(eigen!, …)` first and throw the old error with "load GenericLinearAlgebra.jl for BigFloat" | `lp094.jl` "finding 2" (the branch follows the guard's own condition) |
| G3 | a plate of linear constraints whose replicas differ in element type rounds the offset to replica 1's type: `c = 2^24 + 1`, replica 2's η off by 0.278 | yes, already on `dev-danielb` | each replica's offset in its own covariance's type | `lp094.jl` "finding 3" |
| G4 | `GaussianPrior`: (a) an integer stored mean widens a Float32 frozen belief to Float64; (b) a dual stored mean throws in `energy` and `local_free_energy` | yes, both on `dev-danielb` | rule: a stored number enters the working type only when it carries a derivative, plain configuration is converted (Q4); `_derivative_carrier` in `_prior_relation`, `energy`, `local_free_energy`. Part 2's first fix added the mean itself, which widened a Float32 prior with a Float64 mean on Float32 data (the session's probe; `dev-danielb` gave Float32); corrected. The fixed-covariance message keeps the stored types (LP-091), unchanged | `lp094.jl` "finding 4" (integer, plain Float64, dual against a central difference) |
| G5 | an integer `logw` widens the switching and regime factors to Float64 | yes, on `dev-danielb` | `_logweights(ps, inputs...)` in the call's working type, every call site passing what carries it (messages, `Q`, the regime covariances) | `lp094.jl` "finding 5" |
| G6 | integer parameters with a declared `output` widen (the Bool output coefficient took `float(Int)`) | yes, on `dev-danielb` | `_param_eltype(f, ps) = _compute_eltype(float(eltype(f.Q)), ps)`: the noise's type only as the fallback when the parameters hold no float, so P34 holds | `lp094.jl` "finding 6" (linear and switching) |
| G7 | integer `b`, `logs` widen `ScaleShiftFactor` | yes, on `dev-danielb` | `_scale_shift_map` takes the inputs' type too | `lp094.jl` "finding 7" |
| G8 | `ToricFactor`'s `factor_belief` and `belief_free_energy` take the type from an integer `η` alone | yes, on `dev-danielb` | both pass the messages (the frozen `logq`) to `_logits`, as its message and free energy do | `AlgebraicFactors/test/lp094.jl` |
| G9 | `LikelihoodFactor` converts the stored `A`, `b` to the input's type: a dual `A` throws; a Float64 `A` with a Float32 input gives Float32 | the dual case yes (on `dev-danielb`); the Float32 case is the intended conversion | `_predictor_eltype` widens only for a derivative-carrying `A`/`b`, in the four places that convert | `lp094.jl` "finding 9" |
| G10 | the matrix checks only that a dual's partials are finite, so a path that drops a derivative passes | yes | `check_eltype_matrix` compares each partial of the `:Dual` row with a central difference of the fixture built at (1 ± 10⁻⁶/8) times its stored floats (`Scaled{s}`, the direction the dual seeds). Five operations exempt with reasons (the stopped `ScaleShiftBelief` energy, the regime plate's per-replica layout). A mutation that stops μ's gradient in `_prior_relation` fails exactly the four prior belief operations | the matrix itself (`unchecked_partials` in `lib/Lenticulum.jl/test/eltypes.jl`) |

## Checks

Float64 results bitwise unchanged: Codex's recorders against `9a213e36` (99/99 and 462/462 operations, on
part 1 with the session's fix), Mistral's dump of the probe cases and every fixture operation of both
packages (5,221 values, part 2 against part 1, after the session's last change). Matrices: Lenticulum
2,518/2,518, AlgebraicFactors 255/255. Suites: the suites of LenticulumCore, Mycelium, MyceliumLearning, Lenticulum (both tiers, with LP-094's 33 tests), AlgebraicFactors, ImplicitLayers, VariationalDiffusion and AdversarialFactors passed on the committed tree (`~/.cache/lenticulum/q19/`).

Not checked: the fixed-covariance `GaussianPrior` message with a Float64 mean and Float32 covariance still
returns mixed types (by LP-091's rule; whether that rule is right is not examined); G9's Float32 rule for a
plain stored `A` is kept, not revisited; the device paths of the new operations (Q20).
