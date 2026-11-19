#open-problem

> **Status: resolved (2026-11-19).** The review of P34 (the fixed output coefficient of the constraint
> factors takes its number type from the parameters at use) found one defect, an integer coefficient
> rounded at use so that tied factors with equal signatures had different coefficients, and one
> missing test (ForwardDiff); both done, with regression tests.
> Related: [[Feature Backlog]] (P34), [[LP-061 Findings of the review of P32]],
> [[lib/Lenticulum.jl/src/constraint|constraint]] "The fixed output coefficient (P34)".

> Sources: implemented by the maintainer's session (Claude); reviewed by Codex (GPT, `codex exec -s
> read-only` in the main checkout, one run). Brief, diff, answer and the reproduction of finding 1:
> `../Lenticulum.jl-Review-2026-11-19/p34/`.
> code: `lib/Lenticulum.jl/src/constraint.jl` (`_output_eltype`, `_fixed_at`, `_param_eltype`, `_coeffs`),
> `regime.jl`, `switching.jl`, `ext/LenticulumChainRulesCoreExt.jl`;
> test: `lib/Lenticulum.jl/test/p34_output_coeff.jl`.

**Problem.** The requirement given to the reviewer: the fixed output coefficient (`output_coeff`, default
`I`) of `RegimeFactor`, `SwitchingConstraintFactor` and `LinearConstraintFactor` takes its number type
from the parameters at use, not from the constructor's `noise`; Float64 results bit for bit unchanged;
ForwardDiff and Zygote keep working; explicit floating coefficients keep their type. The reviewed
version stored every exact coefficient exact (`I` as Bool, `2I` and integer matrices as integers) and
converted it at use.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | Two tied factors with `output_coeff = fill(16777217, 1, 1)` and `fill(16777217.0, 1, 1)` have equal `m_step_signature`s, but with Float32 parameters the integer one becomes 16777216 at use while the floating one stays 16777217: their statistics are pooled under the first factor's coefficient | yes, reproduced (`finding1.jl`: signatures equal, `_coeffs(...).y` 1.6777216f7 against 1.6777217e7) | only Booleans are stored exact and converted at use (0 and 1 are exact in every type); other integers keep the rule from before P34 (`2I` in the noise's type, an integer matrix Float64) | "other integer and floating coefficients" in `p34_output_coeff.jl` (failed before the fix); also a Boolean and a floating identity: equal signatures, equal coefficients |
| 2 | No ForwardDiff check of the conversion of the Boolean identity to duals | yes, a coverage gap (no failure claimed) | added: the gradient of the local free energy in the coefficients and offsets, default `I` against an explicit Float64 identity, for the three factors | the ForwardDiff lines in `p34_output_coeff.jl` (equal bit for bit; pass) |

Not checked by the reviewer: running anything (read-only sandbox), device arrays, plates.
