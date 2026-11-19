#open-problem

> **Status: resolved (2026-11-19).** The review of P12 (`density_lambda` as the default λ of an
> `ImplicitProx` with Gaussian or mixture messages) found one defect, a type instability in resolving
> the default; fixed, with a regression test.
> Related: [[Feature Backlog]] (P12), [[LP-024 Findings of the review of 2026-11-09]] (finding 14),
> [[lib/VariationalDiffusion.jl/src/implicit_factor|implicit_factor]] §5.

> Sources: implemented by the maintainer's session (Claude); reviewed by Codex (GPT, `codex exec -s
> read-only` in the main checkout, one run). Brief, diff and answer: `../Lenticulum.jl-Review-2026-11-19/p12/`;
> log: `~/.cache/lenticulum/p12/`.
> code: `lib/VariationalDiffusion.jl/src/implicit.jl` (`_resolve_lambda`), `factor.jl`, `implicit_factor.jl`;
> test: `lib/VariationalDiffusion.jl/test/p12_default_lambda.jl`.

**Problem.** The requirement given to the reviewer: `density_lambda(schedule, nodes)` becomes the
default λ of `:gaussian` and `:mixture` messages; point messages keep λ = 1; an explicit λ is kept;
Float32 nodes keep a Float32 λ.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | `_resolve_lambda` read the field `prox.λ` twice, so inference kept `Union{Nothing,T}` as its return type | yes, reproduced: `@inferred` failed for every message, default and explicit λ, Float64 and Float32 nodes | the field is read once into a local, which the `=== nothing` test narrows | the `@inferred` loop in `p12_default_lambda.jl` (failed before the fix, passes after) |

The reviewer also checked, without a finding: the 24 checks of the first test version, `LuxCore.setup`,
and building the factor inside ForwardDiff and Zygote differentiation. Not checked by it: the full
suite, differentiation through inference.
