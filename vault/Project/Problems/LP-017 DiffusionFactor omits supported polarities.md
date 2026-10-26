#open-problem

> **Status: open (2026-10-26).** Related: [[LP-008 No conformance test for the factor interface]], [[L-SPEC-001 The Factor Interface]], [[factor]].

> Sources: original repository analysis; code: `lib/VariationalDiffusion.jl/src/factor.jl`, `lib/VariationalDiffusion.jl/test/conformance.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

**Problem.** `supports_polarity` accepts every role assignment with an unobserved channel;
`supported_polarities` lists only one-target/all-other-observed assignments. At arity two,
(U,L), (L,U) and (U,U) are accepted but omitted. This was already described in the
implementation note; the common checker now makes the mismatch executable. Moreover
`invert` returns only the first unobserved block at (U,U), without a joint posterior.

**Options.** Enumerate all supported role assignments and define a joint inversion API;
or narrow the advertised support to the splits whose full contract is implemented, with
an explicit separate API for target-only messages. Either choice affects graph polarity
resolution and latent/joint posterior conventions, so this is not changed silently.

**Resolution so far.** The checker uses explicit `polarity_gap` reasons for the three
omitted binary assignments. The two latent/single-target Gaussian cases still check the
lens, independent site/posterior fixtures, pointwise energy and finite local energy. The
multi-target case explicitly opts out of message checks; a separate independent fixture
checks the returned first block. Three `@test_broken` assertions record missing enumeration
and one records the absent two-coordinate posterior. Game gaps remain in LP-009. The
ordinary enumerated binary and unary cases retain the default checker checks.
