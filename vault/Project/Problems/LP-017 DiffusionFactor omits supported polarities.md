#open-problem

> **Status: resolved (2026-10-26).** Complete capped enumeration, predicate-based structural
> checks and joint inversion on unobserved blocks. Related: [[LP-008 No conformance test for the factor interface]], [[L-SPEC-001 The Factor Interface]], [[factor]].

> Sources: original repository analysis; code: `lib/LenticulumCore.jl/src/channels.jl`, `lib/VariationalDiffusion.jl/src/factor.jl`, `lib/VariationalDiffusion.jl/src/implicit_factor.jl`, `lib/VariationalDiffusion.jl/test/conformance.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

**Problem.** The predicate accepted every assignment with an unobserved channel but the
list omitted latent and multiple-target splits. Inversion returned only the first
unobserved block. Three broken membership tests and one broken dimension test recorded it.

**Resolution.** Option A: `supported_polarities` now returns all $3^n-2^n$ default-precision
Observed/Unobserved/Latent assignments through arity four (65 entries). One-target/all-other-
observed entries retain their order at the front; the remainder follows lazy traversal.
Above four channels it throws `ArgumentError` pointing to `supports_polarity` and
`polarities`. The predicate remains the authority. The lazy iterator is type-unstable,
since each assignment has its own type, and complete traversal still costs exponential work.

Core `hasnopolarity` and `isunidirectional` retain list-based defaults; diffusion overrides
both without enumeration. Mycelium's `issink` delegates to `hasnopolarity`. Graph polarity resolution constructs each per-target assignment from available messages
and checks it with the predicate; it does not need a list.

`invert` and `likelihood_message` return a joint belief on concatenated unobserved blocks
in channel order. Point messages select the prox solution; Gaussian and mixture messages
restrict the existing full-state approximation. Likelihood messages divide out the
product of diagonal target anchors. An improper Gaussian is allowed; an improper mixture
component throws because its normalized weight is undefined. The diffusion prior remains
in the message. `factor_message(f, target, …)` remains per-target. Joint prior points use
the same concatenation; shared-block points retain their prior convention.

**Checks.** The four broken assertions now pass in "Diffusion factor partial conformance",
with no LP-017 `polarity_gap` or `message_gap`. For the isotropic fixture the field
$g(z)=cz$ integrates to $c\|z\|^2/2$: the site has precision $cI$, so multiplying by
$N(0,I)$ gives $N(0,I/(1+c))$. Both target coordinates and the covariance are checked.
"Diffusion polarity enumeration and cap (LP-017)" checks counts at arities two, three and
four, equality with lazy traversal, ordering and structural answers above the cap.
"Joint diffusion messages (LP-017)" checks noncontiguous unequal blocks, joint-prior
cotangents, point/Gaussian/mixture messages, anchor pooling and correlated covariance.
"Joint mixture rejects undefined anchor division (LP-017)" checks that removing the only
confining anchor throws instead of assigning normalized weights to flat components.
"Joint mixture anchor pooling (LP-017)" checks recovery of both components and weights
when a two-target mixture site is multiplied by its diagonal anchor. Core "Lazy polarity
enumeration (LP-017)" checks lazy traversal independently of a list.
Game gaps remain in LP-009; this does not claim arbitrary-prior or stochastic-backend
conformance.

## Validation run (2026-10-26)

Full suites, using `julia --project=lib/<Package>.jl -e 'using Pkg; Pkg.test()'`
(and `--project=.` for Lenticulum):

| Package | Pass | Fail | Error | Broken |
|---|---:|---:|---:|---:|
| LenticulumCore | 230 | 0 | 0 | 0 |
| Mycelium | 459 | 0 | 0 | 0 |
| Lenticulum | 775 | 0 | 0 | 0 |
| VariationalDiffusion | 1143 | 0 | 0 | 0 |
| ImplicitLayers | 316 | 0 | 0 | 0 |
| Adversarial | 232 | 0 | 0 | 3 |
| AlgebraicFactors | 14396 | 0 | 0 | 0 |

VariationalDiffusion's original testset passed 889 assertions in 6m50.1s; its conformance
file passed 254, including the joint and cap testsets above. The remaining three broken
assertions belong to Ratio particle conformance (LP-016).
