#implementation

> Statistical-game combinators and parameter containers; gradient annotations are guarded pending L-SPEC-002.

> Sources: code: `statistical_game.jl`
>
> Theory (CT-ML wiki): [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Para Construction](https://mathstruct.org/CategoryTheory-ML-Wiki/Para-Construction) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

Implements: [[Factors are Parameterized Statistical Games]] (Def. 20), [Composition of Statistical Games](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#composition-energies-add-entropies-chain-definition-22) (Defs. 22,
25), [[Factors are Parameterized Statistical Games]] (Defs. 27, 28) — i.e. the factor interface itself.

`free_energy` and `entropy` are also extension points for concrete assembled lenses.
`scalar_free_energy` accepts either a factor or a lens when its family supplies the game
and scalarisation; there is no generic lens game. In ImplicitLayers, the factor game
scores incoming Dirac points, and the lens game scores its inversion's point plus a prior
anchor. That lens level is Definition 20's expectation under the inversion; see [[point_game]].
An arbitrary point is scored with `energy`/`scalar_energy` (L-ADR-0003), not a keyword
on `free_energy`.

## The composition laws, side by side

|  | AutoBayes Def. 22 | here |
|---|---|---|
| energy | $l^{dc} = l^c + l^d$ | `compose_energy` → a `GradedEnergy` pair |
| entropy | $H^{dc} = \mathbb{E}_{(y,b)\sim d'_{c_*\pi}(z)}[H^c] + H^d(c_*\pi,z)$ | `compose_entropy`, same shape, into a summand |
| loss | Thm 23: $F^{dc} = \mathbb{E}[F^c] + F^d$ | `compose_free_energy`, same |
| defect | — | `chain_rule_defect` = the Jensen gap |

The paper's `+` for energies has become a direct sum. Everything else is unchanged in shape.
Applying a `GradedScalarisation` recovers the paper exactly when $\sigma$ is linear, and
differs by `chain_rule_defect` when it is convex. See [[Scalar and Multivariate Energy]].

## Two adjectives that are the whole content of Definition 22

`compose_entropy(Hc_samples, Hd)` takes samples of $\mathbf{H}^c(\pi, y)$ at $(y,b)$ drawn
from the **downstream** inversion, at the **pushforward** prior. Get either adjective wrong
and you have a plausible-looking number that is not the free energy. The docstring says so;
there is no way to check it from inside this function, because it receives the samples
already drawn. **The correctness burden sits in the caller** — i.e. in the graph walker that
does not exist yet. Flagging it now so it is not discovered later.

## `ComposedFactor` is a `ContainerFactor`

```julia
struct ComposedFactor{A,B,C} <: AbstractLenticulumContainerFactor{(:first, :second)}
```

so `initialparameters` produces `(; first = ..., second = ...)` — the product
$\Phi \times \Theta$ of Definition 28, realised as literally the same `NamedTuple`
structure Lux uses for `Chain`. That is not a coincidence worth hiding; it is the clearest
evidence that both libraries are doing `Para`.

## Inheriting the LuxCore interface

The `for op in (:initialparameters, :initialstates)` loop extends `LuxCore.$op` for the
three factor abstract types, mirroring LuxCore's own definitions. Deliberately **not**
subtyping `AbstractLuxLayer` — see [[abstract_types]] §"Why factors are not a subtype".

## Implementation difficulties

### 1. Generic interfaces do not implement composite evaluation

The generic game/lens interfaces are extension points for concrete factor families.
Core supplies domain-checking fallbacks; Lenticulum implements a Gaussian pair as a
collapsed endpoint game. Mycelium computes graph Bethe free energy and explicit
coupling replay. Generic non-Gaussian composition still needs a pushforward law and
inversion rule; the parameter/state container alone cannot supply them.

### 2. `compose_entropy` and `compose_free_energy` take a plain collection of samples

They average with `reduce(+, samples) * (1/length(samples))`, which assumes an unweighted
Monte Carlo estimate. A weighted particle set, a quadrature rule, or an analytic expectation
in the conjugate case all need something else. The signature should eventually take a belief
plus a functional, not a vector of evaluations — but that requires the belief interface
that [[open_model]] §4 says is still an open question. Interim: the current signature is
adequate for sampling-based inversions and wrong for the others.

### 3. `TensorFactor` has no energy/entropy law implemented

Definition 25 says both halves add across summands. `TensorFactor` currently only derives
`energyspace` and `scalarisation`; there is no `compose_tensor_energy`. Straightforward, but
absent — and it inherits the mean-field laxness of [[lens]] §3, so it should not be added
without deciding how to report that.

### 4. `chain_rule_defect` measures one edge, not a path

`chain_rule_defect(σ, Fc_samples)` gives the Jensen gap across a single composition. For a
chain of $n$ factors the total discrepancy is a sum of per-edge gaps, each taken under a
different downstream posterior — it does **not** telescope into one variance. Any "total
laxness" diagnostic must accumulate per edge during the walk, not be computed once at the
end. Currently there is no walker, so this is a note for whoever writes it.

### 5. `setup` is defined here rather than re-exported

`LuxCore.setup` would work verbatim, but defining `LenticulumCore.setup` for
`AbstractLenticulumFactor` keeps the factor interface self-contained and avoids a user
needing `using LuxCore` to set up a factor. The cost is a name that shadows nothing but
duplicates something. Reconsider if `Lenticulum.jl` ends up re-exporting LuxCore anyway.

### 6. Historical gradient guards (2026-10-26; superseded below)

`ComposedFactor` retains its `coupling` field for the planned design, but only accepts
`DiagonalCoupling()`. Keyword, positional and explicitly parameterized constructors throw
`ArgumentError` for other values. The inner constructor closes the direct-construction
bypass; the inferred outer constructor also checks before selecting a type parameter, so an
invalid value such as `nothing` produces the same error. Checks happen at construction only.
The `LP-011 composition coupling guard` testset in `../test/runtests.jl` covers these paths,
all three non-default built-in coupling families, baselines, a custom subtype and defaults.
This implements option 2 of [[LP-011 Gradient couplings and isfrozen are declared but unused]];
option 1, gradient semantics, remains with [[L-SPEC-002 Stop Gradient]].

`isfrozen` still defaults to false. Mycelium rejects true in both `build` and `validate`,
including for non-learnable factors: `islearnable == false` only excludes a node from an
optimiser list and does not guarantee frozen input-cotangent semantics. No backward behavior
is added here, and the default coupling annotation does not insert AD stops.

Related: [[Factors are Parameterized Statistical Games]], [Composition of Statistical Games](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#composition-energies-add-entropies-chain-definition-22), [Composition of Gradients](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients), [[energy]], [[LenticulumCore]]

## Container conformance scope (2026-10-26)

`ComposedFactor` and `TensorFactor` still provide parameter/state trees and energy spaces,
not wired statistical games: neither implements `channels`, assembly, pointwise energy,
game free energy or entropy. Their default polarity enumeration is empty. The common
factor checker is therefore inapplicable; treating them as sinks would conceal missing
wiring. `test/runtests.jl`, "Container conformance scope (LP-008/009)", checks these absences.
Composition needs a pushforward prior and downstream inversion law; tensor needs a joint
belief/input convention before its component laws can be evaluated. No arbitrary
expectation or mean-field approximation is added here. The original LP-011 guards were later superseded by the implementation below.

## Bounds and exact container games (2026-10-27, LP-009)

This section supersedes the earlier statements that all container game laws are absent.
`FreeEnergyUpperBound(value)` carries an analytically evaluated scalar variational bound;
`FreeEnergyBoundEstimate(value,standard_error,nsamples)` carries a sampled estimate of a
bound's expectation. Neither is a number or accepted by the ordinary scalarisation API.
They leave `free_energy` and `entropy` as exact-game interfaces in the new supported domains.
The producer must define the normalised model, variational law and reference measure;
[[bound]] supplies the finite VP-chain instance. Estimates can fall below negative log
evidence, so a wrapper identifying only the inequality direction would be insufficient.

TensorFactor now implements exact independent **product games**: `π` and `y` are
NamedTuples keyed by `parts`, each describing that component's channels. Results retain the
branch grading and component state updates. The exact-product trait defaults to false;
Lenticulum enables it for its linear-Gaussian factors and supported pair composition.
Correlated joint beliefs and undeclared components throw. Tensor graph wiring and
inversions remain absent, hence no wired common-checker claim is made.

Lenticulum implements ComposedFactor for a pair of GaussianFactors by integrating out the
shared channel and delegating to the exact collapsed Gaussian conditional. Its energy space
is the collapsed Gaussian grading, not the pathwise `(first,second)` grading of the generic
container. [[composition]] records the distinction, restrictions and closed-form/common
conformance tests in `test/game_composition.jl`. Generic composition still throws with an
exact-pushforward reason. The value-level `compose_*` sample combinators are unchanged.

## LP-011 implementation (2026-10-27)

`isfrozen=true` pins all parameters while preserving input derivatives.
`islearnable=false` identifies an intrinsically nontrainable factor; graph-local subset
freezing uses `factor!(...; freeze)`. Composite gradient evaluation dispatches on coupling
and throws when the family has no implemented rule. `straight_through` has optional
ForwardDiff and ChainRules extensions. The old option-2 section above is historical.

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].
