#open-problem

> **Status: resolved (2026-10-26), option A.** RatioFactor emits a likelihood potential;
> posterior reweighting remains explicit. Related: [[LP-006 Gaussian entropies are relative to Lebesgue measure]],
> [[LP-008 No conformance test for the factor interface]], [[LP-009 Game-level free energy is declared but not implemented]],
> [[L-SPEC-001 The Factor Interface]], [[ratio]].

> Sources: original repository analysis; code: `lib/LenticulumCore.jl/src/open_model.jl`, `lib/Mycelium.jl/src/messages.jl`, `lib/Adversarial.jl/src/ratio.jl`, `lib/Adversarial.jl/test/conformance.jl`, `lib/Mycelium.jl/test/runtests.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## Defect and closed form

The former outgoing particle message reweighted the cavity and was therefore a posterior.
Pooling it with the cavity failed (two SampleBeliefs have no product rule); a supplied target
input could override the cavity. The graph energy also used the incoming weights instead
of the factor-belief weights.

For $p=N(0,1)$ and $q=N(1,1)$,

$$
\log r(x)=\log(p(x)/q(x))=1/2-x.
$$

On equally weighted particles $\{0,1\}$, the posterior weights are
$\{e/(1+e),1/(1+e)\}$. The incoming energy expectation is zero, whereas the factor-belief
expectation of $-\log r$ is $1/(1+e)-1/2$.

## Resolution

- `AbstractPotentialBelief` is the extensible improper likelihood interface;
  `LogPotentialBelief(logφ)` wraps a callable. Log-density is defined up to an additive
  constant. There is no sampling, mean, or entropy contract. Richer subtypes can expose
  gradients, parametric structure, or batched evaluation.
- RatioFactor's graph and lens likelihood messages return this potential without reading
  the cavity or target inputs. `invert` and explicit `reweight` return the posterior.
  The callable captures parameters/state; density evaluations discard returned network
  states. Use inference-mode state and do not mutate captured values while messages are used.
- Mycelium reweights particles through `belief_logdensity`, adds log-potentials, preserves
  Dirac dominance and Trivial unit laws, and refuses arbitrary potential products with
  Gaussians, categoricals, or mixtures. It does not resample. Callable identity gives zero
  distance; other evaluators give `Inf`. Damping potentials is an inspectable no-op.
- Mycelium's free-energy call passes incoming variable→factor messages $\mu$.
  The unary factor belief is $b_f \propto r\mu$. RatioFactor now averages $-\log r$ under
  that reweighted belief. This is the Bethe energy contribution only: no continuous particle
  entropy is supplied (LP-006), so it is not a complete continuous free energy. Dirac game
  quantities remain; particle game methods still throw for lack of a chosen reference measure.

## Verification

`lib/Adversarial.jl/test/conformance.jl` replaces the three broken LP-016 checks with
passing closed-form assertions for pooling once, target exclusion, and posterior energy
expectation. It checks the lens likelihood separately from inversion and tests a particle
prior with two RatioFactors on one variable in both insertion orders. The marginal weights
are proportional to the prior times $r_1r_2$ once.

`lib/Mycelium.jl/test/runtests.jl` checks the potential products, unit and Dirac laws,
unsupported parametric products, finite-support associativity, distance, damping, and method
ambiguities. Ambiguity detection exposed the pre-existing `(nothing,nothing)` distance
ambiguity; an explicit `Inf` method now resolves it.

Full package suites run with `julia --project=<package> -e 'using Pkg; Pkg.test()'`:

| package | pass | fail | error | broken |
|---|---:|---:|---:|---:|
| LenticulumCore | 236 | 0 | 0 | 0 |
| Mycelium | 494 | 0 | 0 | 0 |
| Lenticulum (top level) | 775 | 0 | 0 | 0 |
| Adversarial | 254 | 0 | 0 | 0 |
| ImplicitLayers | 316 | 0 | 0 | 0 |
| AlgebraicFactors | 14396 | 0 | 0 | 0 |

These counts include the LP-017 changes already present in the working tree. The
Mycelium suite includes `Test.detect_ambiguities(Mycelium, LenticulumCore; recursive=true)`;
no Aqua suite is configured in these package tests.
