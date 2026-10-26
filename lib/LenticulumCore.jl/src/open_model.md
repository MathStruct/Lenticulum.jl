#implementation

> Open-model results retain latent values; belief types include probability representations
> and improper callable likelihood potentials.

> Sources: code: `open_model.jl`
>
> Theory (CT-ML wiki): [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

Implements: [[Open Models and Latent Channels]] (AutoBayes Def. 1) and the belief types that flow along the
graph.

## The one idea

$$c : X \rightsquigarrow \llbracket c \rrbracket \times Y$$

The latent space $\llbracket c \rrbracket$ exists so that
[composition](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model#composition-without-integration-definition-4) can *file away* the intermediate value instead of
integrating it out:

$$(q \circ\!\!\!\bullet\; p)(ds,dy,dt,dz \mid x) = q(dt,dz\mid y)\,p(ds,dy\mid x)$$

No integral. `OpenModelResult(observed, latent)` is the runtime witness of this: a forward
pass returns both, and the `latent` field is exactly an autodiff tape entry. Same trade of
memory for tractability, under [the same chain rule](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion).

## Interface

| function | paper | note |
|---|---|---|
| `forward(m, x, ps, st)` | sample $c$ | returns `OpenModelResult` |
| `logdensity(m, x, a, y, ps, st)` | $\log p_c(a,y\mid x)$ | needed whenever the energy is a NLL |
| `pushforward(m, π, ps, st)` | $c_*\pi$ | **expensive** — see below |
| `latentspace` / `observedspace` / `unobservedspace` | $\llbracket c\rrbracket$ / $Y$ / $X$ | |
| `ispure(m)` | $\llbracket c\rrbracket \cong 1$ | **not preserved by composition** |

Beliefs: `DiracBelief` (a clamp / a cup), `SampleBelief` (particles), `TrivialBelief` (the
unit space).

## Implementation difficulties

### 1. `pushforward` is one of the two things that make this hard

The paper's own closing discussion says it: computing $c_*\pi$ is marginalisation, "similarly
expensive to computing exact inversions", and the recommended remedy is belief propagation
or variational message passing. Mycelium supplies the graph and message-passing machinery.

The consequence for this file is that `pushforward` must be allowed to lie. It returns an
`AbstractBelief` with no promise of exactness, and `isexact` (defaulting to `false`) is how
a caller finds out. An implementation that returns a moment-matched Gaussian is legitimate;
one that returns it while claiming `isexact() == true` is a bug.

### 2. Conjugate families are not preserved by pushforward

Also from the paper's closing discussion: conjugate models make everything cheap, but
pushing forward does not generally preserve the family, so you need moment-matching
projections back into it — and encoding "which family this wire carries" into the game data
"requires annotating the games with predicates, which can be done compositionally, but
laxly".

Nothing in this file does that yet. When it lands it will want a `family(belief)` trait and
a `project(belief, family)` operation, and the laxness of the annotation will need the same
treatment as everything else here: track it, report it, do not hide it.

### 3. `ispure` defaults to `false` and composition must not override it naively

Purity ($\llbracket c \rrbracket \cong 1$) is a genuine optimisation: a pure model needs no
latent bookkeeping. It is also **destroyed by composition** — composing two pure models
gives a model whose latent space is the intermediate space. Any future `ispure(::ComposedModel)`
must return `false` unconditionally, not `all(ispure, parts)`. Writing it the natural way
would be wrong, which is why it is called out here.

### 4. Belief representations are the real open question

`GaussianBelief` now supplies canonical parameters, categorical beliefs store normalised
log-probabilities, and samples provide a finite approximation. `AbstractPotentialBelief`
separates likelihoods known only through a log-density from normalised probability
representations. There is no shared exponential-family or natural-gradient interface for
all these types; adding one still needs a declared family and projection contract.

### Weight normalization

Categorical probabilities and mixture weights are normalized in log space after checking
that the input is finite, nonnegative and has positive mass. This avoids overflow when
several finite weights are near the floating-point maximum, preserves zero support, and
keeps log probabilities finite for positive weights across the floating-point range.
The construction tests cover Float32 and Float64 large and subnormal weights.

### 5. `logdensity` on an unnormalised measure

[[Open Models and Latent Channels]] notes that cyclic models require unnormalised measures, so
`logdensity` may be a log-*potential* rather than a log-density, differing by an unknown
constant. For energy purposes this is fine (the constant is irrelevant to gradients), but
for anything that compares densities across models it is not. Currently unmarked. A
`isnormalised(m)` trait will be needed before any model-comparison feature.

Related: [[Open Models and Latent Channels]], [Composition of Open Models](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model#composition-without-integration-definition-4), [[lens]], [[Inversions and Bayesian Lenses]]

## Likelihood potentials (LP-016)

`AbstractPotentialBelief <: AbstractBelief` represents an improper, unnormalised likelihood
known through `Mycelium.belief_logdensity(b, x)`, up to an additive constant independent of
$x$. `LogPotentialBelief(logφ)` wraps the callable `logφ(x)`. It is intended as a
factor→variable message, with no sampling, mean, covariance, or entropy contract.

The abstract type leaves room for likelihoods with gradients, a parametric form, or batched
evaluation. Those types provide their own log-density and can add exact product rules.
Mycelium sums log-potentials and reweights explicit particles; it refuses arbitrary
parametric products. Core construction tests check the representation; Mycelium and
Adversarial tests check pooling and target exclusion against closed forms.
