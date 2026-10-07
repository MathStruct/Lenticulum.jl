#definition #overview

> A **belief** is what travels along an edge of a factor graph and what sits on a variable:
> a representation of a probability distribution, or of a likelihood, over that variable's
> space. Four concrete types exist, each with its own note: [[Trivial Belief]], [[Dirac Belief]],
> [[Gaussian Belief]] and [[Sample Belief]]. Their algebra is one operation, `combine`, the product
> of densities.

> Sources: original to this vault (design); code: `open_model.jl` (`AbstractBelief`, `DiracBelief`, `SampleBelief`, `TrivialBelief`), `beliefs.jl` (`GaussianBelief`), `messages.jl` (`combine`, `belief_distance`, `belief_logdensity`)
>
> Theory (CT-ML wiki): [Giry Monad](https://mathstruct.org/CategoryTheory-ML-Wiki/Giry-Monad) · [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Distribution Monad](https://mathstruct.org/CategoryTheory-ML-Wiki/Distribution-Monad)

## What a belief is

Categorically, a belief on a variable $X$ is an element of $\mathcal P X$, a state $1 \to X$ in a
[Markov category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) (for continuous spaces, the [Giry monad](https://mathstruct.org/CategoryTheory-ML-Wiki/Giry-Monad)).
It is the prior $\pi$ of a [[Inversions and Bayesian Lenses|Bayesian lens]] and the posterior that
inference returns. On an edge it may also be a **likelihood**: a non-negative function that
need not integrate to one, which is why some beliefs are allowed to be *improper*.

All belief types subtype `LenticulumCore.AbstractBelief` and are defined in `LenticulumCore`,
`GaussianBelief` included (it moved there from the top-level package in October 2026, so that
every factor package can emit one); their message rules are in `Mycelium`. The diffusion
factors now return Laplace Gaussians on request; the equilibrium factors still return points
([[DEQ as a Relation]] §5).

## The six types

| type | represents | where | `isexact` | typical source |
|---|---|---|---|---|
| [[Trivial Belief]] | no information (the unit) | `LenticulumCore` | true | an unset message, the prior of a prior |
| [[Dirac Belief]] | a point mass, $\delta_x$ | `LenticulumCore` | true | an observation (a hard clamp), a point inference |
| [[Gaussian Belief]] | $\mathcal N$ in canonical form $(\eta, \Lambda)$, possibly improper | `Lenticulum` | false (the default) | Gaussian factors, Laplace approximations |
| [[Sample Belief]] | a weighted particle set | `LenticulumCore` | false | samplers, the fallback |
| [[Categorical Belief]] | a distribution over finitely many states | `LenticulumCore` | false | discrete variables, switches, Bernoulli events |
| [[Mixture Belief]] | a weighted sum of beliefs of one type | `LenticulumCore` | false | several answers at once (branches), Gaussian-sum filters |

`isexact` says whether a computed belief is exact rather than approximate. It defaults to
`false` so that silence never implies exactness; only the Trivial and Dirac beliefs override it.
A Gaussian *can* be exact (on a linear-Gaussian tree it is), but the flag does not claim it.

## The algebra: `combine`

Pooling two beliefs about one variable is the pointwise product of their densities, the
**sum–product** step at a variable node ([[Messages are Inversions]]). `Mycelium.combine`
implements exactly the cases it can do honestly:

| combine | result | why |
|---|---|---|
| Trivial with anything | the other | the unit |
| Gaussian with Gaussian | add $(\eta, \Lambda)$ | exact, associative, commutative |
| Dirac with Gaussian, Sample or Trivial | the Dirac | a hard clamp is the $\Lambda \to \infty$ limit and dominates |
| two equal Diracs | that Dirac | idempotent |
| two different Diracs | **error** | contradictory hard clamps are a modelling error |
| Categorical with Categorical | elementwise product, renormalised | exact and total; **error** only for disjoint supports |
| Sample with anything that has a density | the samples, **reweighted** by that density | importance reweighting |
| Mixture with anything | each component pooled, weights scaled by its overlap | exact for Gaussian and categorical components |
| two Sample beliefs | **error** | neither has a density; project one first (`moment_match`) |
| anything else | **error** | no rule for the pair |

So beliefs form a partial commutative monoid under `combine`, with the Trivial belief as unit
and the Dirac beliefs as absorbing elements. Gaussians and categoricals are closed under it,
which is why Gaussian belief propagation is exact on trees ([[The Linear Gaussian Chain]]);
mixtures are closed too, at the price of growing (`reduce_mixture` bounds them). The rest of
the catalogue (addition, logic, projection, tempering) is in [[Belief Algebra]].

## Other operations

- `belief_distance(a, b)` — how much a message changed, the convergence test of
  `propagate!`. It returns `Inf` whenever it cannot tell, so that a schedule never declares
  convergence it cannot verify.
- `belief_logdensity(b, x)` — $\log p_b(x)$; for Gaussians, categoricals (``x`` a state or label) and mixtures.
- `moment_match(b)` — the Gaussian with the same mean and covariance (samples, Gaussian mixtures); the projection step of expectation propagation.
- `reduce_mixture(m; max_components)` — merge Gaussian mixture components, preserving mean and covariance.
- `variable_entropy(b)` — the entropy in the [[Bethe Free Energy]]; for Gaussians it can be negative.

Everything else one might do with beliefs (addition, mixture, logic, projection, tempering)
is catalogued, with what each needs, in [[Belief Algebra]].

Related: [[Messages are Inversions]], [[Factor Graphs]], [[Channels and Polarity]],
[[Inversions and Bayesian Lenses]], [[Probabilistic Types]]

````tabs
tab: Julia
**Docs:** [Lenticulum API](https://mathstruct.org/Lenticulum.jl/dev/packages/lenticulum/) · [Mycelium API](https://mathstruct.org/Lenticulum.jl/dev/packages/mycelium/)
```julia
using Lenticulum, LenticulumCore, Mycelium
using LenticulumCore: DiracBelief, TrivialBelief, SampleBelief
a = Gaussian(1.0, 4.0); b = Gaussian(3.0, 4.0)       # N(1, 4) and N(3, 4), stored as (η, Λ)
c = combine(a, b)                                     # pooling = adding canonical parameters
(belief_mean(c), belief_cov(c))                       # ≈ ([2.0], [2.0;;]): precision-weighted mean, halved variance
combine(TrivialBelief(), a) === a                     # true: the unit
u = uninformative(1); combine(u, a).η == a.η          # true: (0, 0) is the same unit in canonical form
combine(DiracBelief([0.5]), a)                        # the Dirac: a hard clamp dominates (Λ → ∞)
lik = GaussianBelief([2.0, 0.0], [1.0 0.0; 0.0 0.0])  # constrains x₁ only: an improper likelihood message
isproper(lik)                                         # false — fine for a message, not for a mean
belief_mean(combine(lik, Gaussian([0.0, 0.0], [1.0 0.0; 0.0 1.0])))   # ≈ [1.0, 0.0]: proper after a prior
```
````
