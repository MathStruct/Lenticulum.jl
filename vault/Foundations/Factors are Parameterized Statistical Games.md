#definition #design #implementation

**A Lenticulum factor is a parameterized statistical game** in the sense of AutoBayes (Definition 27): a [Bayesian lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) — a forward kernel plus an approximate inversion — decorated with an **energy** and an **entropy**, all of which may depend on parameters `ps`. This is the one-line difference from Lux.jl:

$$
\underbrace{\mathbf{Para}(\mathbf{Lens}(\mathcal C))}_{\text{a Lux layer}}
\qquad\text{vs.}\qquad
\underbrace{\mathbf{Para}(\mathbf{StatGame})}_{\text{a Lenticulum factor}}
$$

and the lens is not stored but **assembled on demand**, once a [[Channels and Polarity|polarity]] says which channels are observed. Lenticulum's own departure from the paper is that the energy is **vector-valued** ([[Scalar and Multivariate Energy]]).

> Sources: St Clere Smithe & Perin, *AutoBayes* (arXiv:2503.18608) Definitions 20, 22, 27–29, Theorem 23, Remarks 24, 26, 30 and the closing discussion; code: `lib/LenticulumCore.jl/src/statistical_game.jl`, `lens.jl`, `energy.jl`, `abstract_types.jl` ([[statistical_game]], [[energy]], [[lens]]).
>
> Bibliography: [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Para Construction](https://mathstruct.org/CategoryTheory-ML-Wiki/Para-Construction) · [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [paper note](https://mathstruct.org/CategoryTheory-ML-Wiki/Papers/AutoBayes---A-Compositional-Framework-for-Generalized-Variational-Inference)

## The four pieces, as four independent choices

| piece | in the paper | in Lenticulum | what varies |
|---|---|---|---|
| forward kernel $c$ | $X \rightsquigarrow [\![c]\!] \times Y$ | the model half of `assemble(f, polarity, ps, st)` | architecture |
| inversion $c'$ | $\mathcal P X \times Y \rightsquigarrow X \times [\![c]\!]$ | an `AbstractInversion`: exact, amortised, solver, proximal, trivial | how the posterior is approximated ([[Inversions and Bayesian Lenses]]) |
| energy $l^c$ | $X \times [\![c]\!] \times Y \to [0, \infty]$ | `energy(f, …)`, a **vector** in `energyspace(f)` | NLL, residual, robust loss |
| entropy $H^c$ | $\mathcal P X \times Y \to [0, \infty]$ | `entropy(f, π, y, …)`, in the same energy space | Shannon, a KL, $\beta$-weighted, zero |

and the loss is `free_energy(f, π, y, ps, st)` — a vector — collapsed by `scalar_free_energy` through the factor's `scalarisation`. The type difference matters: the energy eats *points* (evaluated inside an expectation, cheap, no normalisation), the entropy eats a *distribution*. That is why they compose differently.

## Composition: energies add, entropies chain

Definition 22 composes $c : X \multimap Y$ and $d : Y \multimap Z$ by adding energies and chaining entropies, and Theorem 23 gives the free-energy chain rule $F^{dc}(\pi, z) = \mathbb E_{(y,b)\sim d'_{c_*\pi}(z)}[F^c(\pi, y)] + F^d(c_*\pi, z)$. In Lenticulum:

- `compose(c, d; coupling)` builds a `ComposedFactor`; `compose_energy` returns a `GradedEnergy` (the direct sum that replaces the paper's $+$), `compose_entropy` and `compose_free_energy` implement the chain rule from samples of the downstream inversion, and `chain_rule_defect` reports the **Jensen gap** between the vector and scalar chain rules.
- `TensorFactor` is parallel composition — lax, with the defect the mutual information of the branches (Remark 26). Lenticulum's rule is **report the laxness, do not hide it**.
- On a graph, "downstream" is a property of the message schedule, not of the wiring, so Mycelium reports the order-free **Bethe** form instead ([[Bethe Free Energy]]).

The recursion is a fold with two sweeps — **priors forward (pushforward), samples backward (inversions)** — the same shape as forward/backward autodiff, except that the backward sweep is stochastic and the forward one carries distributions. The paper names the two expensive steps: pushforward priors (marginalisation — Mycelium's job, by message passing) and expectations under the inversions (sampling or conjugacy).

## Priors, data and losses are factors (Remark 24)

A pure game's loss lacks the prior term of the free energy; AutoBayes fixes this by making the prior its **own game** with trivial inversion, energy $-\log p_\pi$ and zero entropy. Lenticulum takes this literally: a prior is a `PriorFactor`, data is a `DataFactor` (a cup — a hard clamp), a loss is a `LossFactor` (a sink), and an optimiser is an `OptimiserFactor` on an exposed parameter variable. See [[Everything is a Factor]].

## Parameters, and the gradient the paper asks for

Any of the four pieces may depend on $\theta$ (Definition 27): a decoder, an amortised encoder, a learned loss or critic, a learned regulariser. The parameter tree is Lux's — `initialparameters` returns a nested `NamedTuple`, which is the $\mathbf{Para}$ composite $\Phi \times \Theta$ of Definition 28 with labels. The paper's default semantics is descent **with respect to the Fisher metric** (the Bayesian learning rule); the multivariate energy is what makes a Gauss–Newton approximation of that metric available ([[Scalar and Multivariate Energy]] §6).

Definition 29 composes gradients block-diagonally and is therefore **lax**: it drops the terms where an upstream parameter moves the downstream pushforward prior and where a downstream parameter moves the sampling distribution of the upstream inversion — the reparametrisation/score-function terms. Lenticulum makes the choice explicit per edge, as an `AbstractGradientCoupling`:

| coupling | keeps | cost |
|---|---|---|
| `DiagonalCoupling()` | block diagonal only (stop-gradient) | cheapest, biased |
| `PathwiseCoupling()` | differentiates through the sampler | needs a reparametrisable inversion |
| `ScoreFunctionCoupling(baseline)` | REINFORCE estimate of the dropped term | unbiased, high variance |
| `ExactCoupling()` | everything | conjugate / analytic factors only |

These are the paper's "different semantics functors" — different lax sections of the gradient fibration (Remark 30) — and [[The Type Discipline of a Factor Graph]] §4 observes that together they form an effect system with no composition rule yet.

## The engineering agenda the paper leaves

1. **Pushforward by message passing** — belief propagation / variational message passing approximates $c_*\pi$ ([[Factor Graphs]], [[Schedules]]).
2. **Expectations under inversions** — sampling, or conjugacy where available (the linear-Gaussian fragment: [[The Linear Gaussian Chain]]).
3. **Conjugacy is not preserved by pushforward** — moment-matching projections back into a family are needed, and they are again lax.

````tabs
tab: Julia
**Docs:** [LenticulumCore API](https://mathstruct.org/Lenticulum.jl/dev/packages/lenticulumcore/) · [Lenticulum API](https://mathstruct.org/Lenticulum.jl/dev/packages/lenticulum/)
```julia
using Lenticulum, LenticulumCore, Mycelium, Random
f = GaussianFactor(1 => 1; noise = 0.25, channels = (:x, :y))   # y = A x + b + ε
ps, st = LenticulumCore.setup(Random.Xoshiro(0), f)              # the Para parameter tree
keys(ps)                                                         # (:A, :b)
supported_polarities(f)                                          # x → y and y → x
energyspace(f)                  # graded: (:fit, :complexity, :negentropy) — a vector energy
islinear(scalarisation(f))      # true: scalarising commutes with composition here
```
````
