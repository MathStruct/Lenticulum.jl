#definition #design #implementation

Every Lenticulum factor carries, next to its forward kernel, an **inversion** — the backward half of a [Bayesian lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens). The inversion's extra input is a **prior**, exactly as a lens's backward pass takes the cached forward input: *the prior plays the role of the linearisation point* ([Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)). Because Bayesian inversion satisfies a chain rule, one may attach an *approximate* inversion to each factor locally and still obtain a correctly structured posterior for the whole graph — the inference analogue of "define an `rrule` per primitive".

> Sources: *AutoBayes* (arXiv:2503.18608) §3, Definitions 9–16, Theorem 13, Remark 11, footnotes 3–4; St Clere Smithe, *Bayesian Updates Compose Optically* (arXiv:2006.01631) Theorem 5.2; code: `lib/LenticulumCore.jl/src/lens.jl` ([[lens]]), `src/gaussian.jl`, `lib/Mycelium.jl/src/passing.jl` ([[passing]]).
>
> Bibliography: [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]] · [[Bibliography#^stclere2020optics|St Clere Smithe 2020]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Almost-Sure Equality](https://mathstruct.org/CategoryTheory-ML-Wiki/Almost-Sure-Equality) · [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## The inversion is a free choice — and a type

`BayesianLens(model, inversion)` pairs a forward model with any `AbstractInversion`; nothing requires the inversion to be exact. The families are the paper's list (§3, footnote 4) made into types:

| type | realises $c'_\pi$ as | family |
|---|---|---|
| `ExactInversion()` | Bayes' law on the model's own kernel, $c^\dagger_\pi$ | conjugate / linear-Gaussian |
| `AmortisedInversion(net)` | a Lux network with its own parameters (an encoder) | VAEs, amortised VI |
| `SolverInversion(solver)` | root-finding on a residual | algebraic and equilibrium factors ([[Implicit Learners]]) |
| `ProximalInversion(prox)` | a proximal step on an energy | diffusion priors (RED-Diff, ProxDM) |
| `TrivialInversion()` | nothing to infer | priors (Remark 24) |

The paper's notation, worth keeping because ML notation conflates the two: $c(dy \mid x)$ is the likelihood, $c^\dagger_\pi(dx \mid y)$ the **exact** posterior, $c'_\pi(dx \mid y)$ the **approximate** posterior the code computes (Remark 11). `isexact(inversion)` distinguishes them.

> [!note] Two independently parametrised halves
> A factor has *two* Lux-style sub-models — forward kernel and inversion — each with its own `ps`/`st`. This is the structural reason a factor cannot be a Lux layer: a layer has one direction, not two independently parametrised ones.

## The backward pass returns the latent space too

`invert(lens, π, y, ps, st)` returns a belief over $X \times [\![c]\!]$, not just $X$: it reconstructs the latent scratch space as well, because that is what the next factor upstream consumes (Definition 9). Dropping it breaks the chain rule. See [[Open Models and Latent Channels]].

## Posterior versus message

`invert` returns the **posterior** $c^\dagger_\pi(y)$, prior included. A belief-propagation message is the **likelihood**, with the prior divided out; if a factor sent the posterior, a variable of degree $d$ would count its prior $d$ times. For Gaussians in canonical form the two differ by one addition — `combine(π, message) == posterior`, asserted in the test suite — and on a chain the difference is invisible, which is why the paper never meets it. Mycelium's `factor_message` returns the likelihood; see [[Messages are Inversions]].

## Granularity is a graph annotation

Nothing fixes the granularity of the inversions (footnote 4): one monolithic amortised encoder for a whole composite, one inversion per factor (structured VI), or anything in between are the same formalism with the $(-)'$ annotations placed differently. In Lenticulum that choice is where factors and composite factors sit in the graph — a graph annotation, not a rewrite.

## Laxness is surfaced, not hidden

- **Parallel composition is lossy** (Remark 16): a `TensorLens` feeds each branch only the *marginal* of a joint prior, so the composite inversion is mean-field; the discrepancy is the mutual information between the branches (Remark 26). Mycelium's one-channel-at-a-time messages are the same laxness at the message level ([[Polarity Resolution]] §"Joint messages").
- **Almost surely** (footnote 3): inversions are unique only up to [almost-sure equality](https://mathstruct.org/CategoryTheory-ML-Wiki/Almost-Sure-Equality). Numerically, conditioning on a (near-)null event is exactly where the inverse is undetermined, and implementations must guard against it.
- **Inexact is legal**: a solver stopped early, a damped message, a moment-matched pushforward — all are inexact $c'$, which Definition 9 permits. The loss gets worse; nothing breaks. That is a gentler failure mode than a divergent unrolled solver.

````tabs
tab: Julia
**Docs:** [LenticulumCore: BayesianLens, invert](https://mathstruct.org/Lenticulum.jl/dev/packages/lenticulumcore/) · [Lenticulum: GaussianFactor](https://mathstruct.org/Lenticulum.jl/dev/packages/lenticulum/)
```julia
using Lenticulum, LenticulumCore, Mycelium
f = GaussianFactor(1 => 1; noise = 0.25, channels = (:x, :y))       # y = A x + b + ε
ps, st = (A = fill(2.0, 1, 1), b = [1.0]), NamedTuple()
# one factor, two lenses: assemble it forwards and backwards
fwd, _ = assemble(f, Polarity(; x = Observed(), y = Unobserved()), ps, st)
bwd, _ = assemble(f, Polarity(; x = Unobserved(), y = Observed()), ps, st)
post_y, _ = invert(fwd, uninformative(1), (x = DiracBelief([1.0]),), ps, st)
belief_mean(post_y), belief_cov(post_y)                  # ([3.0], [0.25;;]): prediction
post_x, _ = invert(bwd, Gaussian([0.0], fill(1.0, 1, 1)), (y = DiracBelief([3.0]),), ps, st)
round.(belief_mean(post_x); digits = 4)                   # [0.9412] = 16/17: Bayes' law
round.(belief_cov(post_x); digits = 4)                    # [0.0588;;] = 1/17
```
````
