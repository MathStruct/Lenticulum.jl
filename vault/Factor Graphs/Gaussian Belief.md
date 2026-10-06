#definition #theorem

> `GaussianBelief(η, Λ)` is a Gaussian in **canonical (information) form**,
> $\mathcal N^{-1}(\eta, \Lambda) \propto \exp(-\tfrac12 x^\top\Lambda x + \eta^\top x)$, with precision
> $\Lambda = \Sigma^{-1}$ and information vector $\eta = \Lambda\mu$. In this form pooling is addition,
> and a precision may be singular, so likelihood messages are representable too.

> Sources: code: `beliefs.jl` (in the top-level `Lenticulum` package), `messages.jl`; Koller & Friedman, *Probabilistic Graphical Models* (MIT Press 2009), §14.2 (canonical forms)
>
> Bibliography: [[Bibliography#^koller2009pgm|Koller & Friedman 2009]]
>
> Theory (CT-ML wiki): [Gaussian Relations](https://mathstruct.org/CategoryTheory-ML-Wiki/Gaussian-Relations) · [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

## Why the canonical form

**Pooling is addition.** The product of two Gaussian densities over the same variable is

$$
\mathcal N^{-1}(\eta_1, \Lambda_1) \cdot \mathcal N^{-1}(\eta_2, \Lambda_2) \;\propto\; \mathcal N^{-1}(\eta_1 + \eta_2,\; \Lambda_1 + \Lambda_2),
$$

so `combine` is exact, associative, commutative and never fails. In moment form $(\mu, \Sigma)$ the
same operation needs two matrix inversions.

**Improper beliefs are allowed.** A factor → variable message is a *likelihood*, and a
likelihood that constrains only some directions has a rank-deficient precision (a measurement of
$x_1$ says nothing about $x_2$). The moment form cannot represent it at all; the canonical form
just has zeros. Such a belief is valid as a message and becomes a distribution once combined with
a prior that covers the remaining directions. `isproper(b)` checks $\Lambda \succ 0$, and `belief_mean`,
`belief_cov`, `belief_logdensity` and `variable_entropy` refuse improper beliefs rather than
return a silent pseudo-answer.

## The other beliefs are its limits

| limit | belief |
|---|---|
| $(\eta, \Lambda) = (0, 0)$ | the unit: [[Trivial Belief]] (`uninformative(n)`) |
| $\Lambda \to \infty$, $\mu$ fixed | a point mass: [[Dirac Belief]], which is why a Dirac dominates `combine` |
| a precision $\rho^2$ on some coordinates | a soft clamp ([[Channels and Polarity]]); the diffusion factor's $\rho$ is exactly this |

So the precision is a single dial from "no information" to "certainty", and the polarity
precisions of [[Implicit Diffusion Learners]] §2 are Gaussian beliefs on the clamped coordinates.

## Operations

- `Gaussian(μ, Σ)` builds one from moments; `belief_mean`, `belief_cov` go back.
- `logpartition(b)` $= \tfrac12\eta^\top\Lambda^{-1}\eta - \tfrac12\log\det\Lambda + \tfrac n2\log 2\pi$.
- `variable_entropy(b)` $= \tfrac n2(1+\log 2\pi) - \tfrac12\log\det\Lambda$. It can be **negative**, and the
  sign matters for the Bethe counting correction ([[Bethe Free Energy]]).
- `kl_divergence(q, p)` for proper beliefs; `belief_distance` compares canonical parameters
  directly, so it works for improper messages too.
- Gaussian messages can be damped (mixed) during loopy message passing ([[Loopy Message Passing]]).

## Where it is missing

`GaussianBelief` lives in the top-level package, so factors in `lib/` (equilibrium, diffusion)
cannot return one. Their point inferences have a natural Gaussian upgrade — the **Laplace
approximation** $\Lambda = J_{FF}$ at the solution, which the implicit diffusion adjoint already
computes ([[The Implicit Diffusion Factor as a Statistical Game]] §6).

Related: [[Beliefs]], [[Dirac Belief]], [[Trivial Belief]], [[The Linear Gaussian Chain]],
[[Messages are Inversions]], [[Bethe Free Energy]]
