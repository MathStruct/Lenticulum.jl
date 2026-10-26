#implementation

> Gaussian belief pooling, densities, entropy, damping and public log overlaps
> live in Mycelium alongside the categorical and mixture rules.

> Sources: code: `gaussian.jl`, `../test/overlap.jl`
>
> Theory: [[Gaussian Belief]] · [[Belief Algebra]] · [[Bethe Free Energy]]

Implements: the Mycelium belief interface for `LenticulumCore.GaussianBelief`.

## 1. The rules

| function | for Gaussians |
|---|---|
| `combine` | addition of canonical parameters: exact, associative, total |
| `combine` with a `DiracBelief` | the Dirac wins (the $\Lambda \to \infty$ limit) |
| `belief_logdensity` | closed form; proper beliefs only |
| `variable_entropy` | $\tfrac n2(1 + \log 2\pi) - \tfrac12\log\det\Lambda$, may be negative |
| `belief_distance` | max-norm on $(\eta, \Lambda)$, defined for improper beliefs too |
| `can_damp`, `_damp` | convex combination of canonical parameters |
| `combine_with_overlap` | the pooled Gaussian and its log-overlap; proper first argument, second may be improper |

## 2. History

These methods were in the top-level `Lenticulum` package with the type, as extensions of
Mycelium's functions. They moved here when the type moved to `LenticulumCore`
([[gaussian_belief]] §2), so that the rules sit beside the categorical and mixture rules in
[[messages]]. The numerical rules are unchanged; the overlap rule is now public.

Related: [[messages]], [[free_energy]], [[gaussian_belief]]

For proper c,d, logZ is log N(μ_c;μ_d,Σ_c+Σ_d). For improper d it is
A(c⊙d)−A(c), with its undefined normalisation constant omitted. That constant
depends only on d and cancels across mixture components. `test/overlap.jl`
checks the proper formula and differences across two proper first arguments
for a rank-one improper likelihood.
