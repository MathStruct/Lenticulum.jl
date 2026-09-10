# [Lenticulum](@id lenticulum)

```@meta
CurrentModule = Lenticulum
```

The user-facing package: **Gaussian beliefs** and the **linear-Gaussian factors** built on
them. If you want to try the framework on something, start here — everything in this package
has a closed form, so you can check the answers.

## Gaussian beliefs

`GaussianBelief` stores a Gaussian in **canonical form** — information vector `η = Λμ` and
precision `Λ = Σ⁻¹` — rather than as a mean and covariance. Two reasons:

1. **Pooling becomes addition.** Combining two beliefs adds their canonical parameters, which
   is exact, associative, commutative and total.
2. **Improper beliefs are representable.** A factor → variable message is a *likelihood*, and
   a likelihood constraining only some directions has a singular `Λ`. The moment form cannot
   write that down at all.

```julia
g = Gaussian([1.0], fill(2.0, 1, 1))   # from mean and covariance
belief_mean(g), belief_cov(g)           # back again
uninformative(1)                        # η = 0, Λ = 0
isproper(g)                             # is Λ positive definite?
```

`belief_mean` on an improper belief throws rather than returning a plausible-looking
pseudo-mean.

## The factors

| factor | is | channels |
|---|---|---|
| `GaussianFactor` | ``p(y \mid x) = \mathcal{N}(y; Ax+b, Q)`` | two, named |
| `GaussianPrior` | ``\mathcal{N}(\mu, \Sigma)`` on one channel | one |
| `LinearConstraintFactor` | ``0 = \sum_i A_i x_i - c + \varepsilon`` | **any number** |

`GaussianFactor` is *bidirectional*: both `(x observed, y unobserved)` and the reverse, each
exact. Parameters are `(A, b)`; `noise` is a fixed hyperparameter, not learned.

`LinearConstraintFactor` is the **acausal** generalisation — one equation over `n` channels,
none of them distinguished, with one supported polarity per channel. It subsumes
`GaussianFactor` exactly (same messages, same energy) and is what you want for conservation
laws, balance equations and anything written as a relation rather than an assignment:

```julia
# Kirchhoff's current law at a three-way node: i₁ + i₂ + i₃ = 0
f  = LinearConstraintFactor((i1 = 1, i2 = 1, i3 = 1), 1; noise = 1e-8)
ps = (A = (i1 = ones(1,1), i2 = ones(1,1), i3 = ones(1,1)), c = [0.0])
```

## Known gaps

- `Q` is a fixed hyperparameter everywhere; learning it needs a positive-definite
  parametrisation, and a contrastive term to stop it collapsing to zero.
- A `LinearConstraintFactor` returns an uninformative message if any non-target channel is
  unconstrained, which is blunter than necessary.
- A loopy graph of constraint factors cannot bootstrap without a weak prior on every variable.

## API

```@index
Modules = [Lenticulum]
```

```@autodocs
Modules = [Lenticulum]
```
