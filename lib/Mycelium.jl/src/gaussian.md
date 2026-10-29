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


## Device diagonal messages (2026-10-27)

Gaussian entropy/density use Core's precision-logdet helper. The CPU path still
uses `logdet` of its Cholesky factor; a device Diagonal uses `sum(log, diag)`.
Julia's generic diagonal logabsdet reduction has no GPU neutral element for dual
pairs, which broke ForwardDiff replay even when the counting coefficient was zero.
Elementwise positive-diagonal algebra avoids this reduction. Exact/pathwise and
terminal score tests with JLArrays compare against CPU and closed-form derivatives
in `lib/ImplicitLayers.jl/test/device_arrays.jl`. Dense device precision remains
unsupported. See [[gaussian_belief]] and [[gradient_couplings]].

Device density/entropy specialisations use only diagonal fields and typed scalar
constants. Mixing a native diagonal matrix-product pullback with direct `diag`
field reads produced incompatible Zygote cotangents; the elementwise quadratic
form avoids that mix. Generic `inv(Diagonal(device))` also scalar-indexes on this
stack: Core's device moment constructor/covariance accessor instead broadcast
reciprocals. Their derivatives and score estimates are explicitly tested.


## Batch C damping precision (2026-10-28)

The GPUArraysCore extension casts damping weights to diagonal-Gaussian or Dirac
payload precision, before broadcasting. A Float64 keyword weight otherwise widens
Float32 device messages. CPU damping arithmetic is unchanged. New realised
flooding/residual replay tests compare Float32/Float64 diagonal Gaussian tree/loop
messages and derivatives against CPU references with scalar indexing disabled.
See `test/graph_adjoint_devices.jl` and [[gradient_couplings]].

## Overlaps without cancellation (2026-10-29, LP-019 items 2 and 6)

`combine_with_overlap` used to return $A(c \odot d) - A(c) - A(d)$. Each log-partition holds a
term $\mu^\top\Lambda\mu/2$, and at $|\mu| \sim 10^8$ the difference loses every digit: two
identical $\mathcal N(10^8, 1)$ gave $-4$ instead of $-\tfrac12\log 4\pi$, and a mixture's
product weights shifted by $10^8$ became `[0.5, 0.5]` instead of `[0.5312, 0.4688]`. Proper
pairs now use the convolution $\log\mathcal N(\mu_c - \mu_d;\, 0,\, \Sigma_c + \Sigma_d)$, which
depends only on the difference of the means. For an improper $d$ the value is still
$\log\int p_c(x)\exp(\eta^\top x - x^\top\Lambda x/2)\,dx$ (the convention `overlap.jl` tests),
expanded about $\mu_c$. That value is genuinely of size $\eta^\top\mu_c$, which is harmless when
the mixture weights are normalised.

Translation invariance holds until the canonical form itself fails: beyond $|\mu| \sim 10^{12}$,
$\mu = \Lambda^{-1}\eta$ is recovered only to $\sim 10^{-16}|\mu|$, which no overlap formula
repairs.

The rule needs a proper first argument. `_mixture_product` now puts a proper Gaussian first,
whichever side of the product it came from, so pooling a mixture with an improper site no
longer depends on the factor insertion order. Tested in `test/lp019.jl`.
