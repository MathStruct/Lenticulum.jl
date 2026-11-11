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
| `combine` | addition of canonical parameters: exact, associative, total; keeps the `improper` flag only beside zero precision (`gaussian_belief.md`, "Declared improper") |
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

### After review (2026-10-30)

Codex's review of the fix found four gaps, now closed: the proper case called `inv` on a
Cholesky factor (no Zygote rule) and scalar-indexed diagonal device arrays, so it now goes
through `belief_cov` and LenticulumCore's `_precision_factor`/`_solve`/`_logdet`; finite
covariances near floatmax overflowed when added, so $\Sigma_c + \Sigma_d$ is scaled by its
largest diagonal entry; and an improper site far from the origin left log masses of size
$10^{16}$, so mixture products re-centre each improper site at one reference point (the first
proper component's mean on the other side; `_overlap(c, d, ref)`) and normalise by subtracting
the maximum first. `combine_with_overlap(c, d)` itself keeps the unshifted convention tested in
`overlap.jl`.

Second review (2026-10-30): the proper case scales per coordinate,
$D^{-1}(\Sigma_c + \Sigma_d)D^{-1}$ with $D = \operatorname{diag}(\sqrt{\max(\Sigma_{c,ii}, \Sigma_{d,ii})})$
(`_scaled_sum`, diagonal-preserving, so huge and tiny variances coexist and device/Zygote
paths broadcast); and mixture products re-centre improper sites only when the other side has
exactly one, since the dropped constant differs between sites.

## Expectation propagation (2026-11-03)

The Gaussian pieces of the `ExpectationPropagationRule()` (`passing.md`):

- `ep_project(m::MixtureBelief{<:GaussianBelief})`: the total mean $\mu = \sum_k w_k \mu_k$ and
  covariance $\sum_k w_k(\Sigma_k + (\mu_k - \mu)(\mu_k - \mu)^\top)$, as Lenticulum's `moment_match`, but
  returning `nothing` instead of throwing when a component is not proper or the covariance is not
  positive definite: the rule then skips the update.
- `ep_divide(p, c)`: $(\eta_p - \eta_c, \Lambda_p - \Lambda_c)$, declared improper unless the difference is positive
  definite. EP sites are often not: on the clutter problem a site's precision is negative for an
  outlier the cavity already explains.
- `_indefinite(b)`: a precision with an eigenvalue below $-16\, d\, \varepsilon\, \max|\lambda|$; an EP node
  skips an update whose factor would receive such a cavity, since the factors take positive
  semidefinite inputs only.

## Products of improper components (2026-11-03)

`combine_with_overlap(c, d)` used to require a proper `c`. When both are improper, unnormalised
exponentials $\exp(\eta^\top x - \tfrac12 x^\top \Lambda x)$ as a likelihood mixture carries them, the product is
exact: the exponentials multiply, $(\eta_c + \eta_d, \Lambda_c + \Lambda_d)$, and there is no overlap integral to take.
If the sum is proper it is normalised and its mass
$\tfrac12 \eta^\top \Lambda^{-1} \eta + \tfrac d2 \log 2\pi - \tfrac12 \log\det \Lambda$ moves into $\log Z$; otherwise it stays an unnormalised
exponential with $\log Z = 0$. With a re-centring point `ref` (LP-019) the same constant of `d` is
dropped as in the proper case, $d$ evaluated at `ref`. Two mixtures of likelihoods, the messages
of two data points to one variable, can now be pooled; exact belief propagation on the clutter
problem needs it (`test/mixture_rules.jl`, which also checks the formula on a 1-d pair and the
improper case).

The normalisation subtracts no log-partitions from each other, but it does compute
$\tfrac12 \eta^\top \Lambda^{-1} \eta$ directly, which is large when the product is far from the origin; the LP-019
care for proper pairs (the convolution form) has no counterpart here.

## Extended Gaussians (2026-11-10)

`combine` takes `LenticulumCore.ExtendedGaussian`s (`lib/LenticulumCore.jl/src/extended_gaussian.md`):
two of them, or one and a `GaussianBelief`, combine by the exact product (`_conjoin`, beside the
type), which stacks hard constraints, adds precisions on what they leave free, and keeps the
directions flat in both. A `DiracBelief` still dominates, as against a Gaussian, but is checked
first against the hard constraints of the other side: a clamp that violates one is an
`ArgumentError`, like two unequal clamps. The result of a product with an extended Gaussian is an
extended Gaussian, also when both inputs were representable in canonical form.

**The store takes them (2026-11-11).** `belief_distance` compares $\mu$, $\Sigma$ and the projector onto
the flat directions (independent of their basis); against a `GaussianBelief`, `DiracBelief` or
`TrivialBelief` it compares the other's extended form, so a message that changes kind converges.
`damp` is the geometric mixture `LenticulumCore._damp_extended`: the canonical rule on Gaussian
directions, gradual on directions flat on one side, the weighted average of two hard values.
`HardConstraintFactor` (Lenticulum) emits extended Gaussians under any schedule. Tests:
`test/runtests.jl`, "combine with extended Gaussians" and "the message store takes extended
Gaussians".

**Entropy (2026-11-11).** `variable_entropy` of an extended Gaussian is
`LenticulumCore.support_entropy`: the entropy relative to Lebesgue measure on the belief's support,
`0` for a point, as for a `DiracBelief`. A variable that a hard constraint fixes in some directions
therefore has a finite counting term, and flat directions (an improper marginal) throw. On trees
with a hard relation the Bethe free energy is then $-\log Z$ (Lenticulum `test/hard_free_energy.jl`;
the convention is in `lib/LenticulumCore.jl/src/extended_gaussian.md`, "Entropy").
