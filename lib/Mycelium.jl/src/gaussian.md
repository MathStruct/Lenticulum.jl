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
Since 2026-11-12 (P1) this holds for a mixed pair too, an extended Gaussian and a `GaussianBelief`
in either order: the `GaussianBelief` is damped as its extended form. Before, `can_damp` had no
method for the pair, and `damp` returned the new message undamped without a sign, which happens
whenever a likelihood changes form between sweeps (flat while an input is missing, proper once it
arrives). Against a `GaussianBelief` of dual numbers (ForwardDiff), `combine`, `belief_distance` and `damp`
take the extended side in canonical form when it has no hard directions; with hard directions
there is no canonical form, and since P4 (2026-11-16) the dual side is taken in its extended form,
which moves at constant rank (LenticulumCore `extended_gaussian.md`, "Derivatives"). Until P4
that pair was an `ArgumentError` (`can_damp` was false). Before,
`combine` threw a `MethodError` there, inside `infer!`, on a variable that hears a dual flagged message
and a plain extended one ([[LP-028 Findings of the review of P1]], F1). `test/forms_crossing.jl` checks `combine`, `belief_distance` and `damp` on
150 random pairs of proper, flagged and zero-precision beliefs, each side in either form, against
the operation with both sides in canonical form (3 003 checks); the damping checks fail against
the code before the fix; and the dual pairs on 60 random instances (1 080 checks). Not checked: the distance near the rank threshold. A message whose
precision crosses it between sweeps changes form, and its distance in extended form then counts
the change of the flat subspace (up to 1) and the large variance, where the canonical distance
would be small. This is expected from the definitions, not measured on a run.
`HardConstraintFactor` (Lenticulum) emits extended Gaussians under any schedule. Tests:
`test/runtests.jl`, "combine with extended Gaussians" and "the message store takes extended
Gaussians".

**Entropy (2026-11-11).** `variable_entropy` of an extended Gaussian is
`LenticulumCore.support_entropy`: the entropy relative to Lebesgue measure on the belief's support,
`0` for a point, as for a `DiracBelief`. A variable that a hard constraint fixes in some directions
therefore has a finite counting term, and flat directions (an improper marginal) throw. On trees
with a hard relation the Bethe free energy is then $-\log Z$ (Lenticulum `test/hard_free_energy.jl`;
the convention is in `lib/LenticulumCore.jl/src/extended_gaussian.md`, "Entropy").

## Mixtures meeting extended Gaussians (2026-11-17, P27)

A `SwitchingConstraintFactor` on a variable that a `HardConstraintFactor` also reaches sends a
mixture of Gaussians where the relation sends an extended Gaussian with hard directions. The
mixture product (`_mixture_product`, `messages.jl`) needs `combine_with_overlap` for the pair, and
had none: `ArgumentError: no product rule for mixture components`.

**The overlap.** Let $d$ fix $H^\top x = h$ (its hard directions $H$, orthonormal) and be Gaussian
or flat on its support $S$, written $x = x_0 + U s$ with $U$ an orthonormal basis of $H^\perp$
(LenticulumCore `_hard_reduction`), where its Gaussian part is the canonical pair $(\eta_s,
\Lambda_s)$. The weight of a component $c$ is

$$
Z \;=\; \int_S p_c(x)\, p_d(x)\, d\sigma(x),
$$

with $\sigma$ Lebesgue measure on $S$ in the coordinates $s$. It is the limit of
$\int p_c(x)\,\mathcal N(x;\mu_d,\Sigma_d+\varepsilon I)\,dx$ as $\varepsilon\to0$, and finite: for $d$
flat nowhere it is $\mathbb E_{x\sim d}[p_c(x)]$, an expectation under a probability measure on $S$,
so weights of components against supports of any dimension compare. A proper $c$ factors on $S$
into the density of its marginal $H^\top x$ at $h$ and its conditional given $H^\top x = h$, whose
canonical form in $s$ is $(U^\top(\eta_c-\Lambda_c x_0),\, U^\top\Lambda_c U)$:

$$
\log Z \;=\; \log\mathcal N\bigl(h;\, H^\top\mu_c,\, H^\top\Sigma_c H\bigr) \;+\; \log\int c_s(s)\, d_s(s)\, ds .
$$

The second term is the Gaussian overlap above (`_overlap`), exact for $d$ proper on $S$ and up to a
constant of $d$ alone when $d$ is flat somewhere (as for an improper Gaussian site: the weights of
one mixture against the one site are right). Neither term subtracts large log-partitions. An
improper $c$ (an unnormalised exponential) has no marginal; its term is its value at $x_0$ plus the
overlap of its restriction. Without hard directions the rule is the Gaussian pair's. Two extended
Gaussians that both have hard directions are refused: two laws on lower-dimensional supports meet
in a set of measure zero for both, and their overlap is not finite. Code: `_extended_overlap`.

**Expectation propagation.** With `ExpectationPropagationRule()` on the switching node the cavity
is extended, and the tilted belief (cavity × mixture) a mixture of extended Gaussians that share
the cavity's hard directions. `ep_project` of it is the extended Gaussian of the total moments,
zero variance along those directions, so they stay hard (`nothing` if a component is flat
somewhere). `ep_divide(p, cavity)` is defined on the cavity's support only, where both are
Gaussian with log densities $\eta^\top x - \tfrac12 x^\top\Lambda x + \text{const}$ for their canonical
parts (zero precision along hard and flat directions): the site is the `GaussianBelief`
$(\eta_p-\eta_c,\,\Lambda_p-\Lambda_c)$, flat along the hard directions, declared improper, and as
any Gaussian site possibly indefinite. A projection whose hard directions or hard values differ from
the cavity's gives `nothing` (the update is skipped; the values were not checked before the review,
[[LP-046 Findings of the review of P27]], F1).

**An indefinite Gaussian times an extended one.** `combine(a::GaussianBelief, b::ExtendedGaussian)`
went through `extended(a)`, which refuses a negative direction, so the site above could not meet
the cavity again. For an indefinite `a` (`_indefinite`) the product is now the product of
potentials $[H_b^\top x = H_b^\top\mu_b]\cdot\exp((\eta_a+\eta_b)^\top x - \tfrac12 x^\top(\Lambda_a+\Lambda_b)x)$
through LenticulumCore `_from_potential` (`_potential_conjoin`). It checks that the summed precision is
positive semidefinite on what $b$ leaves free, since `_psd_split` would count a negative eigenvalue
as a zero one and return a flat direction where the product is not normalisable. It also checks
that nothing is left along a direction where the precision is zero. Other Gaussians take the old
path unchanged.

**Lenticulum's side.** The messages that leave the variable carry the mixture into the relation
factors: `HardConstraintFactor` refused a mixture input, and `LinearConstraintFactor` collapsed only
Gaussian mixtures. Both now collapse a mixture, also of extended components, to its total moments
(`moment_match`), as they collapse Gaussian mixtures (assumed-density filtering, Lenticulum
`gaussian.md`, "Mixture inputs").

**Tests.** `test/overlap.jl`, "P27: combine_with_overlap of a Gaussian and an extended Gaussian":
60 random instances (2 to 4 dimensions, 1 to $n$ hard directions, flat directions, proper and
improper components) against the closed form $\log\int c\,dd$ over $x = \mu_d + Gz + Ft$ (z standard
normal, t Lebesgue), to $10^{-9}$, plus a translation by $10^6$, the point, the refusal and the soft
case (383 checks).
"P27: expectation propagation with an extended cavity": `combine(ep_divide(p, cavity), cavity)` gives
$p$ back on 40 random instances, indefinite sites among them, and the product of an indefinite
Gaussian with a hard relation against the dense solve on the support, and the refusal of F1
(285 checks). Lenticulum
`test/mixture_extended.jl`: switching factor and hard relation on one variable, in plates, under
both schedules and both rules, against enumeration of the outcome (1152 checks, largest error
$1.4\times10^{-11}$).

**What it does not do.** Two extended components with hard directions: refused (above). No
re-centring for an improper component far from the origin (LP-019's reference point is used only
for an improper Gaussian site), so it may lose digits there; not measured. A proper one does not:
a translation of both sides by $10^6$ changes `logZ` by less than $10^{-7}$ (tested). Derivatives (ForwardDiff,
Zygote) through the new overlap and through `_potential_conjoin` are not tested. The Bethe free
energy of a graph whose variable marginal is a mixture has no `variable_entropy` (not for Gaussian
mixtures either: a mixture's entropy has no closed form), so `scalar_free_energy` refuses it under
`BeliefPropagation()`; under `ExpectationPropagationRule()` the marginal is the projection and it
runs (probe, not a test).

**Difficulties.** The first oracle, the limit $\Sigma_d + \varepsilon HH^\top$ and $L\cdot FF^\top$, agreed only
to $10^{-6}$ (truncation at large $\varepsilon$, rounding at small); the closed form over $x = \mu_d + Gz +
Ft$ replaced it. The silent flat direction of `_potential_conjoin` was found by its refusal test.

## Mixture reduction (2026-11-12, P17)

`reduce_mixture(m; max_components, prune)` and `reduce_message(::MixtureBelief{<:GaussianBelief}, M)`,
the message of a `MixtureReduction(M)` node, are defined here. They were in Lenticulum's `beliefs.jl`,
where the `reduce_message` method was type piracy (Mycelium's function on LenticulumCore's types),
and Mycelium without Lenticulum refused to reduce a Gaussian mixture. The code is unchanged; Lenticulum
re-exports `reduce_mixture`. The theory (Runnalls' bound on the KL divergence of a merge, which keeps the
total mean and covariance) and the refusal of improper components over the budget are in Lenticulum's
`beliefs.md`, "The reduction of a message".

Tested in `test/mixture_reduction.jl` with Mycelium and LenticulumCore only: on 30 random mixtures of
2–6 components, every budget from 1 to K + 1 gives the mixture unchanged within the budget, the
moment-matched Gaussian at one, and otherwise the requested number of components with the total mean
and covariance kept to $10^{-10}$. Mistral compared the moved functions with the code before the move on
40 random mixtures and found them identical ([[LP-029 Findings of the review of P17 and P18]]).

## Damping and the improper flag (I18, 2026-11-17)

`_damp` combined the metadata as `a.improper && b.improper`, so two improper sites with
complementary nullspaces — whose weighted precision sum is positive definite — kept the result
improper and `belief_cov` refused it. The flag of a damped pair is now decided from the combined
support (`_damped_improper`): when both sides are flagged, the result is proper only if each
precision is numerically rank-deficient (the flag's claim, visible in the spectrum) and the sum is
full rank; coincident nullspaces, a flag contradicted by a full-rank side, and a singular sum stay
improper, and the endpoints reduce to the flagged side's own. The rank is computed on values
(`stop_gradient`) with the tolerance of the decomposition's element type (`_numerical_rank`), so
a Float32 precision is judged at `eps(Float32)`. The device `_damp` (diagonal precisions) applies
the same rule through reductions only. Tests: `test/runtests.jl`, "I18" (complementary, coincident,
contradicted flags, endpoints, dense and diagonal precisions, Float32), and
`test/device_improper.jl` under `LENTICULUM_DEVICE_TESTS=true`.
