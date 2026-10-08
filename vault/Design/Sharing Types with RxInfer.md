#design #later

> Could `Mycelium` store its beliefs (and perhaps its factors) as the types RxInfer.jl already
> uses, instead of its own structs that are converted on the way in and out? **For beliefs,
> yes, by wrapping**: keep this project's abstract types and dispatch, store an ExponentialFamily
> object inside. **For factors, no**: the two packages mean different things by a factor, and the
> useful sharing is RxInfer's approximation code inside a hand-written nonlinear factor. Noted
> for later; nothing here is built.

> Sources: original to this vault (design); the source of ReactiveMP.jl 5.x (`src/approximations/`: linearization, unscented, Laplace, CVI) and of BayesBase.jl / ExponentialFamily.jl, read October 2026; Bagaev, Podusenko & de Vries, *RxInfer: A Julia package for reactive real-time Bayesian inference*, JOSS 2023
>
> Bibliography: [[Bibliography#^bagaev2023rxinfer|Bagaev et al. 2023]]
>
> Theory: [[RxInfer as a Backend]] · [[Belief Algebra]] · [[interop]]

## 1. Where things stand

Beliefs are this project's own structs (`GaussianBelief`, `CategoricalBelief`, …, in
`LenticulumCore`), with their rules in `Mycelium`. `as_distribution` / `as_belief` convert to
and from the BayesBase / ExponentialFamily types RxInfer passes ([[interop]]), allocating a new
object each way.

The allocation is not the problem: copying an $n \times n$ precision is cheap next to anything
done with it, and `as_distribution` could share the arrays instead of copying them. What a
shared type would really buy is **shared rules**: ExponentialFamily already has products,
entropies, densities and natural-parameter arithmetic for dozens of families (Gamma, Wishart,
Beta, Dirichlet, von Mises, …), which `Mycelium` would otherwise re-implement one by one.

## 2. Three ways to hold a belief

| | own struct (today) | use the RxInfer type directly | wrap the RxInfer type |
|---|---|---|---|
| example | `GaussianBelief(η, Λ)` | `MvNormalWeightedMeanPrecision` *is* the belief | `GaussianBelief{D}(d::D)`, `d` an ExponentialFamily object |
| conversion | allocate | none | unwrap / wrap, no copy |
| own supertype and dispatch (`AbstractBelief`) | yes | **no**: an external type cannot subtype ours | yes |
| families available | the six we wrote | all of ExponentialFamily's | all, one thin wrapper each |
| our extras (improper messages, labels, `TrivialBelief`, polarity-aware rules) | yes | must be bolted on outside the type | yes, in the wrapper |
| dependency of the belief layer | `LinearAlgebra` | ExponentialFamily (heavy) | ExponentialFamily (heavy) |

Using the types directly loses the one thing the project's design rests on: a belief type
hierarchy it controls (`AbstractBelief`, `isexact`, `combine` rules with polarity in mind).
Wrapping keeps it and still gets every rule for free by delegation: `combine` on two wrapped
Gaussians calls BayesBase's `prod`; entropy calls `entropy`; a family with no wrapper yet can
use a generic `ExpFamBelief{D}`.

## 3. Where the wrapped beliefs would live

Not in `LenticulumCore`: ExponentialFamily pulls in ForwardDiff, SpecialFunctions, HCubature
and more, and the core is meant to be light, like LuxCore. This is the `MyceliumCore` /
`Mycelium` split discussed earlier ([[Belief Algebra]] §6, item 3), now with a reason:

| package | content | dependencies |
|---|---|---|
| `LenticulumCore` | abstract types, the factor interface, today's light beliefs | light |
| `MyceliumCore` (new) | graphs, schedules, `combine` for the light beliefs | light |
| `Mycelium` | wrapped ExponentialFamily beliefs, their rules by delegation, the RxInfer bridges | ExponentialFamily, BayesBase |

Factor packages that only need Gaussians and points keep depending on the light layers.

## 4. Factors: share the approximations, not the structs

A Lenticulum factor and an RxInfer node are different objects:

| | Lenticulum factor | RxInfer node |
|---|---|---|
| is | a struct with parameters and state (`ps`, `st`), like a Lux layer | a declaration (`@node`) plus one rule per interface (`@rule`) |
| direction | chosen per call by a `Polarity` | one rule per direction, written in advance |
| learned | yes, by gradients through inference | parameters by inference over them |

A common struct would fit neither. What can be shared is the bridge in both directions, step 3
of [[RxInfer as a Backend]] §4 (a learned relation as an RxInfer node), and, the part that
matters most, **RxInfer's approximation code inside a hand-written nonlinear factor**.

ReactiveMP's delta nodes already implement, as reusable functions, the Gaussian approximations
through a known function $f$ (`src/approximations/`): `Linearization` (first-order Taylor),
`Unscented` (sigma points), Gauss–Hermite and spherical-radial cubature, `Laplace`, and CVI. A
`NonlinearFactor(f; method = Unscented())` in `Mycelium` could call these for its messages,
which gives Lenticulum the factor it lacks ([[Open Problems in Implicit Diffusion Learning]]
I11) without writing a sigma-point library. The cost is a ReactiveMP dependency (heavier than
ExponentialFamily, and its internals change between versions); the fallback is writing
linearisation and the unscented transform here, about a hundred lines with
DifferentiationInterface.

A hand-written nonlinear factor and a learned one would then sit side by side in one graph:
known physics ($y = h(x)$, a measurement model) next to a learned relation. That is the
combination real problems need (the particle and SLAM examples).

## 5. Risks

- **Load time and weight.** ExponentialFamily and ReactiveMP are large; time to first
  inference grows. Hence the split in §3.
- **Improper messages.** This project's likelihood messages often have a singular precision.
  ExponentialFamily represents them, but its `mean`, `cov` and `entropy` assume proper
  distributions; the wrapper must keep this project's checks.
- **API stability.** BayesBase and ExponentialFamily are stable packages; ReactiveMP's
  approximation internals are less so. Depend on the former; treat the latter as optional.
- **Automatic differentiation.** The implicit adjoints differentiate through beliefs only in
  a few places, but Zygote or Enzyme through ExponentialFamily types is untested here.
- **Labels and polarity.** Categorical labels and polarity-aware rules have no counterpart
  upstream; they stay in the wrappers.

## 6. When to do it

The trigger is the **first non-Gaussian exponential family** `Mycelium` needs: a Gamma belief
for a learned noise precision, a Wishart for a covariance, a Dirichlet for categorical
parameters. Writing those by hand would duplicate ExponentialFamily, so at that point wrap
instead (§2) and split the package (§3). The nonlinear factor (§4) can come earlier and
independently, with ReactiveMP's approximations as an optional extension and a small built-in
linearisation as the default.

Related: [[RxInfer as a Backend]], [[Belief Algebra]], [[interop]], [[gaussian_belief]],
[[Related Julia Projects]], [[Why Julia]]
