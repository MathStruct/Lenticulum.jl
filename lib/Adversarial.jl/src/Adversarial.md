# Adversarial.jl — package note

> Implicit **generative** models as factors — Mohamed & Lakshminarayanan
> ([arXiv:1610.03483](https://arxiv.org/pdf/1610.03483)) — and the GAN diagram read as a factor
> graph.

## Three factors, all unidirectional

| factor | is | note |
|---|---|---|
| `NoiseSource` | the latent prior $q(z)$, as an emitting factor | [[generator]] |
| `GeneratorFactor` | $x = G_\theta(z)$ — samples out, no density | [[generator]] |
| `RatioFactor` | $\log r(x) = \log p(x) - \log q(x)$ | [[ratio]] |

Every one has exactly one polarity, so **the GAN diagram is a DAG** — Lux-shaped, using none
of Mycelium's bidirectionality. That is the finding, not a shortcoming: a generator is a
function with no residual behind it. See [[Three Senses of Implicit]].

## What it filled in

Two slots `LenticulumCore` declared and nobody had ever occupied:

- **`SampleBelief`** — declared as *"the default fallback whenever no conjugate structure is
  available, which is most of the time"*, and never constructed until now.
- **`pushforward`** — declared alongside `forward` and `logdensity`, never implemented. And
  for particles it is *cheap*, against the warning attached to its declaration; see
  [[generator]] §2.

## What it exposed

- **The `combine` gap is now reachable.** Two `SampleBelief`s exist, and `Mycelium.combine`
  throws on them — `messages.md` §1's oldest recorded blocker, no longer hypothetical.
- **And a discriminator is a route around it.** A log-density *ratio* is what importance
  reweighting needs, and a classifier estimates one without either density. `reweight` does
  it; [[ratio]] §4 says why it is still not `combine` (not idempotent, and quality
  unreported).
- **An estimated energy is a new kind of inexactness.** `Bayesian Lens.md` licenses inexact
  *inversions*; this factor has an inexact *energy*, and the free energy has no term for it
  because the free energy is the thing being estimated. [[ratio]] §5.
- **`AmortisedInversion` is literally true for the first time.** The discriminator is a
  network with its own parameters, trained separately — which `lens.md` calls the structural
  reason a factor cannot be a Lux layer.

## What it cannot do

**Train adversarially.** The generator descends the same quantity the discriminator ascends,
and a Bethe free energy has one sign. The graph holds the GAN's *wiring* exactly and its
*objective* not at all — [[GANs as Two Factors]] §4 argues the missing structure is
Ghani–Hedges–Winschel–Zahn's **open game**, a third lens-shaped object beside the parametric
lens and the statistical game.

Dependencies: `LuxCore`, `Random`, `LinearAlgebra`. No `Lux`, no AD, no training loop.

Concept notes: [[Implicit Generative Models]], [[GANs as Two Factors]],
[[Three Senses of Implicit]].
