#definition

> `SampleBelief(samples, weights)` represents a distribution by **weighted particles**: the
> general-purpose fallback when no closed form is available, and the natural output of a sampler.

> Sources: code: `open_model.jl`, `messages.jl`; Doucet, de Freitas & Gordon (eds.), *Sequential Monte Carlo Methods in Practice* (Springer 2001)
>
> Bibliography: [[Bibliography#^doucet2001smc|Doucet et al. 2001]]
>
> Theory (CT-ML wiki): [Distribution Monad](https://mathstruct.org/CategoryTheory-ML-Wiki/Distribution-Monad) · [Giry Monad](https://mathstruct.org/CategoryTheory-ML-Wiki/Giry-Monad)

## What it is

A finite, weighted sum of point masses, $\sum_i w_i\,\delta_{x_i}$ (equal weights when `weights` is
`nothing`): an element of the finite [distribution monad](https://mathstruct.org/CategoryTheory-ML-Wiki/Distribution-Monad) used as an
approximation of a continuous distribution. It is an **approximation by construction**
(`isexact` is false), and two different particle sets can represent the same distribution.

## What it cannot do yet

- **Pooling.** `combine` of two sample beliefs needs the density of at least one of them
  (importance reweighting), and `belief_logdensity` is not defined for particles, so
  `combine` throws. A Dirac or the trivial belief still combines with it.
- **Pooling with evidence.** A density or `AbstractPotentialBelief` can reweight particles
  via `belief_logdensity`; two clouds alone still have no product rule. RatioFactor emits
  such a potential, and Mycelium warns on severe ESS collapse ([[ratio]], LP-016).
- **Convergence checks.** `belief_distance` compares weights and corresponding particles;
  different particle counts give `Inf`. It does not compare distributions independently
  of the particle ordering.
- **A point.** The diffusion factor's `assemble_state` ignores sample beliefs, because a particle
  set has no single value to clamp to.

It is the representation a **sampling** inference would return (DPS, Langevin;
[[Inference Signatures]] #6). Weighted particles can represent several branches of a
multivalued relation; mixtures provide another representation.

Related: [[Beliefs]], [[Dirac Belief]], [[Gaussian Belief]], [[Inference Signatures]]
