# generator.jl — implementation note

> `NoiseSource` and `GeneratorFactor`: the ``z \sim q(z),\ x = G_\theta(z)`` half of a GAN.
> Two small factors that between them fill **two** slots `LenticulumCore` declared and nothing
> ever occupied.

## 1. The first `SampleBelief`

`LenticulumCore.SampleBelief` has existed since the beginning — *"a particle representation of
$\pi \in \mathcal{P}X$. The default fallback whenever no conjugate structure is available,
which is most of the time."* Nothing has ever constructed one. Every factor in the project so
far returns a `DiracBelief` or a `GaussianBelief`.

A generator is what the type was for, and the moment one exists the consequence is immediate:

```julia
combine(SampleBelief(...), SampleBelief(...))   # throws
```

which is `messages.md` §1's recorded main gap, no longer hypothetical. See [[ratio]] §4 for
the route around it.

## 2. The first `pushforward`

`open_model.jl` declares `forward`, `logdensity` and `pushforward`. This file implements the
third, for the first time — `LenticulumCore.pushforward(f::GeneratorFactor, π, ps, st)`.

And it contradicts, in a useful way, the warning attached to the declaration:

> *"This is one of the two expensive operations. Computing $c_*\pi$ is marginalisation, and is
> about as costly as exact inversion."*

True for densities. **False for samples**: pushing a particle set through a deterministic map
is `map`, with no integral anywhere. That asymmetry is the entire appeal of implicit
generative models, and it is worth stating as a rule:

| representation | `pushforward` | `logdensity` |
|---|---|---|
| density / parametric | expensive (marginalisation) | cheap |
| **particles** | **cheap (map)** | **unavailable** |

An implicit generative model is the bottom row. `isexact(::GeneratorModel) == true`, because
transporting particles through a deterministic map introduces no approximation at all — the
approximation was already in the particle set.

## 3. Weights survive a generator

`pushforward` carries a weighted `SampleBelief`'s weights through unchanged. That is not a
convenience: a deterministic map is a bijection on particle *indices*, so importance weights
attach to indices and are untouched. It is what lets a [[ratio]] factor be applied downstream
of a generator, and it is asserted in the test suite.

## 4. Implementation difficulties

### 4.1 One polarity, and it is not a wrapper limitation

`supported_polarities` returns one element. Inverting $G_\theta$ is the GAN-inversion problem;
`factor_message` on the latent channel throws, naming it.

The reason is structural rather than practical: **a generator has no residual.** A
`DEQFactor` has $r(x,z) = z - g(z,x)$ and can solve it for either argument
([[DEQ as a Relation]]); a generator has $x = G(z)$ and there is nothing to solve, only a
function to evaluate. So "implicit generative model" and "implicit learner" name different
things — [[Three Senses of Implicit]].

### 4.2 A generator's energy is zero, and that is the paper's whole point

`energy` returns `0.0`. A function has no residual, so every $(z, G(z))$ pair satisfies it
exactly and there is nothing to charge.

Which means a generator **cannot be trained from its own free-energy contribution**. All the
signal comes from a separate comparison factor — which is exactly Mohamed &
Lakshminarayanan's thesis that implicit models must be learned *by comparison* rather than by
likelihood ([[Implicit Generative Models]] §2). The zero here is not a stub; it is the
statement of the problem.

### 4.3 `nsamples` lives on the factor, and is ignored

`GeneratorFactor` carries `nsamples` and `rng` and uses neither: the particle count is
whatever the incoming belief has, because pushing forward maps over what arrives. The fields
would matter only if the factor sampled its own latents, which would duplicate `NoiseSource`.

Left in place because a `GeneratorFactor` asked to emit with *no* incoming latent belief
ought to fall back to its own prior — and it cannot, because it does not know one. Recorded
as dead configuration rather than removed, since the fix is a design decision (does a
generator own its prior, or is the prior a separate factor?) and [[Everything is a Factor]]
says the latter.

### 4.4 `NoiseSource` re-draws on every call

Each `factor_message` draws a fresh particle set from the RNG. On a single sweep that is
correct; under iterated message passing it means **the graph's latent cloud changes every
sweep**, so nothing converges and `belief_distance` between two draws is meaningless.

For the one-shot generative use this is right. For anything iterative the samples must be
frozen at the first draw — which is the common-random-numbers trick, and it is not
implemented.

Related: [[ratio]], [[Adversarial]], [[Implicit Generative Models]],
[[Three Senses of Implicit]], [[GANs as Two Factors]]
