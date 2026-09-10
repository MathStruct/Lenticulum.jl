# The Diffusion Factor

> Where [[ImplicitREDDiff]]'s selection matrices stop being notation. The factor owns one
> space $\mathbb{R}^n$, its channels are blocks of that space, and a `Polarity` supplies
> $P_{in}, P_{out}, P_{latent}$ and their precisions.
>
> Implemented in `lib/VariationalDiffusion.jl/src/factor.jl`; see [[factor]].

## 1. The polarity *is* the selection matrices

The prompt's sketch asks for diagonal $P_{in},P_{out},P_{latent}\in\{0,1\}^{n\times n}$ with

$$P_{in}+P_{out}+P_{latent}=\mathrm{Id},\qquad P_{in}P_{out}=P_{in}P_{latent}=P_{out}P_{latent}=0$$

and then $P=\rho_{in}P_{in}+\rho_{out}P_{out}+\rho_{latent}P_{latent}$. Every piece of that
already existed in `LenticulumCore`, which is the point of
[[Channels and Polarity]] — the note that reconciled the two notations before there was
anything to reconcile them for:

| the sketch | `LenticulumCore` |
|---|---|
| $P_{in}$ | channels marked `Observed()` |
| $P_{out}$ | channels marked `Unobserved()` |
| $P_{latent}$ | channels marked `Latent()` |
| the partition identity | `ispartition` — free, from the `NamedTuple` representation |
| $\rho_{in},\rho_{out},\rho_{latent}$ | `channel_precision(p, name)` |
| $\rho_{in}=\infty$ (a hard clamp) | `default_precision(Observed()) == Inf` |

So `precision_vector(f, p)` is a *reading* of the polarity and the factor stores no
configuration for it. Note which way the default falls: **the hard clamp is the default; the
soft clamp is what you opt into.** An observation is evidence unless you say otherwise — and
$\rho_{in}=\infty$ is the categorical *cup* of [[Copiers Cups and Caps]], taken literally by
projection rather than numerically by a large penalty.

## 2. One network, one state space, many directions

The `NoisePredictor` is over the whole $\mathbb{R}^n$, not one per channel. That is what makes
this a **relation** rather than a bundle of conditionals: the model knows the joint
distribution, so any subset of blocks can be inpainted from any other.

Concretely this is why `supports_polarity` accepts *any* assignment with at least one
unobserved channel, while `supported_polarities` enumerates only the $n$ "one unobserved, rest
observed" cases. The predicate is the truth; the enumeration is the listable subset a
scheduler can plan with. `channels.md` says the two exist so they *can* differ; this is the
first factor where they do, and the full set has $3^n-2^n$ elements.

Read against [[README]]'s table, this factor scores the "symmetry handling: no distinguished
input/output" row about as well as anything in the repository —
[[ModelingToolkit as an Acausal Relation]]'s `LinearConstraintFactor` gets it by having no
direction, this one by having a prior over everything at once.

## 3. The inversion is a proximal operator, which the core already knew

```julia
assemble(f, polarity, ps, st) -> BayesianLens(DiffusionModel(f, polarity),
                                              ProximalInversion(f.prox))
```

`ProximalInversion` was already in `lens.jl`, with a docstring naming *RED-Diff*,
*ProxDM* and *this package*, written before any of it existed. That the implementation
slotted in without touching the core is the strongest evidence so far that
[[Bayesian Lens]]'s "nothing constrains the inversion to be exact" was a real abstraction and
not a hedge.

The model is **not pure**: `latentspace` returns the `Latent()` channels, which the prox
reconstructs and nobody reads. That is exactly AutoBayes' $\llbracket c\rrbracket$ — internal
coordinates composition hid — and it is the first factor in the project with a non-trivial one.

## 4. What the graph gets, and it is a Dirac

`invert` returns a `DiracBelief`, faithfully to RED-Diff's $\sigma\to0$ variational family.
On a single inverse problem that is unobjectionable. In a factor graph it has three
consequences, and all three are real problems rather than infelicities.

### 4.1 The message is a posterior, not a likelihood

Every other factor returns a *likelihood* from `factor_message`, with the prior divided out,
because a variable of degree $d$ would otherwise count the prior $d$ times —
[[Messages are Inversions]] is built on the distinction, and `gaussian.jl`'s `invert`
docstring is explicit that `combine(π, message) == posterior`.

**A diffusion factor cannot divide its prior out.** The prior is a neural network; there is no
subtraction available in any representation. So:

> Correct at degree 1 — one diffusion prior, one measurement, the usual inverse problem.
> Approximate at degree > 1, and nothing detects it.

### 4.2 A Dirac dominates every `combine` it meets

`Mycelium.combine` gives a `DiracBelief` absolute precedence (it is the $\Lambda\to\infty$
limit, per [[Channels and Polarity]]). So a diffusion factor does not *negotiate* with its
neighbours, it **overrides** them; and two diffusion factors disagreeing about one variable
raise "contradictory hard clamps" rather than averaging.

### 4.3 The entropy slot holds something that is not an entropy

The Bethe free energy of [[Bethe Free Energy]] wants $F_c = U_c - H(b_c)$ with a counting
correction $(1-d_v)$ per variable. Here $b_c$ is a Dirac and $H(b_c) = -\infty$, so the
score-matching term stands in for it — structurally correct (it depends on the learned
distribution) and numerically not an entropy.

**Consequence:** mixing a `DiffusionFactor` with a `GaussianFactor` in one graph gives a total
free energy that is not $-\log p(y)$ for any model. The identity that
[[The Linear Gaussian Chain]] §4 verifies to machine precision simply does not hold once this
factor is in the graph.

## 5. The fix, and it is in the paper

Keep $\sigma>0$. RED-Diff's §3 derives the general Gaussian variational family and the
experiments drop it; dropping it is what produces every problem in §4. A `GaussianBelief` with
finite precision would:

- make the message poolable rather than dominant,
- give a real entropy for the Bethe sum,
- and let the prior be (approximately) divided out in canonical form, restoring the
  likelihood/posterior distinction.

It needs a variance update in the prox's inner loop — a second optimisation, over $\sigma$ —
and it is the single most valuable missing piece in this family.

## 6. Other gaps, briefly

- **`local_free_energy` re-runs the prox**, because the message store keeps beliefs rather
  than the internal state the prox converged to. Caching on `st` is the obvious fix.
- **The free energy is stochastic.** Calling it twice gives different numbers. Every other
  factor's is a closed form, and anything in `Mycelium` that compares free energies across
  iterations will see noise.
- **No convergence report.** `reddiff_solve` runs exactly `steps` iterations and returns;
  there is no residual to measure, so there is nothing to report. Against
  `Mycelium`'s conventions, where every inversion says whether it converged.
- **`energy`'s signature does not fit.** The method here reads the `a` argument — AutoBayes'
  latent slot — as the polarity, because $P$ depends on it and there is nowhere else to put
  it. [[ModelingToolkit as an Acausal Relation]] hit the same wall from the acausal side; two
  independent factor families now means it is an interface bug in `LenticulumCore`, not a
  quirk of either.

Related: [[The Diffusion Family]], [[RED-Diff as a Statistical Game]], [[The VP-SDE]],
[[Channels and Polarity]], [[Copiers Cups and Caps]], [[Messages are Inversions]],
[[Bethe Free Energy]], [[factor]], [[ImplicitREDDiff]]
