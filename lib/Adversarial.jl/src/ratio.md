# ratio.jl — implementation note

> The discriminator, read as what it is: a **density-ratio estimator**. And the reason this
> file matters to Lenticulum has nothing to do with generating images.

## 1. The identity

Train a classifier to separate $p$ (label 1) from $q$ (label 0) with equal class priors. At
the optimum

$$D^\ast(x) = \frac{p(x)}{p(x)+q(x)}
\qquad\Longrightarrow\qquad
\operatorname{logit} D^\ast(x) = \log p(x) - \log q(x)$$

**The logit of the optimal discriminator is the log density ratio.** That is Mohamed &
Lakshminarayanan §3, it is why GANs work, and `logratio` / `discriminator` are the two readings
of the same number — the test suite checks they are logistic transforms of one another against
a closed-form Gaussian pair.

## 2. Why the energy is $-\log r$

$\mathbb{E}_{q}[-\log r] = \mathrm{KL}(q\,\|\,p)$. So the factor's energy is the integrand of a
KL divergence, and minimising it over whatever produced $q$ pushes $q$ toward $p$. That is the
generator's objective, expressed as a factor energy rather than as a training loop.

## 3. `AmortisedInversion` is literally true here

`lens.md` describes `AmortisedInversion(net)` as *"$c'$ realised by a learned network — a VAE
encoder… That a factor has **two independently parametrised halves** is the structural reason
a factor cannot be a Lux layer."*

This is the first factor in the project where that description is exact. The discriminator is
trained separately from whatever produced the samples it scores; the inversion genuinely is a
network with its own parameters. Every other factor's "inversion" is arithmetic on the forward
model.

## 4. The payoff: a route around the `combine` gap

`messages.md` §1 records the package's oldest blocker:

> *Two `SampleBelief`s cannot be pooled without importance reweighting, which needs
> `belief_logdensity`, which no belief type implements. **Every downstream feature — particle
> messages, conjugate messages, moment matching — is blocked on this.***

A log-density **ratio** is exactly what importance reweighting needs, and a classifier
estimates one **without either density**. `reweight` is the operation:

$$w_i \;\propto\; w_i^{\text{old}}\, r(x_i)$$

computed in log space with a max-shift, because raw ratios overflow for any interesting pair.
The test suite draws 20 000 samples from $q = \mathcal{N}(0,4)$, reweights by the exact ratio
against $p = \mathcal{N}(0.5,1)$, and recovers $p$'s mean and variance.

> [!important] So why is this not `Mycelium.combine`?
> Because the operation is defined but its **quality is not**. `effective_sample_size` can
> collapse to a handful of particles without any error being raised, and a `combine` that
> silently degenerates is worse than one that throws.
>
> There is a second reason, and it is sharper: **reweighting is not idempotent.** Applying the
> same ratio twice squares the weights, and the test suite asserts that doing so moves the
> answer *away* from $p$. `combine` is expected to be associative and commutative
> (`messages.md` §2); a ratio-based one is neither, unless the caller tracks which ratios have
> already been applied. That is bookkeeping the belief type cannot do on its own.

## 5. The energy is estimated, and nothing accounts for that

Every other factor in this project computes its energy. This one *fits* it: $\log r$ comes from
a network trained on a finite sample, so the energy carries an estimation error of unknown
sign and size.

`Bayesian Lens.md` licenses inexact **inversions** — *"the quality of the choice is what the
free energy measures."* This is a different animal: an inexact **energy**, and the free energy
has no term for it because the free energy *is* the thing being estimated. You cannot measure
the error of your ruler with the ruler.

Recorded as a genuine gap in the framework's accounting, not merely in this file. A candidate
fix is the discriminator's own validation loss as a confidence term, which nothing computes.

## 6. Implementation difficulties

### 6.1 The message is a posterior

A unary factor has no channel other than its target, so the only thing to reweight is the
incoming belief — making the message a posterior rather than a likelihood, double-counting on
a variable of degree > 1. The same wall [[deq]] §4.2 and `VariationalDiffusion`'s `factor.md`
§5.1 hit, for the same reason: nothing can be divided out of a neural network.

### 6.2 Only `SampleBelief` can be reweighted

`_score` throws on a parametric belief, because reweighting needs particles. A `GaussianBelief`
would have to be sampled first — and `GaussianBelief` is not reachable from a `lib/` package
anyway ([[The Equilibrium Family]] §5, the fourth package to hit this).

A `DiracBelief` passes through unchanged, matching `combine`'s rule that a hard clamp
dominates.

### 6.3 No training, so `logratio` is only as good as what you hand it

This package scores; it does not fit. An untrained `net` gives a meaningless ratio and
everything downstream is confidently wrong, with no diagnostic. The Gaussian oracle in the
tests exists precisely because a *perfectly trained* discriminator is the only one whose
output can be checked.

Related: [[generator]], [[Adversarial]], [[Implicit Generative Models]],
[[GANs as Two Factors]], [[messages]]
