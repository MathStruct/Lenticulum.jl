# Implicit Generative Models

> Mohamed & Lakshminarayanan, *Learning in Implicit Generative Models*
> ([arXiv:1610.03483](https://arxiv.org/pdf/1610.03483)). Entry point for the adversarial
> family, implemented in `lib/Adversarial.jl`.
>
> The paper's thesis in one line: **without a likelihood you must learn by comparison**, and
> every way of comparing two distributions you can only sample from reduces to estimating a
> density ratio or difference.
>
> Its relevance here is not GANs. It is that a density ratio is exactly what
> [[messages]] §1 has been blocked on since the beginning.

## 1. The definition

An **implicit generative model** specifies a sampling procedure and no density:

$$z \sim q(z), \qquad x = G_\theta(z)$$

You can draw $x$. You cannot evaluate $p_\theta(x)$ — doing so would require the change-of-
variables formula, which needs $G_\theta$ invertible with a tractable Jacobian determinant.
(A normalising flow is exactly the case where you pay for that; see
[[Three Senses of Implicit]] §2 for why that makes a flow *not* implicit in this sense.)

This is a different sense of "implicit" from [[README]]'s, and conflating the two is the
subject of [[Three Senses of Implicit]]. In short: an implicit *generative* model gives up the
**density** and keeps the **direction**; an implicit *learner* gives up the **direction**.

## 2. Learning by comparison

No likelihood ⇒ no maximum likelihood. What is left is to compare the model $q_\theta$ with
the truth $p^\ast$ using only samples from each, and drive the comparison to indifference. The
paper organises the ways of doing that into four, all of which turn out to estimate the same
object:

| approach | estimates | gives you |
|---|---|---|
| **class-probability estimation** | $r = p^\ast/q_\theta$ via a classifier | **GANs** |
| **divergence minimisation** | an $f$-divergence via its variational bound | $f$-GAN |
| **ratio matching** | $r$ directly, by least squares | LSQ ratio estimation |
| **moment matching** | differences of feature expectations | MMD / GMMN |

The unifying object is the **density ratio** $r(x) = p^\ast(x)/q_\theta(x)$, and the paper's
point is that all four are ways of getting at it without either density.

There is a second connection the paper draws that is worth keeping in view: this is the same
problem as **likelihood-free inference** / ABC in statistics, where a simulator plays the role
of $G_\theta$. Which means the machinery here is not a deep-learning trick — it is the
established answer to "I have a simulator and no likelihood".

## 3. The identity that makes it work

Train a classifier to separate $p^\ast$ (label 1) from $q_\theta$ (label 0) with equal class
priors. At the optimum

$$D^\ast(x) = \frac{p^\ast(x)}{p^\ast(x)+q_\theta(x)}
\qquad\Longrightarrow\qquad
\operatorname{logit} D^\ast(x) = \log p^\ast(x) - \log q_\theta(x)$$

**The logit of the optimal discriminator is the log density ratio.** Everything else in the
GAN literature is a consequence of this line plus a choice of what to do with $r$.

`Adversarial.RatioFactor` is this identity as a factor: `logratio` and `discriminator` are the
two readings of one number, and the test suite checks them against a closed-form Gaussian
pair, where the optimal discriminator's logit is a known quadratic.

## 4. What it looks like as factors

Three factors, all unidirectional ([[GANs as Two Factors]] §3):

| factor | is |
|---|---|
| `NoiseSource` | $q(z)$ — the latent prior, emitting a `SampleBelief` |
| `GeneratorFactor` | $x = G_\theta(z)$ — the pushforward |
| `RatioFactor` | $\log r(x)$ — the comparison |

The division of labour is exactly §2's thesis made structural: **the generator carries no
energy at all** (`energy` returns `0.0`; a function has no residual, so every $(z, G(z))$ pair
satisfies it), and *all* the learning signal lives on the comparison factor. That zero is not
a stub — it is the statement of the problem.

## 5. Why this matters beyond GANs

Here is the connection that makes the paper worth a note in this vault rather than a citation.

`messages.md` §1 has recorded the same blocker since the package began:

> *Two `SampleBelief`s cannot be pooled without importance reweighting, which needs
> `belief_logdensity`, which no belief type implements. **Every downstream feature — particle
> messages, conjugate messages, moment matching — is blocked on this.***

And importance reweighting does not need densities. It needs a **ratio**:

$$w_i \;\propto\; \frac{p(x_i)}{q(x_i)} \;=\; r(x_i)$$

which is precisely what §3 estimates from samples alone. So:

> [!important] The GAN discriminator is a candidate answer to this project's oldest gap
> `Adversarial.reweight` performs the pooling. The test suite draws 20 000 samples from
> $q=\mathcal{N}(0,4)$, reweights by the exact ratio against $p=\mathcal{N}(0.5,1)$, and
> recovers $p$'s mean and variance. The operation works.

It is still not `Mycelium.combine`, for two reasons recorded in [[ratio]] §4:

1. **Quality is unreported.** `effective_sample_size` can collapse to a handful of particles
   with no error raised, and a `combine` that silently degenerates is worse than one that
   throws.
2. **Reweighting is not idempotent.** Applying the same ratio twice squares the weights, and
   the test suite asserts this moves the answer *away* from $p$. [[messages]] §2 requires
   `combine` to be associative and commutative; a ratio-based one is neither unless the caller
   tracks which ratios have already been applied — bookkeeping the belief type cannot do.

Both are surmountable and neither is surmounted. But the direction is now clear, and it did
not come from the message-passing literature.

## 6. What it costs

**The energy becomes an estimate.** Every other factor in this project computes its energy;
this one *fits* it, so the energy carries estimation error of unknown sign and magnitude.
[[Bayesian Lens]] licenses inexact *inversions* — "the quality of the choice is what the free
energy measures" — but this is an inexact **energy**, and the free energy has no term for it
because the free energy is the thing being estimated. [[ratio]] §5.

**And the comparison is adversarial.** Fitting $r$ is itself a learning problem played against
the generator, which is where [[GANs as Two Factors]] §4's sign obstruction comes in.

## Sources

- Mohamed & Lakshminarayanan, *Learning in Implicit Generative Models*,
  [arXiv:1610.03483](https://arxiv.org/pdf/1610.03483) — the definition, the four estimators,
  the likelihood-free-inference connection.
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014 — the original, and the
  optimal-discriminator computation §3 rests on.
- Nowozin, Cseke & Tomioka, *f-GAN*, NeurIPS 2016 — the divergence-minimisation column.
- Sugiyama, Suzuki & Kanamori, *Density Ratio Estimation in Machine Learning*, 2012 — the
  statistics this predates GANs by.
- Tran, Ranganath & Blei, *Hierarchical Implicit Models and Likelihood-Free Variational
  Inference*, NeurIPS 2017 — ratio estimation used for *inference* rather than generation,
  which is the reading §5 needs.

Related: [[GANs as Two Factors]], [[Three Senses of Implicit]], [[Adversarial]],
[[generator]], [[ratio]], [[messages]], [[Implicit Learners]], [[Bayesian Lens]]
