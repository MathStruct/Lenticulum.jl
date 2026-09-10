# Language Models

> **Could several diffusion LLMs pass messages until they agree — a Mixture of Experts as a
> factor graph of expert models?**
>
> Yes, and the mechanism is real and has literature. But the name is wrong in an instructive
> way: what is described is a **Product of Experts**, which is the better object. And the
> factor-graph *machinery* buys much less than the factor-graph *framing* does — §4 is the
> deflation, §5 is where it would earn its place.
>
> This note sits apart from the other six in [[Motivating Examples]]: it shares their sixth
> property (the residual means something) and not their first (a network of relations).

## 1. Product, not mixture

The distinction is not pedantry — the two are different distributions with different behaviour.

| | form | logic | who must be satisfied |
|---|---|---|---|
| **Mixture** of experts | ``p(x) = \sum_i w_i\, p_i(x)`` | **OR** | any one |
| **Product** of experts | ``p(x) \propto \prod_i p_i(x)^{w_i}`` | **AND** | all of them |

"Run until they agree" is a conjunction: a sequence is good when *every* model finds it
plausible. That is the product, and in energy terms it is the operation this whole project is
built on:

$$E(x) \;=\; \sum_i w_i E_i(x)$$

**A factor graph is always a product.** `Mycelium.combine` multiplies densities and adds
energies ([[Energy-Based Factor Graphs]] §1); there is no mixture operation anywhere in the
framework, and there could not easily be one — a mixture is not a Frobenius multiplication.

Worth noting that "Mixture of Experts" in current LLM practice (Switch Transformer, Mixtral)
means something else again: **sparse routing**, where a learned gate sends each token to top-``k``
experts. That is a mixture with a router, chosen for compute efficiency, and it has nothing to
do with agreement.

## 2. Why *diffusion* LLMs specifically — the instinct is right

An autoregressive LLM is a chain of committed samples: it emits token ``t``, conditions on it,
and moves on. There is no full-sequence state to refine and no natural way for two of them to
negotiate — by the time they disagree, both have committed.

A diffusion LLM is different in exactly the way that matters. LLaDA, SEDD and MDLM generate by
**iteratively denoising a masked sequence**, with bidirectional attention and no causal mask, so
at every step there is:

- a **whole-sequence state** (partially masked), and
- a **per-position distribution over tokens** conditioned on it.

That intermediate object *is* a belief, and a denoising step *is* a message update. Which is why
the question lands on diffusion LLMs rather than on GPT-style models — the machinery needs
something to pass, and only the diffusion family has it mid-generation.

## 3. The belief type this wants is the easiest one the project lacks

Text is discrete, so the belief over a position is a **categorical distribution over the
vocabulary**. The project has `DiracBelief`, `GaussianBelief`, `SampleBelief`, `TrivialBelief`
— and no categorical.

That is worth adding, because it is the *cheap* one:

> [!important] `combine` for categorical beliefs is elementwise multiplication
> `messages.md` §1 records that pooling is blocked because it needs densities that no belief
> type has. For a distribution on **finite support** the density is a *table*. Combining two is
> elementwise multiply and renormalise — exact, cheap, associative, commutative, total.
>
> So the LLM case wants the one belief type for which the project's oldest recorded blocker
> simply does not arise.

Two details make it fit better than it first looks. A vocabulary of ``10^5`` times a sequence of
``10^3`` is large but the operation is elementwise, which is GPU-shaped. And a *per-position*
categorical is precisely the mean-field factorisation diffusion LLMs already make at each
denoising step — the belief representation matches the model's own approximation rather than
imposing a new one.

## 4. The deflation: for ``k`` experts on one sequence, the graph is a star

Be honest about what the graph looks like. Two ways to draw it:

- **Variables = token positions, factors = the LLMs.** Each model touches every position, so
  every factor has degree ``n``. That is a densely connected graph — the worst case for message
  passing, and nothing about it is sparse.
- **One variable = the whole sequence, factors = the LLMs.** Then it is a **star**: ``k`` unary
  factors on one variable.

On a star, message passing degenerates. The marginal is `combine` of the ``k`` messages, which is
**adding the ``k`` energies** — the thing you would have written anyway.

> The factor-graph machinery buys nothing here. Scheduling, the exclusion principle, polarity
> resolution and the Bethe correction all have nothing to do on a star. What survives is the
> *framing*: it is a product, energies add, and the residual means disagreement.

## 5. Where the graph would earn its place

When there is actual structure — which means when the factors have **different scopes**:

- **Heterogeneous spans.** A code model over code blocks, a mathematics model over equations, a
  general model over prose. Different factors covering different, overlapping regions is a real
  graph rather than a star.
- **Hard constraints.** A JSON schema, a grammar, a type-correctness check, a unit check. These
  are genuine acausal relations — a `LinearConstraintFactor` in spirit, a hard factor in effect
  — and combining them with a soft model belief is exactly what a factor graph does.
- **Tool outputs as clamped variables.** A retrieved document, a computed result, a compiler
  verdict: a `DataFactor` pinning part of the state, with everything else inferred around it.
- **Cross-document or multi-turn structure**, where a claim in one place must be consistent with
  a claim in another.

> [!note] Constrained decoding is already this
> Grammar-constrained decoding works by **masking logits** to forbid tokens the grammar
> disallows. That is `combine` of a categorical belief with a hard constraint factor, done at
> every step, by everyone, and never called factor-graph inference.

## 6. Much of modern decoding is already energy arithmetic

Once the framing is in place, a cluster of otherwise unrelated tricks turn out to be one
operation with different signs and weights:

| technique | in energy terms |
|---|---|
| logit averaging / ensembling | a **mixture** — arithmetic mean of probabilities |
| product-of-experts ensembling | ``\sum_i w_i E_i`` — energies add |
| classifier-free guidance | ``E(x\mid c) + w\,[E(x \mid c) - E(x)]`` — a weighted energy sum |
| **contrastive decoding** | ``\log p_{\text{expert}} - \log p_{\text{amateur}}`` — an energy **difference** |
| constrained decoding | combine with a hard factor |

Contrastive decoding is the interesting row: a *difference* of log-probabilities is exactly what
`Adversarial.RatioFactor` computes ([[ratio]] §1), used with a negative weight. So a factor
graph with one positive and one negative factor is contrastive decoding, and the graded energy
of [[Scalar and Multivariate Energy]] is the natural bookkeeping for all five rows at once.

## 7. The real technical obstruction: composing at ``t > 0``

This is the part that would bite in practice, and it has a literature.

Adding the scores of several diffusion models does **not** sample from the product of their data
distributions. If ``q_t`` is the forward noising kernel, then

$$q_t * (p_1 p_2) \;\neq\; (q_t * p_1)\,(q_t * p_2)$$

— convolution does not distribute over products. The composition is exact at ``t = 0`` and
**approximate all along the diffusion trajectory**, which is precisely where the models are
evaluated.

Du et al.'s *Reduce, Reuse, Recycle* is about this: the fix is MCMC (annealed Langevin or HMC)
at each noise level to correct the naive score sum. Liu et al.'s composable diffusion models
make the same composition work for images.

This is a laxness of exactly the shape the vault already tracks — a composition that leaves the
category and is corrected afterwards, like [[Composition is Elimination]]'s closure or
[[Composition of Bayesian Lenses]] Remark 16's mean-field tensor. It would want the same
treatment: report the defect rather than hide it.

## 8. Agreement can fail in specific ways

- **The product can be nearly empty.** If two experts disagree strongly, the product has low
  mass everywhere and is hard to sample from. Products of experts are famously difficult to
  sample — that difficulty is why contrastive divergence exists.
- **Different training data means different supports**, and the product concentrates on the
  intersection, which may be small or degenerate.
- **The product of two language models is not a language model.** The intersection of what
  every model finds probable is bland — agreement selects for generic text — and where the
  models' errors are complementary the product can put mass on sequences neither would have
  generated.

So "run until they agree" is not automatically an improvement; it is a specific inductive bias
with a specific failure mode, and the failure mode is blandness.

## 9. The cheap, genuinely useful thing

Independent of whether you *generate* from the product, the free energy of the graph is an
**agreement score**:

> High free energy means the experts do not agree. Per-factor energies say *which* one dissents.

That is a calibrated abstention signal — *don't answer, or escalate* — and per-expert
attribution of disagreement, obtained without any new machinery. It is the same observation as
[[Motivating Examples]] §6: the residual means something in the domain, and here it means
"the models are not confident together".

For a system that routes hard queries to a larger model, that is a directly usable quantity, and
it does not require the composition of §7 to be exact.

## 10. Verdict

- **The method is real** — product of experts over diffusion models, with a known correction for
  the ``t>0`` problem.
- **The framing is correct and clarifying**: it is a product not a mixture, energies add, and the
  residual is disagreement.
- **The machinery is overkill for a star** (§4), and this project would be a poor vehicle
  compared with adding scores directly in an established ML stack — there is no categorical
  belief, no GPU path ([[Parallelism and Compilation]]) and no tokeniser.
- **Where it would genuinely pay**: heterogeneous spans, hard grammar and type constraints, and
  tool outputs as clamped variables — that is, when the graph is actually a graph.

## Sources

- Hinton, *Training Products of Experts by Minimizing Contrastive Divergence*, Neural
  Computation 2002 — the product, and why it is hard to sample.
- Nie et al., *Large Language Diffusion Models* (LLaDA); Lou, Meng & Ermon, *Discrete Diffusion
  Modeling by Estimating the Ratios of the Data Distribution* (SEDD, ICML 2024); Sahoo et al.,
  *Simple and Effective Masked Diffusion Language Models* (MDLM) — the diffusion LLM family.
- Du et al., *Reduce, Reuse, Recycle: Compositional Generation using Energy-Based Diffusion
  Models and MCMC*, ICML 2023 — [arXiv:2302.11552](https://arxiv.org/pdf/2302.11552). Why the
  naive score sum is wrong and what fixes it.
- Liu, Li, Du, Tenenbaum & Torralba, *Compositional Visual Generation with Composable Diffusion
  Models*, ECCV 2022.
- Li et al., *Contrastive Decoding* — the energy-difference row of §6.

Related: [[Motivating Examples]], [[Energy-Based Factor Graphs]], [[Training Energy-Based Models]],
[[The Diffusion Family]], [[ratio]], [[Scalar and Multivariate Energy]], [[messages]],
[[Parallelism and Compilation]]
