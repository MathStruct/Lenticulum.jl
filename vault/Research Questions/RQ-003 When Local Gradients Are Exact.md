#open-problem

> **Status: open (2026-10-24).** Area: variational inference, optimisation · Prerequisites: statistical games, the envelope theorem, Fisher's identity · Related: [[Factors are Parameterized Statistical Games]], [[L-SPEC-002 Stop Gradient]], the CT-ML wiki's [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) and [Automatic Differentiation of Expected Values](https://mathstruct.org/CategoryTheory-ML-Wiki/Automatic-Differentiation-of-Expected-Values).
>
> Sources: St Clere Smithe & Perin, *AutoBayes*, arXiv:2503.18608, 2025 (Definitions 27–29, Theorem 23, Remarks 24, 30); the question is original to this vault.
>
> Bibliography: [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]]

## Question

When parameterized statistical games are composed, AutoBayes composes their gradients locally
(Definition 29) and drops two terms: where an upstream parameter moves the pushforward prior seen
downstream, and where a downstream parameter moves the sampling distribution of an upstream
inversion (Remark 30). **When are the dropped terms zero, and when they are not, how far is the
fixed point of local-gradient descent from a stationary point of the true composite free energy?**

## Setting

Two games $c : X \multimap Y$ with parameter $\theta$ and $d : Y \multimap Z$ with parameter $\varphi$,
a prior $\pi$ on $X$ (as a prior game, Remark 24), an observation $z$, and the composite free energy
$F(\theta, \varphi) = F^{dc}(\pi, z)$ of Theorem 23. The **local gradient** is
$\bigl(\mathbb E_{d'}[\nabla_\theta F^c],\ \nabla_\varphi F^d\bigr)$ with the inversions held fixed. In
Lenticulum this is `DiagonalCoupling()`, the default on every edge.

## What is known

- **Expected, to be proved:** if every inversion is exact (the Bayesian posterior), both dropped
  terms vanish. The term through the inversion vanishes by the envelope theorem, because the free
  energy is stationary in the belief at the exact posterior. The term through the pushforward prior
  should vanish by Fisher's identity $\nabla \log p(z) = \mathbb E_{\text{post}}[\nabla \log p(z, \cdot)]$.
  Then local gradients are exact gradients of $-\log p(z)$, as in the EM algorithm.
- With amortised or otherwise approximate inversions the terms are nonzero in general; this is
  what pathwise (reparametrisation) and score-function estimators restore, at a cost in variance.

## What would count as an answer

A proof of the exact-inversion case; for approximate inversions, a bound on the distance between
the fixed points of local and true gradient descent in terms of the inversion error (for example
the KL divergence from the exact posterior); and a linear-Gaussian chain where both can be computed
in closed form, as a test.

## Why it matters here

It decides when `DiagonalCoupling` can stay the default and when a graph must ask for
`PathwiseCoupling` or `ScoreFunctionCoupling`, which the stop-gradient specification
([[L-SPEC-002 Stop Gradient]]) has to settle.

## Leads

The envelope theorem for the exact case; the variational EM literature for the approximate one
(the gap between the ELBO gradient and the evidence gradient is a KL derivative).
