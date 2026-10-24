#open-problem

> **Status: open (2026-10-24).** Area: information geometry, message passing · Prerequisites: Fisher metric, belief propagation · Related: [[Scalar and Multivariate Energy]] §6, [[LP-002 The Jacobian side is not implemented]], the CT-ML wiki's [Information Geometry of Probabilistic Programs](https://mathstruct.org/CategoryTheory-ML-Wiki/Information-Geometry-of-Probabilistic-Programs) and [Natural Gradient](https://mathstruct.org/CategoryTheory-ML-Wiki/Natural-Gradient).
>
> Sources: Karwowski, Nielsen & Staton, *Compositional Information Geometry of Probabilistic Programs*, arXiv:2610.17886, 2026 (Proposition 8, Theorem 2, Example 1); Martens, *New Insights and Perspectives on the Natural Gradient Method*, JMLR 21, 2020 (§9.2); Welling & Teh, *Linear Response Algorithms for Approximate Inference in Graphical Models*, Neural Computation 16, 2004; the question is original to this vault.
>
> Bibliography: [[Bibliography#^karwowski2026compositional|Karwowski et al. 2026]] · [[Bibliography#^martens2020natural|Martens 2020]] · [[Bibliography#^welling2004linear|Welling & Teh 2004]]

## Question

For probabilistic programs, Karwowski, Nielsen & Staton prove a chain rule for the Fisher
metric of a composite: $g^{G\circ F} = g^F + F^\sharp g^G$, the local metric plus the downstream metric
pulled back (arXiv:2610.17886, Proposition 8(7)). **Is there a local formula of this kind for the
Fisher metric in the parameters of a factor graph with loops, computed from the messages, and does
natural-gradient descent with it converge to the same points as with the exact metric?**

## Setting

A factor graph with parameterized factors $\psi_a(z_a; \theta_a)$, data clamped on some variables,
and the model evidence $p(\text{data} \mid \theta)$. The exact Fisher metric is
$I(\theta) = \mathbb E[\nabla_\theta \log p \,\nabla_\theta \log p^\top]$. A *local* metric is one assembled
from per-factor quantities evaluated at the current messages, e.g. per-factor Gauss–Newton blocks
$J_a^\top Q_a^{-1} J_a$ plus coupling blocks.

## What is known

- For first-order probabilistic programs, composed sequentially and in parallel, the chain rule
  above holds exactly for the presentation-level metric of Karwowski et al., which is lax monoidal
  and depends on how the model is presented (their Theorem 2 and Example 1). Whether it carries over
  to an undirected factor graph, even a tree, is part of the question.
- For a single Gaussian factor with mean $f(\theta)$ the Fisher metric is the Gauss–Newton matrix
  $J^\top Q^{-1} J$ (Martens, §9.2); Lenticulum keeps $J$ available for this (`energy_jacobian`, by
  finite differences for now).
- On loopy graphs nothing is known here.

## What would count as an answer

A formula for a local metric on loopy graphs with its error relative to the exact metric (for
example in terms of the Bethe approximation error), or a proof that no local metric can be exact;
and an experiment on a small loopy linear-Gaussian graph where the exact metric is computable.

## Why it matters here

AutoBayes intends parameterized games to be trained by Fisher-metric descent. A local metric is what
would make that tractable on graphs, and it is the main payoff of keeping energies as vectors.

## Leads

The Bethe Hessian; linear response for belief propagation (Welling & Teh); the presentation
dependence found by Karwowski et al., which suggests the answer depends on how factors are grouped.
