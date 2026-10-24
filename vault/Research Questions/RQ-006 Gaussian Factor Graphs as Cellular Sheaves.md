#open-problem

> **Status: open (2026-10-24).** Area: applied topology, numerical linear algebra · Prerequisites: cellular sheaves, Gaussian belief propagation · Related: [[LP-012 Factor graphs and schemes]], [[Loopy Message Passing]], [[The Linear Gaussian Chain]].
>
> Sources: Hansen & Ghrist, *Toward a Spectral Theory of Cellular Sheaves*, J. Appl. Comput. Topol. 3, 2019; Malioutov, Johnson & Willsky, *Walk-Sums and Belief Propagation in Gaussian Graphical Models*, JMLR 7, 2006; the question is original to this vault.
>
> Bibliography: [[Bibliography#^hansen2019sheaves|Hansen & Ghrist 2019]] · [[Bibliography#^malioutov2006walksums|Malioutov et al. 2006]]

## Question

A linear-Gaussian factor graph can be read as a cellular sheaf: each variable and factor carries
a vector space, and each edge a linear map saying how they must agree. **What do the sheaf's
Laplacian and cohomology say about Gaussian belief propagation, in particular about its fixed points
and about the variances that loopy propagation gets wrong?**

## Setting

Variables $v$ with stalks $\mathbb R^{d_v}$; factors $a$ with stalks $\mathbb R^{m_a}$ and the factor
$\exp(-\tfrac12 \lVert \sum_{v \in a} A_{a,v} z_v - c_a \rVert^2_{Q_a^{-1}})$. The restriction maps
are $A_{a,v} : \mathbb R^{d_v} \to \mathbb R^{m_a}$, the coboundary is
$(\delta z)_a = \sum_{v \in a} A_{a,v} z_v$, and the weighted sheaf Laplacian is
$L = \delta^\top Q^{-1} \delta$. The joint precision of the graph (with priors) is $L$ plus the prior
precisions.

## What is known

- The MAP estimate (equal to the posterior mean) minimises the weighted Dirichlet energy
  $\tfrac12\lVert\delta z - c\rVert^2_{Q^{-1}}$ plus the priors: a sheaf-harmonic extension problem.
  This much is a restatement, not a result.
- Gaussian belief propagation computes exact means on loopy graphs when it converges (walk-summable
  models), but its variances are wrong in general ([[Loopy Message Passing]]).
- Global sections of the sheaf ($\delta z = 0$) are the configurations that satisfy every factor
  exactly (Hansen & Ghrist 2019).

## What would count as an answer

A statement relating the variance error of loopy Gaussian belief propagation to a sheaf-theoretic
quantity (the first cohomology, the spectrum of $L$, or the cycle structure seen by the sheaf), with
a proof or a counterexample; and a check on a small loopy graph with exact variances.

## Why it matters here

It would give the CT-ML wiki a *Cellular Sheaf* note with a concrete payoff, and Lenticulum a
diagnostic for when loopy variances can be trusted.

## Leads

Walk-sum analysis of Gaussian BP (Malioutov, Johnson & Willsky); sheaf Laplacians and diffusion
(Hansen & Ghrist); Robinson's *Topological Signal Processing*.
