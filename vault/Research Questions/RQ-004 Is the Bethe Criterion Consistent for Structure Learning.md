#open-problem

> **Status: open (2026-10-24).** Area: statistics, graphical models · Prerequisites: Gaussian graphical models, Bethe free energy, model selection · Related: [[Dynamic Graphs]] §4, [[Bethe Free Energy]], [[LP-010 Where the graph of a problem comes from]].
>
> Sources: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 2005; Malioutov, Johnson & Willsky, *Walk-Sums and Belief Propagation in Gaussian Graphical Models*, JMLR 7, 2006; the question is original to this vault.
>
> Bibliography: [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^malioutov2006walksums|Malioutov et al. 2006]]

## Question

Lenticulum's design adds a factor to a graph when doing so lowers the Bethe free energy of the
data ([[Dynamic Graphs]] §4). Used repeatedly, that is a score-based structure-learning method.
**Is it consistent: with enough data, does it recover the true graph?**

## Setting

Data $z_1, \dots, z_N$ i.i.d. from a Gaussian graphical model on $n$ variables with graph $G^\ast$.
A candidate graph $G$ is scored by the scalar Bethe free energy $F_{\mathrm{Bethe}}(G)$ of the data
after message passing, with the factor parameters fitted (by maximum likelihood, or integrated out
under a prior). The greedy procedure adds the edge (factor) whose addition lowers the score most,
while any addition lowers it.

## What is known

- On a tree with exact messages, the Bethe free energy is the variational free energy, and with
  parameters integrated out it is the negative log marginal likelihood; Bayesian model selection by
  marginal likelihood is consistent for Gaussian graphical models under standard conditions.
- On a loopy graph the Bethe free energy is an approximation, and loopy Gaussian belief propagation
  can underestimate variances. Whether the approximation error can change the ranking of graphs as
  $N \to \infty$ is not known here.
- With parameters fitted rather than integrated out, the score has no complexity penalty and always
  prefers the complete graph; a penalty (BIC-like) or a parameter prior is needed.

## What would count as an answer

A consistency theorem for the loopy case, under conditions (walk-summability, bounded degree), or
a counterexample in which the Bethe score prefers a wrong graph for all large $N$; and a simulation
protocol comparing it with the graphical lasso and with BIC on known graphs.

## Why it matters here

It decides whether the Bethe criterion can be the structure-learning step of
[[LP-010 Where the graph of a problem comes from]], and which complexity term it needs.

## Leads

Asymptotics of the Bethe approximation for walk-summable Gaussian models; the relation between
Bethe free energy and pseudo-likelihood; existing results on Bethe-based model selection in
discrete models.
