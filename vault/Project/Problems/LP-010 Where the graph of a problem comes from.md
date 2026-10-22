#open-problem

> **Status: open, theory (2026-10-22, Daniel: "How do we actually find out the graph for a problem? That looks to me a lot like topological data analysis. And should be done before.").** Related: [[Dynamic Graphs]] §4, [[Bethe Free Energy]], [[Loopy Message Passing]], [[Motivating Examples]], [[ModelingToolkit as an Acausal Relation]], [[RxInfer as a Backend]].

**Problem.** Everything in the vault starts from a given factor graph. For a new problem, someone has to find it first: which variables exist, which of them interact, and through which factors. The vault does not say how.

**Routes, from most to least prior knowledge.**
1. *From a model.* Equations (ModelingToolkit) or a probabilistic program (an RxInfer `@model`) already are the graph; most of the [[Motivating Examples]] are of this kind (circuits, SLAM, power grids).
2. *Structure learning from data.* Which variables depend on which: conditional-independence tests (the PC algorithm), score-based search (GES, NOTEARS), the graphical lasso for Gaussian models. Lenticulum's own score is already designed: the Bethe criterion of [[Dynamic Graphs]] §4 compares the free energy with and without a candidate factor, which is score-based structure search one factor at a time.
3. *Topological data analysis.* Mapper draws a graph of the data's shape, and persistent homology finds its clusters and loops. A Mapper graph's nodes are regions of the data, not variables, so it is not a factor graph. TDA is exploration before route 2: which regimes exist, whether one graph fits all of them. It also applies to a factor graph itself, whose cycles decide where loopy message passing is only approximate ([[Loopy Message Passing]]).

**Observation.** Lenticulum has a fourth route the others lack: learn **one relation over the joint space** (a diffusion factor over all variables), then split it. Splitting a joint factor into smaller ones whose product approximates it is structure learning by factorisation, and the free energy measures what the split costs. This is an expectation, not something the code does.

**Leaning.** "Should be done before" is right as an order: explore (TDA) → propose a structure (model, or structure learning) → fit (Lenticulum) → revise with the Bethe criterion. A note in `Factor Graphs/` on this pipeline, with one worked example, is the next step. Moonkale's part would be showing each step's result as a graph to accept or edit.
