#overview #open-problem

> Open research questions that Lenticulum raises, each stated so that it can be worked on
> without reading the rest of the vault: the setting, the question, what is known, and what
> would count as an answer. Numbered `RQ-nnn`; the template is
> `vault/templates/Research Question Template.md`.

This is not the [[Problem Log]]. A problem (`LP-nnn`) is something in the code or the project that
needs a decision or an implementation. A research question is mathematics or statistics whose
answer nobody here knows, and which could be a thesis chapter, a paper, or a long afternoon.
The family notes keep their own ledgers of open points; the ones that are self-contained research
questions are lifted here and linked back:
[[Open Problems in Implicit Diffusion Learning]], [[Open Problems in Algebraic Implicit Learning]].

A question gets an answer by a proof, a counterexample, or a measurement, recorded in its note
with its source; then its status changes and it stays listed.

| # | Question | Area | Status |
|---|---|---|---|
| [[RQ-001 One Potential, Every Polarity]] | When does one smoothed data density give isolated, regular answers for every query direction, including directions with more free coordinates than the data's codimension? | analysis, geometry of densities | open |
| [[RQ-002 Shift Deconvolution with Three or More Noise Draws]] | Does $\sum_j \gamma_j H(u + s_j) = \hat G(u)$ have a solution of polynomial size for $q \ge 3$ shifts? | analysis | open |
| [[RQ-003 When Local Gradients Are Exact]] | When do the gradient terms AutoBayes drops vanish, and how biased is local-gradient descent when they do not? | variational inference | open |
| [[RQ-004 Is the Bethe Criterion Consistent for Structure Learning]] | Does adding factors that lower the Bethe free energy recover the true graph? | statistics | open |
| [[RQ-005 A Chain Rule for the Fisher Metric on Factor Graphs]] | Is there a local, message-based Fisher metric on loopy factor graphs? | information geometry | open |
| [[RQ-006 Gaussian Factor Graphs as Cellular Sheaves]] | What do sheaf Laplacians and cohomology say about loopy Gaussian belief propagation? | applied topology | open |
| [[RQ-007 Risk Scalarisations and Local Decomposition]] | For which risk measures (entropic risk, CVaR) of a factor graph's total energy is there a local, message-passing computation or bound? | variational inference, risk measures | open |
