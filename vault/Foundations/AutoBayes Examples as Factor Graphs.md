#example #design

The five worked examples of the AutoBayes appendix, read as **wirings of factors** rather than as derivations. Each is a sanity check for the claim that Lenticulum's graph-level vocabulary — priors, clamps, exposed parameters — covers the standard algorithms without special machinery.

> Sources: *AutoBayes* (arXiv:2503.18608) Appendix A (Examples 1–5), Appendix B.
>
> Bibliography: [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) (the examples table) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) (cups) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Example 1 — mixture model: maximum likelihood is "all entropies zero"

A Gaussian game $c : M \multimap Y$ with exact inversion after a prior game on the mixture component, parameterized by the mixing weights $\alpha$. Descending $F(\ast, y; \alpha)$ is maximum likelihood. As factors: a `PriorFactor` with learnable weights feeding a Gaussian factor, the observation clamped by a `DataFactor`.

## Example 2 — EM is the two halves of the framework

Lens $c$ and prior lens $\pi$, NLL energies, **zero entropies**: $F^{c\pi}(\ast, y) = \mathbb E_{x \sim c'_\pi(y)}[-\log p(x, y)]$. Evaluating it is the **E-step**; descending it in the parameters is the **M-step**. In a graph: run inference (a schedule) with the parameters fixed, then take an optimiser step with the messages fixed — coordinate descent on one objective, which is why it fits ([[The Two-Part Diagram]]).

## Example 3 — VBEM: a parameter you want a posterior over is a variable

Games $c : \Theta \otimes X \multimap Y$ and $\pi : \Theta \multimap X$ with a hyperprior on $\Theta$. The move to notice: $\Theta$ **migrates from the parameter space into the wire**. In Lenticulum that is a graph-level decision: the same factor can keep its parameters hidden in `ps` (maximum likelihood) or **expose** them as a variable with a prior or an optimiser attached (a posterior over weights). The difference is one edge — see [[Everything is a Factor]].

## Example 4 — supervised learning: Lenticulum reduces to Lux when every wire is clamped

Compose $X \otimes c$ after a **cup** on $X$: inputs and outputs are both observed. The inversion trivialises (the "prior" is a deterministic sample), the regulariser may not, and the loss depends only on parameters and paired data. As factors: `DataFactor`s on both ends, a `LossFactor` sink. The graph is a tree but not a DAG, and `forward_backward_schedule` on its DAG part is a Lux forward pass whose backward *belief* sweep is empty — the backward pass carries cotangents ([[Schedules]]).

## Example 5 — Bayesian deep learning

Cup on $X$ only, a prior game $1 \multimap \Theta$ on the weights, a mean-field inversion over $\Theta$ and $X$: the cup trivialises the $X$ factor and leaves a posterior over weights. In Lenticulum: an exposed weight variable with a `PriorFactor`, and an inversion over it — the same network as Example 4 with one more edge.

## Appendix B — dependent types

A joint on $\sum_x Y_x$ rather than $X \times Y$, a conditional as a stochastic section. See [[Open Models and Latent Channels]] §"Dependent types".

## What the examples show

| example | graph-level vocabulary used |
|---|---|
| 1 | learnable prior, data clamp |
| 2 | inference schedule + optimiser step on one free energy |
| 3 | parameters exposed as a variable, hyperprior |
| 4 | clamps on both ends, loss sink |
| 5 | exposed weights with a prior, mean-field inversion |

No example needs a new node type; each is a choice of which wires are clamped, exposed or latent. That is the design claim of [[Everything is a Factor]], checked against the paper's own examples.
