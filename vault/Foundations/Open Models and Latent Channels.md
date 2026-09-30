#definition #design #implementation

Lenticulum's forward models are AutoBayes **open models** $c : X \rightsquigarrow [\![c]\!] \times Y$: kernels with an explicit **latent space** that holds what composition has hidden ([Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model)). Composing open models **never integrates**: the intermediate variable is filed into the latent space instead of marginalised. In implementation terms the latent space **is the activation cache** — the same phenomenon as reverse-mode autodiff storing intermediate activations, under the same chain rule.

The three spaces of an open model are the three channel polarities of a factor:

| AutoBayes | polarity | role |
|---|---|---|
| $X$, unobserved | `Unobserved()` | solved for; the posterior is over it |
| $Y$, observed | `Observed()` | clamped to data or to an incoming message |
| $[\![c]\!]$, latent | `Latent()` | internal; revealed or marginalised |

(the crossing-over of "observed/input" and "unobserved/output" names is discussed in [[Channels and Polarity]]).

> Sources: *AutoBayes* (arXiv:2503.18608) Definitions 1–8, Remarks 2–8 (and Fong 2013, Theorem 4.5, for Bayesian networks); code: `lib/LenticulumCore.jl/src/open_model.jl` ([[open_model]]), `channels.jl` ([[channels]]), `lib/Mycelium.jl/src/graph.jl` ([[graph]]).
>
> Theory (CT-ML wiki): [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Compact Closed Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Compact-Closed-Category) · [Hypergraph Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Hypergraph-Category) · [Para Construction (CoPara)](https://mathstruct.org/CategoryTheory-ML-Wiki/Para-Construction) · [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category)

## The latent space is state, not output

A factor's forward pass returns the observed part and the latent part **separately** — `forward(model, x, ps, st)` gives an `OpenModelResult` with a `latent` field — and the inversion consumes the latent part. `latentspace`, `observedspace` and `unobservedspace` expose the three spaces; `ispure` marks $[\![c]\!] \cong 1$; `pushforward(model, π, ps, st)` is $c_*\pi$. The price is the paper's trade: memory for tractability — the latent space of a deep composite is the product of all intermediate spaces.

## Reveal and dummy variables

- **reveal** (a 2-cell): move a latent factor into the observed space. It is a retyping, not a computation — in Lenticulum, exposing an intermediate as an observable channel, for probing, debugging or an auxiliary loss.
- **dummy variables** ($A \otimes q$): let information flow past a factor untouched — a skip connection, or "this factor does not depend on that variable".

## From a graph to a string of open models — and where that stops

Every Bayesian network is a composite of open models: sort topologically, reveal each node's parents, pad with dummies, compose in order (the paper, after Fong 2013). That is a concrete compilation procedure from an **acyclic** factor graph to a sequence of composable models. For **cyclic** graphs it does not apply; one needs cups/caps and a message-passing schedule instead. That split — acyclic ⇒ compile to a sequence, cyclic ⇒ schedule messages — is the central design fork of Mycelium ([[Factor Graphs]], [[Schedules]]).

## Copiers, cups, caps — and why that is not yet enough

With a copier, a cup $\mathrm{cup}_A : 1 \nrightarrow\!\!\!\bullet\ A \otimes A$ and an (unnormalised) cap, open models form a **self-dual compact closed** bicategory (Remark 8): a cup bends an unobserved leg into an observed one. In Lenticulum:

- a **copier** is a variable node of degree $> 2$ — no node type is needed;
- a **cup** is a `DataFactor` on an `Emitting` edge — "clamp this variable to data" — and softening the clamp (finite precision) changes a node's type, not the graph;
- bending wires is what lets a factor have **no fixed direction**: the lens exists only once a polarity is chosen ([[Channels and Polarity]]).

Compact closure bends *one* wire at a time. A factor-graph variable of degree three is a three-way **merge**, which needs the stronger hypergraph structure — see [[Acausal Composition is a Hypergraph Category]]. And unnormalised caps hand back a partition-function problem: cycles buy expressiveness and cost normalisation, the same trade as energy-based versus normalised models ([[Energy-Based Learning]]).

## Dependent types (Appendix B)

In a dependently typed model the observed space varies with the unobserved one — a joint lives on $\sum_{x} Y_x$ and a conditional is a stochastic section of $\sum_x Y_x \to X$ (weather reports with different fields at sea and on land). Julia's parametric types and dispatch can express channel types that depend on a value; not needed for v0, but a genuine advantage over a Python implementation. The categorical version is the *dependent Bayesian lens* ([Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)).

````tabs
tab: Julia
**Docs:** [LenticulumCore: open models, channels](https://mathstruct.org/Lenticulum.jl/dev/packages/lenticulumcore/)
```julia
using Lenticulum, LenticulumCore
f = GaussianFactor(1 => 1; noise = 0.25, channels = (:x, :y))
[channelname(c) for c in channels(f)]              # [:x, :y]
# a polarity is a partition of the channels into observed / unobserved / latent
p = Polarity(; x = Observed(), y = Unobserved())
observed_channels(p), unobserved_channels(p)       # ((:x,), (:y,))
ispartition(p)                                     # true
```
````
