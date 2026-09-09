# AutoBayes → Lenticulum.jl — the naming dictionary

> One table per layer. Use this when reading the paper with the code open.

## Paper concept → Julia name

| AutoBayes | Lenticulum | file |
|---|---|---|
| open model $c : X \nrightarrow\!\!\!\bullet\; Y$ | `AbstractOpenModel` | `open_model.jl` |
| latent space $\llbracket c \rrbracket$ | the `latent` field of a forward result | `open_model.jl` |
| $\mathcal{P}X$, a prior | `AbstractBelief` | `open_model.jl` |
| pushforward $c_*\pi$ | `pushforward(model, π, ps, st)` | `open_model.jl` |
| Bayesian lens $(c, c')$ | `AbstractBayesianLens` | `lens.jl` |
| inversion $c'_\pi$ | `invert(lens, π, y, ps, st)` | `lens.jl` |
| exact inversion $c^\dagger_\pi$ | `ExactInversion` | `lens.jl` |
| energy $l^c$ (scalar) | `scalar_energy(f, ...)` | `energy.jl` |
| **vector energy $\mathbf{l}^c$** | `energy(f, ...)` | `energy.jl` |
| entropy $H^c$ | `entropy(f, π, y, ...)` | `energy.jl` |
| free energy $F^c$ | `free_energy(f, π, y, ...)` | `energy.jl` |
| statistical game $(c,c',l,H)$ | `AbstractLenticulumFactor` | `statistical_game.jl` |
| parameterized game $(\Theta, c)$ | a factor + its `ps` from `initialparameters` | `statistical_game.jl` |
| composition $d \diamond c$ | `compose`; or an edge in a `FactorGraph` | `statistical_game.jl`, Mycelium `graph.jl` |
| tensor $c \otimes d$ | parallel branches in the graph | Mycelium `graph.jl` |
| cup / clamp | `DataFactor` on an `Emitting` edge | Mycelium `factors.jl` |
| prior $\pi : 1 \multimap X$ (Remark 24) | `PriorFactor` | Mycelium `factors.jl` |
| copier | a variable node of degree > 2 | Mycelium `graph.jl` |
| inversion $c'_\pi$ | a factor → variable message | Mycelium `passing.jl` |
| pushforward $c_*\pi$ | a variable → factor message | Mycelium `messages.jl` |
| Theorem 23 on a graph | the Bethe free energy | Mycelium `free_energy.jl` |
| optimiser (Cruttwell §3.4) | `OptimiserFactor` on an exposed parameter variable | Mycelium `factors.jl` |
| loss + learning-rate cap (§3.2–3.3) | `LossFactor` (a **sink**) | Mycelium `factors.jl` |
| $X$ unobserved / $Y$ observed / $\llbracket c\rrbracket$ latent | `Unobserved()` / `Observed()` / `Latent()` | `channels.jl` |
| Definition 29 gradient composition | `GradientCoupling` per edge | `statistical_game.jl` |
| "different semantics functors" | the `GradientCoupling` variants | `statistical_game.jl` |

## The three-layer stack

```
┌─────────────────────────────────────────────────────────────┐
│ Lenticulum.jl            — user-facing, mirrors Lux.jl      │
│   concrete factors, training loops, data handling           │
├─────────────────────────────────────────────────────────────┤
│ Mycelium.jl              — factor graphs, message passing   │
│   scheduling, polarity resolution, Bethe free energy        │
├─────────────────────────────────────────────────────────────┤
│ LenticulumCore.jl        — mirrors LuxCore.jl               │
│   abstract types, the factor interface, energy algebra      │
└─────────────────────────────────────────────────────────────┘
        VariationalDiffusion.jl plugs in at the factor level
```

## The one-sentence differences from Lux.jl

Restating [[README]] precisely now that the vocabulary exists:

1. **Lux layers are [[Parametric Lens|parametric lenses]]** (Cruttwell et al. Def 2.5):
   `get` + `put`, fixed at construction. **Lenticulum factors are
   [[Parameterized Statistical Game|parameterized statistical games]]** (AutoBayes Def 27):
   a lens *plus* an energy *plus* an entropy, and the lens is **assembled on demand** once
   a [[Channels and Polarity|polarity]] is chosen. Hence
   $$\mathbf{Para}(\mathbf{Lens}(\mathcal{C})) \quad\text{vs.}\quad \mathbf{Para}(\mathbf{StatGame})$$
2. **Lux wiring is a DAG**; Lenticulum's is any weakly-connected digraph, licensed by
   [[Copiers Cups and Caps|compact closure]]. Factors still use Lux internally.
3. **Message passing is trivial in Lux** (there is one order). In Lenticulum it must be
   scheduled, and dense all-pairs passing should be sparsified for performance. This is
   also what the paper's closing discussion asks for: belief propagation / variational
   message passing to approximate the pushforward priors.

## Reading order for the code

1. `channels.jl` — what a factor's ports are, and what polarity means.
2. `energy.jl` — the two energies and the scalarisation algebra.
3. `open_model.jl` — kernels with latent spaces, pushforward.
4. `lens.jl` — pairing a model with an inversion.
5. `statistical_game.jl` — the factor interface, tying it together.

Then Mycelium, in this order:

6. `graph.jl` — the bipartite structure and the two acyclicity notions ([[Factor Graphs]]).
7. `polarity_resolution.jl` — which way to run a factor ([[Polarity Resolution]]).
8. `passing.jl` — the five lines where a message becomes an inversion ([[Messages are Inversions]]).
9. `free_energy.jl` — Theorem 23 on a graph ([[Bethe Free Energy]]).
10. `factors.jl` — data, losses and optimisers as nodes ([[Everything is a Factor]]).

Related: [[Lux as a Parametric Lens]], [[Parameterized Statistical Game]], [[Scalar and Multivariate Energy]], [[LenticulumCore]]
