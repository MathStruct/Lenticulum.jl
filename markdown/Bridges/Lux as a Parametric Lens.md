# Lux.jl as a Parametric Lens

> Motivating [[Parametric Lens]] by pointing at code you already use.

Lux.jl was not designed from category theory, but it converged on the parametric-lens
structure almost exactly. Seeing that makes both easier to understand — and makes the
*departure* Lenticulum takes precise.

## The dictionary

| $\mathbf{Para}(\mathbf{Lens}(\mathbf{Smooth}))$ | Lux.jl / LuxCore.jl |
|---|---|
| object $(A, A')$ | an array shape, together with its cotangent shape |
| parameter object $P$ | `ps`, a nested `NamedTuple` |
| parameter change object $P'$ | the `NamedTuple` returned by `Zygote.gradient` — same tree, cotangent leaves |
| morphism $(P, (f, f^*))$ | an `AbstractLuxLayer` |
| $f : P \times A \to B$ | `layer(x, ps, st)` — the `get` |
| $f^* : P \times A \times B' \to P' \times A'$ | the `pullback` from `Zygote.pullback(...)` — the `put` |
| $\mathbf{Para}$ composition, $P' \otimes P$ | `Chain`, whose `ps` is `(layer_1 = ..., layer_2 = ...)` |
| the functor $R : \mathcal{C} \to \mathbf{Lens}(\mathcal{C})$ | the AD backend (Zygote / Enzyme / Mooncake) |
| reparametrisation $\alpha : Q \to P$ | `Optimisers.jl` rules, weight tying, LoRA |
| stateful update $U : (S\times P, S\times P) \to (P,P')$ | `Optimisers.setup` / `Optimisers.update` — `S` is `opt_state` |
| loss map $(\mathrm{loss}, B) : B \to L$ | `MSELoss()`, `CrossEntropyLoss()` from `LossFunctions` |
| learning rate cap $\alpha^* : L \to L'$ | the implicit `one(L)` seed handed to `pullback`, times $-\eta$ |

## The two things Lux has that the paper does not name

**1. `st` — state.** LuxCore threads a state `NamedTuple` through: `(y, st) = layer(x, ps, st)`.
Category-theoretically this is a second parameter wire that is *written back on the forward
pass* rather than the backward one. `BatchNorm`'s running statistics, an RNG, a dropout
mask. Definition 2.5 has no slot for it; you can model it as a parameter $P$ whose update
comes from the `get` rather than the `put`, which is a small generalisation.

**2. `initialparameters(rng, layer)`.** The paper takes $P$ as given. Lux takes seriously
that $P$ must be *constructed*, randomly, from a description. The layer is a *description*;
`ps` is an *inhabitant*. That separation is the reason Lux layers are immutable structs and
is worth preserving verbatim in Lenticulum.

## The two constraints Lenticulum must break

### Constraint 1: the wiring must be a DAG

`Chain` is sequential; `Parallel`, `BranchLayer` and `SkipConnection` still only build DAGs.
This is forced: $\mathbf{Lens}(\mathcal{C})$ composition is a *function composition*, and
function composition of a cycle does not terminate.

AutoBayes escapes via [[Copiers Cups and Caps|compact closure]]: a cup lets you bend an
input wire into an output wire, so a cycle becomes a straight line with a bent end.
Lenticulum inherits arbitrary weakly-connected digraphs from this.

### Constraint 2: `get` and `put` are fixed at construction

A Lux layer *knows* which side is input. `Dense(3 => 5)` cannot be run backwards. But a
Lenticulum factor is a relation, and which channels are input is a *call-site* decision.
So the factor cannot store $(f, f^*)$; it must be able to **assemble** an $(f, f^*)$ on
demand, once a polarity is chosen:

$$\texttt{assemble}(\texttt{factor},\ \texttt{polarity}) \;\longmapsto\; \text{a parametric lens}$$

That is exactly [[README]]'s "our program must extract or assemble parametric Lenses before
they can be used", and it is why `LenticulumCore` cannot simply subtype `AbstractLuxLayer`.
See [[Channels and Polarity]].

## What is kept unchanged

Everything else. `initialparameters` / `initialstates` / `setup` / `apply` /
`parameterlength` / `statelength` / `display_name` / the `AbstractLuxContainerLayer{layers}`
and `AbstractLuxWrapperLayer{layer}` trick for deriving parameter trees from field names —
all of it applies verbatim to factors. `LenticulumCore` mirrors the LuxCore API name for
name, and a factor's *internals* are Lux layers.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[row sep=large, column sep=huge]
\mathbf{Para}(\mathcal{C}) \arrow[r, "\mathbf{Para}(R)"] \arrow[d, "\text{decorate}"'] & \mathbf{Para}(\mathbf{Lens}(\mathcal{C})) \arrow[d, "\text{decorate}"] \\
\mathbf{Para}(\mathbf{OpenModel}) \arrow[r, "\mathbf{Para}((-)^\dagger)"'] & \mathbf{Para}(\mathbf{StatGame})
\end{tikzcd}
\end{document}
```

Top row: Lux.jl. Bottom row: Lenticulum.jl. Same shape, different backward pass.

Related: [[Parametric Lens]], [[Learning Components as Parametric Lenses]], [[AutoBayes to Lenticulum]], [[Channels and Polarity]]
