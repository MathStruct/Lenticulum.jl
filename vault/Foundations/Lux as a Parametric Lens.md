#definition #design

Lux.jl was not designed from category theory, but it converged almost exactly on the structure of [parametric lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) — morphisms of $\mathbf{Para}(\mathbf{Lens}(\mathbf{Smooth}))$, with the backward pass supplied by automatic differentiation as a [reverse derivative](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category). Seeing that makes both easier to understand, and makes the *departure* Lenticulum takes precise.

> Sources: Cruttwell, Gavranović, Ghani, Wilson & Zanasi, *Categorical Foundations of Gradient-Based Learning* (arXiv:2103.01931) Definitions 2.1–2.5, 3.3, 3.8, 3.11, 3.14, Proposition 2.7; Lux.jl / LuxCore.jl documentation; code: `lib/LenticulumCore.jl` ([[LenticulumCore]]).
>
> Bibliography: [[Bibliography#^cruttwell2022gradient|Cruttwell et al. 2022]]
>
> Theory (CT-ML wiki): [Para Construction](https://mathstruct.org/CategoryTheory-ML-Wiki/Para-Construction) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category) · [Gradient-Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses) · [paper note](https://mathstruct.org/CategoryTheory-ML-Wiki/Papers/Categorical-Foundations-of-Gradient-Based-Learning)

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

AutoBayes escapes via [compact closure](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model#copiers-cups-and-caps-remark-8): a cup lets you bend an
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

## What Lenticulum keeps from the categorical picture, and what it adds

- **Parameters are $\mathbf{Para}$.** A factor carries `ps` separately from its inputs, and `initialparameters` returns a nested `NamedTuple` mirroring the factor tree — the $\mathbf{Para}$ composite $Q \otimes P$, with labels, which makes it order-insensitive. Optimisers, weight tying and LoRA are *reparametrisations*, the 2-cells of $\mathbf{Para}$; Lenticulum turns the optimiser into a node of the graph ([[Everything is a Factor]]).
- **Lenses cannot drop a backward wire.** $\mathbf{Lens}(\mathcal C)$ is monoidal but not cartesian: there is no unique put into $(1, 1)$. In a factor graph this is the reason a loss must be an explicit *sink* (the learning-rate cap of Cruttwell et al., Definition 3.8), and why `LossFactor` supports no polarity.
- **$(P, P')$ are different spaces.** The update to a parameter lives in a (co)tangent space, not in the parameter space; turning it into a step needs a metric. Lenticulum keeps the analogous distinction on the loss side: the **energy space** $E$ of a factor is a separate object from its value spaces, which is what makes Gauss–Newton and Fisher metrics available ([[Scalar and Multivariate Energy]]).
- **Autodiff is one functorial backward pass among three.** Reverse differentiation is a functor $R : \mathcal C \to \mathbf{Lens}(\mathcal C)$ (Proposition 2.7), and it is why a Lux model's gradient is correct. Lenticulum needs two more, both with their own chain rules:

| backward pass | forward | backward | chain rule |
|---|---|---|---|
| autodiff | $f : A \to B$ | $R[f] : A \times B \to A$ | Cruttwell et al. Proposition 2.7 |
| Bayesian inversion | $c : X \rightsquigarrow [\![c]\!] \times Y$ | $c'_\pi : Y \rightsquigarrow X \times [\![c]\!]$ | AutoBayes Theorem 13 ([[Inversions and Bayesian Lenses]]) |
| free energy | $l^c$ | $H^c$ | AutoBayes Theorem 23 ([[Factors are Parameterized Statistical Games]]) |

They stack: a factor's *inversion* is often itself a Lux network trained by autodiff, so the reverse-derivative layer sits inside the statistical-game layer.

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

Related: [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens), [Learning Components as Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses), [[AutoBayes to Lenticulum]], [[Channels and Polarity]]
