#definition #category-theory #computable #catlab

A **natural transformation** $\alpha: F \Rightarrow G$ between [[functor|functors]] $F, G: \mathcal{C} \to \mathcal{D}$ is a family of [[morphism|morphisms]] $\alpha_X: F(X) \to G(X)$ indexed by [[Object|objects]] $X$ in $\mathcal{C}$ such that the [[naturality condition]] holds.

## Naturality Condition

For every [[morphism]] $f: X \to Y$ in $\mathcal{C}$, the following [[commutative diagram]] holds:

G(f)∘αX=αY∘F(f)G(f)∘αX​=αY​∘F(f)

<!-- \begin{tikzcd} F(X) \arrow[r, "\alpha_X"] \arrow[d, "F(f)"'] & G(X) \arrow[d, "G(f)"] \\ F(Y) \arrow[r, "\alpha_Y"'] & G(Y) \end{tikzcd} -->

## Components

- **Components**: The [[morphism|morphisms]] $\alpha_X: F(X) \to G(X)$ are called the **components** of the natural transformation
- **Naturality Square**: Each commutative square witnessing naturality

## Composition

Natural transformations compose **vertically**: if $\alpha: F \Rightarrow G$ and $\beta: G \Rightarrow H$, then $\beta \circ \alpha: F \Rightarrow H$ with components $(\beta \circ \alpha)_X = \beta_X \circ \alpha_X$.

Natural transformations form the [[2-morphism|2-morphisms]] in the [[2-category]] [[Cat]].

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/categorical_algebra/)