#definition #category-theory #computable #catlab

A **commutative diagram** in a [[category]] is a [[diagram]] where all [[morphism|morphisms]] compose to give the same result along any path between the same pair of [[Object|objects]].

## Formal Definition

A [[diagram]] $D: \mathcal{J} \to \mathcal{C}$ from an [[index category]] $\mathcal{J}$ to a category $\mathcal{C}$ is **commutative** if for any two [[morphism|morphisms]] $f, g: j \to k$ in $\mathcal{J}$, we have $D(f) = D(g)$ in $\mathcal{C}$.

## Equivalently

For any two composable sequences of morphisms $$X \xrightarrow{f_1} Y_1 \xrightarrow{f_2} \cdots \xrightarrow{f_n} Z$$ $$X \xrightarrow{g_1} Y_1' \xrightarrow{g_2} \cdots \xrightarrow{g_m} Z$$ we have $f_n \circ \cdots \circ f_2 \circ f_1 = g_m \circ \cdots \circ g_2 \circ g_1$.

## Basic Example

In a commutative square:

<!-- \begin{tikzcd} A \arrow[r, "f"] \arrow[d, "g"'] & B \arrow[d, "h"] \\ C \arrow[r, "k"'] & D \end{tikzcd} -->

The diagram commutes if and only if $h \circ f = k \circ g$.

## Properties

- **[[Composition]]**: Commutative diagrams can be pasted together
- **[[Identity morphism|Identities]]**: Adding identity morphisms preserves commutativity
- **[[Universality]]**: Many [[universal property|universal properties]] are expressed via commutative diagrams

Commutative diagrams are fundamental for expressing mathematical relationships and defining [[universal construction|universal constructions]] in [[category theory]].

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/categorical_algebra/)