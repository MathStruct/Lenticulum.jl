#definition #category-theory #computable #catlab

A **morphism** (or **arrow**) is a directed relationship between [[object|objects]] in a [[category]]. Morphisms encode structure-preserving mappings and are the fundamental building blocks of categorical reasoning.

## Formal Definition

A morphism $f: A \to B$ in a [[category]] $\mathcal{C}$ is an element of the [[hom-set]] $\text{Hom}_{\mathcal{C}}(A, B)$ with:

- **Domain**: The [[object]] $A$ (source)
- **Codomain**: The [[object]] $B$ (target)

## Composition

Morphisms compose [[associativity|associatively]]: if $f: A \to B$ and $g: B \to C$, then their [[composition]] $g \circ f: A \to C$ satisfies: $$(h \circ g) \circ f = h \circ (g \circ f)$$

## Identity

Each [[object]] $A$ has an [[identity morphism]] $\text{id}_A: A \to A$ such that:

- $f \circ \text{id}_A = f$ for any $f: A \to B$
- $\text{id}_B \circ f = f$ for any $f: A to B$

## Examples

- In [[Set]]: functions between sets
- In [[Grp]]: group homomorphisms
- In [[Top]]: continuous maps
- In [[Vec]]: linear transformations

<!-- \begin{tikzcd} A \arrow[r, "f"] & B \arrow[r, "g"] & C \end{tikzcd} Composition: $g \circ f: A \to C$ -->

Morphisms embody the principle that [[category theory]] studies relationships and transformations rather than internal structure of objects.

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/categorical_algebra/)