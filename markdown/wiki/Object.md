#definition #category-theory #computable #catlab

An **object** is a primitive constituent of a [[category]]. Objects are the basic entities between which [[morphism|morphisms]] exist.

## Formal Characterization

In a [[category]] $\mathcal{C}$, objects are the elements of the object class $\text{Ob}(\mathcal{C})$. They have no internal structure in categorical terms - their properties are entirely determined by the [[morphism|morphisms]] between them.

## Properties

- Objects are connected by [[morphism|morphisms]]
- Each object $A$ has an [[identity morphism]] $\text{id}_A: A \to A$
- Objects themselves have no categorical structure beyond their morphisms

## Notation

Objects are typically denoted by capital letters: $A, B, C, X, Y, Z$.

## Examples

- In [[Set]]: sets are objects
- In [[Grp]]: groups are objects
- In [[Top]]: topological spaces are objects
- In [[Vec]]: vector spaces are objects

## Categorical Principle

Objects are characterized entirely by their relationships ([[morphism|morphisms]]) to other objects, embodying the principle that [[category theory]] studies structure rather than internal composition.

<!-- \begin{tikzcd} A \arrow[r, "f"] \arrow[loop left, "\text{id}_A"] & B \arrow[loop right, "\text{id}_B"] \end{tikzcd} -->

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/categorical_algebra/)