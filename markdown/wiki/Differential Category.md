#definition #category-theory #differential #linear-logic #computable

A **differential category** is a [[category]] equipped with a differential operator that axiomatizes the categorical structure underlying differentiation and provides semantics for [[differential linear logic]].

## Structure

A differential category (due to Blute, Cockett & Seely) consists of:

- A [[cartesian category]] with [[coproduct|coproducts]]
- A **deriving transformation** $D: \mathcal{C}(A \times B, C) \to \mathcal{C}(A \times B \times B, C)$
- Satisfying differential coherence axioms

## Differential Operator

For a [[morphism]] $f: A \times B \to C$, the differential $D[f]: A \times B \times B \to C$ captures the directional derivative of $f$ in the direction of the third component.

## Axioms

The differential operator must satisfy:

- **[[Chain rule]]**: $D[g \circ (f \times \text{id})] = D[g] \circ (f \times \text{id} \times \text{id}) + g \circ (D[f] \times \text{id})$
- **[[Leibniz rule]]**: For products of functions
- **[[Linearity]]**: $D$ is linear in its argument
- **[[Constant rule]]**: $D[c] = 0$ for constant morphisms

## Examples

- **[[Smooth manifold|Smooth manifolds]]**: With standard differentiation
- **[[Polynomial functor|Polynomial functors]]**: With formal differentiation
- **[[Synthetic differential geometry]]**: With infinitesimal objects

<!-- \begin{tikzcd} A \times B \times B \arrow[r, "D[f]"] & C \\ A \times B \arrow[u, "\text{id} \times \Delta"] \arrow[r, "f"'] & C \arrow[u, "\text{id}"] \end{tikzcd} -->

## Cartesian Differential Categories

**Cartesian differential categories** are differential categories where the underlying category is cartesian, providing additional structure for handling:

- **[[Product rule]]**: For derivatives of products
- **[[Partial derivative|Partial derivatives]]**: Via projection morphisms
- **[[Higher-order derivative|Higher-order derivatives]]**: Iterated application

## Applications

Differential categories provide abstract formulations of many notions involving differentiation such as directional derivatives, differential forms, smooth manifolds, and De Rham cohomology.

## Related Concepts

- **[[Reverse differential category]]**: Adds reverse-mode differentiation
- **[[Tangent category]]**: Alternative axiomatization using tangent bundles
- **[[Differential linear logic]]**: The logical system this axiomatizes

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)