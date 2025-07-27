#definition #category-theory #computable #catlab

An **endofunctor** is a [[functor]] from a [[category]] to itself.

## Formal Definition

Given a category $\mathcal{C}$, an **endofunctor** is a [[functor]] $F: \mathcal{C} \to \mathcal{C}$ that maps:

- Each [[object]] $X \in \mathcal{C}$ to an object $F(X) \in \mathcal{C}$
- Each [[morphism]] $f: X \to Y$ in $\mathcal{C}$ to a morphism $F(f): F(X) \to F(Y)$ in $\mathcal{C}$

Such that:

1. **Identity preservation**: $F(\text{id}_X) = \text{id}_{F(X)}$ for all objects $X$
2. **[[Composition]] preservation**: $F(g \circ f) = F(g) \circ F(f)$ for all composable morphisms $f, g$

## Basic Examples

- **Identity functor**: $\text{Id}_{\mathcal{C}}: \mathcal{C} \to \mathcal{C}$ mapping each object and morphism to itself
- **Power set functor** on $\mathbf{Set}$: $\mathcal{P}: \mathbf{Set} \to \mathbf{Set}$ mapping sets to their power sets
- **List functor** on $\mathbf{Set}$: $L: \mathbf{Set} \to \mathbf{Set}$ mapping sets to lists over those sets

## Structure

Endofunctors on a category $\mathcal{C}$ form a [[monoid]] under [[functor composition]], with the [[identity functor]] as the identity element:

<!-- \begin{tikzcd} \mathcal{C} \arrow[r, "F", bend left] \arrow[r, "G \circ F"', bend right] & \mathcal{C} \arrow[l, "G", bend left] \end{tikzcd} -->

## Properties

- **[[Monad|Monads]]**: An endofunctor equipped with [[natural transformation|natural transformations]] $\eta: \text{Id} \Rightarrow F$ and $\mu: F^2 \Rightarrow F$ satisfying certain coherence conditions
- **[[Comonad|Comonads]]**: Dual structure with transformations $\epsilon: F \Rightarrow \text{Id}$ and $\delta: F \Rightarrow F^2$
- **[[Fixed point|Fixed points]]**: Objects $X$ such that $F(X) \cong X$ are of particular interest

## Applications

Endofunctors are central to:

- **[[Algebra over a functor|Algebras]]**: Objects equipped with structure maps from $F(X) \to X$
- **[[Coalgebra over a functor|Coalgebras]]**: Objects with structure maps $X \to F(X)$
- **[[Categorical semantics]]** of programming languages and type theory
- **[[Homotopy theory]]** and [[higher category theory]]

Endofunctors provide the foundation for understanding [[monad|monads]], [[comonad|comonads]], and other fundamental structures in [[category theory]].