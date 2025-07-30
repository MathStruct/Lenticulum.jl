#definition #category-theory #computable #catlab

A **functor** $F: \mathcal{C} \to \mathcal{D}$ is a structure-preserving mapping between [[category|categories]] that preserves both [[Object|objects]], [[morphism|morphisms]], [[composition]], and [[identity morphism|identities]].

## Formal Definition

A functor $F: \mathcal{C} \to \mathcal{D}$ consists of:

- **Object mapping**: $F: \text{Ob}(\mathcal{C}) \to \text{Ob}(\mathcal{D})$
- **Morphism mapping**: For each $f: A \to B$ in $\mathcal{C}$, a morphism $F(f): F(A) \to F(B)$ in $\mathcal{D}$

## Functor Laws

- **Composition preservation**: $F(g \circ f) = F(g) \circ F(f)$ for all composable morphisms $f, g$
- **Identity preservation**: $F(\text{id}_A) = \text{id}_{F(A)}$ for all objects $A$

## Types

- **[[Covariant functor]]**: Preserves direction of morphisms
- **[[Contravariant functor]]**: Reverses direction of morphisms
- **[[Endofunctor]]**: Functor from a category to itself
- **[[Forgetful functor]]**: "Forgets" structure
- **[[Free functor]]**: Left adjoint to forgetful functors

## Examples

- **Forgetful functor** $U: \text{Grp} \to \text{Set}$: maps groups to underlying sets
- **Powerset functor** $\mathcal{P}: \text{Set} \to \text{Set}$: maps sets to their powersets
- **Fundamental group** $\pi_1: \text{Top} \to \text{Grp}$: assigns fundamental groups

<!-- \begin{tikzcd} \mathcal{C} \arrow[r, "F", bend left] & \mathcal{D} \\ A \arrow[r, mapsto] \arrow[d, "f"'] & F(A) \arrow[d, "F(f)"] \\ B \arrow[r, mapsto] & F(B) \end{tikzcd} -->

Functors are the [[morphism|morphisms]] in the [[2-category]] [[Cat]] and form the foundation for [[natural transformation|natural transformations]].

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/categorical_algebra/)