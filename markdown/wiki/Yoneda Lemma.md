#theorem #category-theory #representable #computable #catlab

The **Yoneda Lemma** establishes a [[natural isomorphism]] between [[natural transformation|natural transformations]] from a [[representable functor]] to any [[functor]] and the elements of that functor's image.

## Statement

For any [[category]] $\mathcal{C}$, [[Object]] $A \in \mathcal{C}$, and [[functor]] $F: \mathcal{C} \to \text{Set}$, there is a [[natural isomorphism]]:

$$\text{Nat}(\text{Hom}(A, -), F) \cong F(A)$$

where $\text{Hom}(A, -)$ is the [[hom-functor]] and $\text{Nat}$ denotes the set of [[natural transformation|natural transformations]].

## Explicit Isomorphism

The isomorphism is given by:

- **Forward direction**: $\alpha \mapsto \alpha_A(\text{id}_A)$ for $\alpha: \text{Hom}(A, -) \Rightarrow F$
- **Backward direction**: $x \mapsto (f \mapsto F(f)(x))$ for $x \in F(A)$

## Yoneda Embedding

The **Yoneda embedding** $Y: \mathcal{C} \to [\mathcal{C}^{\text{op}}, \text{Set}]$ is defined by: $$Y(A) = \text{Hom}(-, A): \mathcal{C}^{\text{op}} \to \text{Set}$$

This embedding is:

- **[[Full functor|Full]]**: Every natural transformation arises from a morphism
- **[[Faithful functor|Faithful]]**: Different morphisms give different natural transformations
- **[[Conservative functor|Conservative]]**: Reflects isomorphisms

<!-- \begin{tikzcd} \mathcal{C} \arrow[r, "Y"] & {[\mathcal{C}^{\text{op}}, \text{Set}]} \\ A \arrow[r, mapsto] & \text{Hom}(-, A) \\ (f: A \to B) \arrow[r, mapsto] & (\text{Hom}(-, f): \text{Hom}(-, A) \Rightarrow \text{Hom}(-, B)) \end{tikzcd} -->

## Corollary (Yoneda's Lemma for Morphisms)

For objects $A, B$ in $\mathcal{C}$: $$\text{Nat}(\text{Hom}(A, -), \text{Hom}(B, -)) \cong \text{Hom}(B, A)$$

This shows that $\text{Hom}(A, -)$ determines $A$ up to isomorphism.

## Applications

- **[[Representable functor|Representable functors]]**: Characterizes when functors are representable
- **[[Universal property|Universal properties]]**: Expresses universality in terms of representability
- **[[Presheaf]]**: Every presheaf can be expressed as a colimit of representables
- **[[Topos theory]]**: Foundation for geometric morphisms and sites

## Significance

The Yoneda Lemma shows that objects are completely determined by their relationships (morphisms) to all other objects, making it fundamental to the categorical principle that "structure is determined by morphisms."

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/categorical_algebra/)