#definition #category-theory #functor #optics #relation

A **profunctor** is a [[functor]] from the [[product category]] $\mathcal{C}^{\text{op}} \times \mathcal{D}$ to another category, typically $\mathbf{Set}$. Profunctors generalize [[relation|relations]] and provide the foundation for [[optics]], [[lens|lenses]], and bidirectional transformations.

## Formal Definition

A **profunctor** from [[category]] $\mathcal{C}$ to category $\mathcal{D}$ is a [[functor]]:

$$P: \mathcal{C}^{\text{op}} \times \mathcal{D} \to \mathcal{E}$$

where $\mathcal{E}$ is typically $\mathbf{Set}$. This means:

- For objects $C \in \mathcal{C}$ and $D \in \mathcal{D}$, we have $P(C, D) \in \mathcal{E}$
- For [[morphism|morphisms]] $f: C' \to C$ and $g: D \to D'$, we have: $$P(f, g): P(C, D) \to P(C', D')$$

## Variance

A profunctor is:

- **Contravariant** in the first argument: $P(f, \text{id}): P(C, D) \to P(C', D)$
- **Covariant** in the second argument: $P(\text{id}, g): P(C, D) \to P(C, D')$

This mixed variance is written as $P: \mathcal{C} \nrightarrow \mathcal{D}$.

## Basic Examples

### Hom Profunctor

The fundamental example is the [[hom-functor]]: $$\text{Hom}: \mathcal{C}^{\text{op}} \times \mathcal{C} \to \mathbf{Set}$$ $$\text{Hom}(A, B) = \mathcal{C}(A, B)$$

### Constant Profunctor

For a fixed object $X \in \mathcal{E}$: $$\Delta_X: \mathcal{C}^{\text{op}} \times \mathcal{D} \to \mathcal{E}$$ $$\Delta_X(C, D) = X$$

### Matrix Profunctor

For finite sets, a profunctor $P: \mathbf{FinSet}^{\text{op}} \times \mathbf{FinSet} \to \mathbf{Set}$ can be represented as a matrix where $P(m, n)$ is a set of $m \times n$ matrices.

### Relation Profunctor

A [[relation]] $R \subseteq A \times B$ gives rise to a profunctor: $$R: \mathbf{1}^{\text{op}} \times \mathbf{1} \to \mathbf{Set}$$ where $\mathbf{1}$ is the [[terminal category]].

## Profunctor Composition

Profunctors compose via [[coend|coends]]. For profunctors $P: \mathcal{C} \nrightarrow \mathcal{D}$ and $Q: \mathcal{D} \nrightarrow \mathcal{E}$:

$$(Q \circ P)(C, E) = \int^{D \in \mathcal{D}} Q(D, E) \times P(C, D)$$

This gives the [[bicategory]] $\mathbf{Prof}$ where:

- Objects: Categories
- 1-morphisms: Profunctors
- 2-morphisms: [[Natural transformation|Natural transformations]]

## Diagrammatic Notation

Profunctors are often drawn as:

<!-- \begin{tikzcd} \mathcal{C} \arrow[r, "P", rightarrow, bend left] & \mathcal{D} \end{tikzcd} -->

The direction indicates the covariant argument, while contravariance is implicit.

## Relationship to Functors

### Functor to Profunctor

Every [[functor]] $F: \mathcal{C} \to \mathcal{D}$ induces profunctors:

- **Graph**: $\Gamma_F(C, D) = \mathcal{D}(F(C), D)$
- **Cograph**: $\Gamma^F(C, D) = \mathcal{D}(D, F(C))$

### Representable Profunctors

A profunctor $P: \mathcal{C} \nrightarrow \mathcal{D}$ is **representable** if: $$P(C, D) \cong \mathcal{D}(F(C), D)$$ for some functor $F: \mathcal{C} \to \mathcal{D}$.

## Profunctor Optics

Profunctors provide the theoretical foundation for [[optics]]:

### Lens Representation

A [[lens]] from $S$ to $A$ corresponds to: $$\forall P. \text{Strong}(P) \Rightarrow P(A, A) \to P(S, S)$$ where $\text{Strong}(P)$ means $P$ is a [[Tambara module]].

### Prism Representation

A [[prism]] corresponds to: $$\forall P. \text{Choice}(P) \Rightarrow P(A, A) \to P(S, S)$$ where $\text{Choice}(P)$ provides coproduct structure.

## Profunctor Functors

### Contravariant Functors

A [[contravariant functor]] $F: \mathcal{C}^{\text{op}} \to \mathbf{Set}$ is a profunctor $F: \mathcal{C} \nrightarrow \mathbf{1}$.

### Covariant Functors

A [[functor]] $F: \mathcal{C} \to \mathbf{Set}$ is a profunctor $F: \mathbf{1} \nrightarrow \mathcal{C}$.

## Enriched Profunctors

In [[enriched category theory]], profunctors can be enriched over a [[monoidal category]] $\mathcal{V}$: $$P: \mathcal{C}^{\text{op}} \times \mathcal{D} \to \mathcal{V}$$

## Kleisli Construction

For a [[monad]] $T$ on $\mathcal{C}$, the [[Kleisli category]] $\mathcal{C}_T$ gives rise to profunctors: $$\mathcal{C}(-, T(-)) : \mathcal{C} \nrightarrow \mathcal{C}_T$$

## Properties

### Yoneda Lemma

The [[Yoneda lemma]] for profunctors states: $$[\mathcal{C}^{\text{op}} \times \mathcal{D}, \mathbf{Set}](P, \text{Hom}(-, F(-))) \cong [\mathcal{C}, \mathcal{D}](F, P)$$

### Profunctor Theorem

Every profunctor can be written as a [[colimit]] of representable profunctors.

## Applications

### Database Theory

Profunctors model [[query|queries]] and [[bidirectional transformation|bidirectional transformations]] between database schemas.

### Programming Languages

- [[Type theory]] and [[polymorphism]]
- [[Functional programming]] abstractions
- [[Effect system|Effect systems]]

### Algebraic Topology

Profunctors appear in [[correspondence|correspondences]] and [[span|spans]] in [[homotopy theory]].

### Logic

[[Proof theory]] and [[categorical logic]] use profunctors for [[model|models]] and [[interpretation|interpretations]].

Profunctors bridge the gap between [[category theory]] and practical applications, providing a unified framework for [[relation|relations]], [[transformation|transformations]], and [[bidirectional]] data access patterns.