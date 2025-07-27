#definition #category-theory #optics #profunctor #lens

A **Tambara module** is a [[profunctor]] equipped with additional structure that enables the representation of [[lens|lenses]] and other [[optic|optics]] in functional programming and categorical optics theory.

## Formal Definition

Let $(\mathcal{C}, \otimes, I)$ be a [[monoidal category]]. A **Tambara module** for a [[profunctor]] $P: \mathcal{C}^{\text{op}} \times \mathcal{C} \to \mathbf{Set}$ is a [[natural transformation]]:

$$\text{strength}: P(A, B) \to P(C \otimes A, C \otimes B)$$

for all objects $A, B, C \in \mathcal{C}$, satisfying coherence conditions with respect to the [[monoidal structure]].

### Coherence Laws

The strength must satisfy:

1. **Associativity**: The following [[commutative diagram|diagram commutes]]:

<!-- \begin{tikzcd} P(A, B) \arrow[r] \arrow[d] & P(D \otimes A, D \otimes B) \arrow[d] \\ P((C \otimes D) \otimes A, (C \otimes D) \otimes B) \arrow[r] & P(C \otimes (D \otimes A), C \otimes (D \otimes B)) \end{tikzcd} -->

2. **Unit law**: $\text{strength}_{I,A,B} = \text{id}_{P(A,B)}$ where $I$ is the [[monoidal unit]]
    
3. **[[Naturality]]**: The strength is natural in all arguments
    

## Alternative Characterization

A Tambara module can equivalently be defined as a profunctor $P$ equipped with operations: $$\text{first}: P(A, B) \to P(A \otimes C, B \otimes C)$$ $$\text{second}: P(A, B) \to P(C \otimes A, C \otimes B)$$

satisfying appropriate coherence conditions.

## Basic Examples

### Function Profunctor

The [[function]] profunctor $\text{Hom}: \mathcal{C}^{\text{op}} \times \mathcal{C} \to \mathbf{Set}$ with: $$\text{strength}(f: A \to B) = \lambda(c \otimes a). c \otimes f(a)$$

### Kleisli Profunctor

For a [[strong monad]] $T$ on a monoidal category, the Kleisli profunctor: $$P(A, B) = \mathcal{C}(A, T(B))$$ with strength derived from the monad's strength.

## Connection to Lenses

A **[[lens]]** from $S$ to $A$ (focusing on part $A$ within structure $S$) corresponds to a Tambara module morphism: $$\text{Lens}(S, A) \cong \forall P. \text{Tambara}(P) \Rightarrow P(A, A) \to P(S, S)$$

This provides the categorical foundation for van Laarhoven lenses in functional programming.

## Generalized Tambara Modules

### Indexed Tambara Modules

For [[indexed category|indexed categories]] or [[fibration|fibrations]], Tambara modules can be indexed over base categories, enabling more sophisticated optics.

### Strong Tambara Modules

When the underlying category has additional structure (e.g., [[cartesian monoidal category|cartesian]] or [[closed monoidal category|closed]]), stronger versions of Tambara modules arise.

## Applications

### Categorical Optics

Tambara modules provide the theoretical foundation for:

- [[Lens|Lenses]]: Focusing on parts of data structures
- [[Prism|Prisms]]: Handling sum types and partial access
- [[Traversal|Traversals]]: Accessing multiple foci

### Profunctor Optics

The [[profunctor optics]] approach uses Tambara modules to encode various optical structures uniformly:

<!-- \begin{tikzcd} \text{Optic}(S, T, A, B) & P(A, B) \arrow[l] \\ & P(S, T) \arrow[u] \end{tikzcd} -->

### Database Theory

Tambara modules appear in categorical approaches to [[database]] lenses and [[bidirectional transformation|bidirectional transformations]].

## Properties

### Composition

Tambara modules compose via [[profunctor composition]], preserving the Tambara structure under appropriate conditions.

### Monoidal Structure

The category of Tambara modules over a fixed profunctor can inherit [[monoidal category|monoidal structure]] from the underlying category.

### Relationship to Coends

Tambara modules can be characterized using [[coend|coends]] and [[end|ends]], connecting to [[weighted limit|weighted limits]] in category theory.

Tambara modules bridge [[category theory]], [[functional programming]], and [[database theory]], providing a unified framework for understanding [[bidirectional]] data access patterns and [[compositional]] program construction.