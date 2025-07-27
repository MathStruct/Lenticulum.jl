#definition #category-theory #optics #lens #cartesian

A **Cartesian lens** is a [[lens]] in a [[cartesian category]] that provides bidirectional access to a component of a [[product]] using the [[cartesian structure]], enabling both viewing and updating operations in a compositional manner.

## Formal Definition

In a [[cartesian category]] $\mathcal{C}$, a **Cartesian lens** from object $S$ to object $A$ is a pair of [[morphism|morphisms]]:

$$\text{view}: S \to A$$ $$\text{update}: S \times A \to S$$

satisfying the **lens laws**:

1. **View-Update**: $\text{update}(s, \text{view}(s)) = s$ for all $s: \top \to S$
2. **Update-View**: $\text{view}(\text{update}(s, a)) = a$ for all $s: \top \to S$ and $a: \top \to A$
3. **Update-Update**: $\text{update}(\text{update}(s, a), b) = \text{update}(s, b)$ for all $s, a, b$

## Categorical Representation

A Cartesian lens can be represented as a [[morphism]] in the [[Kleisli category]] of the [[state monad]] on $A$:

$$\ell: S \to A \times (A \to S)$$

where the first component is the view operation and the second encodes the update.

## Relationship to Products

For a [[product]] $S = A \times B$ in a cartesian category, the canonical Cartesian lens accessing the first component is:

$$\text{view} = \pi_1: A \times B \to A$$ $$\text{update} = \lambda((a, b), a'). (a', b): (A \times B) \times A \to A \times B$$

<!-- \begin{tikzcd} A \times B \arrow[r, "\pi_1"] \arrow[d, "\text{id} \times \text{view}"'] & A \\ (A \times B) \times A \arrow[r, "\text{update}"'] & A \times B \arrow[u, "\pi_1"'] \end{tikzcd} -->

## Composition

Cartesian lenses compose via [[function composition]]. Given lenses:

- $\ell_1: S \to A$ with $(\text{view}_1, \text{update}_1)$
- $\ell_2: A \to B$ with $(\text{view}_2, \text{update}_2)$

Their [[composition]] $\ell_2 \circ \ell_1: S \to B$ is:

$$\text{view} = \text{view}_2 \circ \text{view}_1$$ $$\text{update}(s, b) = \text{update}_1(s, \text{update}_2(\text{view}_1(s), b))$$

## Profunctor Representation

Cartesian lenses correspond to [[Tambara module|Tambara modules]] for the [[cartesian monoidal category|cartesian monoidal]] structure. A lens $S \rightsquigarrow A$ gives rise to:

$$P(A, A) \to P(S, S)$$

for any [[profunctor]] $P$ with [[Tambara module]] structure.

## Examples

### Record Field Access

For a record type $\text{Person} = \text{Name} \times \text{Age}$, the name lens is:

- $\text{view}(n, a) = n$
- $\text{update}((n, a), n') = (n', a)$

### Nested Lenses

Accessing nested structure $S = A \times (B \times C)$ to get component $C$: $$\ell = \pi_2 \circ \pi_2: A \times (B \times C) \to C$$

## Properties

### Category Structure

Cartesian lenses form a [[category]] $\mathbf{CLens}$ where:

- Objects are objects of the base cartesian category
- Morphisms are Cartesian lenses
- [[Identity morphism|Identity]]: $\text{id}_A = (\text{id}_A, \pi_1)$
- [[Composition]]: As defined above

### Monoidal Structure

$\mathbf{CLens}$ inherits [[monoidal category|monoidal structure]] from the base category: $$\ell_1 \otimes \ell_2: S_1 \times S_2 \to A_1 \times A_2$$

### Relationship to State

Cartesian lenses correspond exactly to [[coalgebra over a functor|coalgebras]] for the [[state monad|state comonad]] $\text{State}(A, -)$ in cartesian categories.

## Generalizations

### Polymorphic Lenses

Lenses where the source and target can change type: $$\ell: S \rightsquigarrow T \text{ focusing } A \rightsquigarrow B$$

### Affine Lenses

Lenses that may fail to update, using [[maybe monad|partial functions]] or [[option type|option types]].

### Higher-Order Lenses

Lenses in [[higher-order category|higher-order]] or [[enriched category|enriched]] categories.

## Applications

### Functional Programming

Cartesian lenses provide [[immutable]] data structure manipulation with:

- Compositional field access
- [[Pure function|Pure]] update operations
- Type-safe transformations

### Database Theory

[[Bidirectional transformation|Bidirectional transformations]] between [[database]] views and base tables using lens composition.

### Optics Libraries

Foundation for [[optics]] libraries in languages like Haskell, providing uniform interfaces for data access patterns.

Cartesian lenses provide a principled categorical foundation for [[bidirectional]] data access in [[cartesian category|cartesian]] settings, combining the [[mathematical]] elegance of [[category theory]] with practical [[software engineering]] applications.