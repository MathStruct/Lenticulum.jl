#definition #category-theory #optics #lens #composition

**Lens composition** is the [[composition|compositional]] combination of [[lens|lenses]] that preserves the [[bidirectional]] access properties, allowing nested focusing through multiple layers of data structures while maintaining the [[lens laws]].

## Formal Definition

Given [[Cartesian lens|Cartesian lenses]]:

- $\ell_1: S \rightsquigarrow A$ with $(\text{view}_1: S \to A, \text{update}_1: S \times A \to S)$
- $\ell_2: A \rightsquigarrow B$ with $(\text{view}_2: A \to B, \text{update}_2: A \times B \to A)$

Their **composition** $\ell_2 \circ \ell_1: S \rightsquigarrow B$ is defined by:

$$\text{view} = \text{view}_2 \circ \text{view}_1: S \to B$$

$$\text{update}(s, b) = \text{update}_1(s, \text{update}_2(\text{view}_1(s), b)): S \times B \to S$$

## Diagrammatic Representation

The composition can be visualized as:

<!-- \begin{tikzcd} S \arrow[r, "\text{view}_1"] \arrow[d, "\langle \text{id}, \text{view} \rangle"'] & A \arrow[r, "\text{view}_2"] & B \\ S \times B \arrow[rr, "\text{update}"'] & & S \arrow[u, "\text{view}"'] \end{tikzcd} -->

## Verification of Lens Laws

### View-Update Law

$$\text{update}(s, \text{view}(s)) = \text{update}_1(s, \text{update}_2(\text{view}_1(s), \text{view}_2(\text{view}_1(s))))$$

By the view-update law for $\ell_2$: $$= \text{update}_1(s, \text{view}_1(s)) = s$$

### Update-View Law

$$\text{view}(\text{update}(s, b)) = \text{view}_2(\text{view}_1(\text{update}_1(s, \text{update}_2(\text{view}_1(s), b))))$$

By the update-view law for $\ell_1$: $$= \text{view}_2(\text{update}_2(\text{view}_1(s), b)) = b$$

### Update-Update Law

Follows from the corresponding laws for $\ell_1$ and $\ell_2$ by [[associativity]] of the composition operation.

## Category Structure

Lens composition makes [[lens|lenses]] form a [[category]] $\mathbf{Lens}$:

### Objects

Objects of the underlying [[cartesian category]]

### Morphisms

[[Cartesian lens|Cartesian lenses]] between objects

### Identity

For each object $A$: $$\text{id}_A = (\text{id}_A: A \to A, \pi_1: A \times A \to A)$$

### Associativity

For lenses $\ell_1: S \rightsquigarrow A$, $\ell_2: A \rightsquigarrow B$, $\ell_3: B \rightsquigarrow C$: $$(\ell_3 \circ \ell_2) \circ \ell_1 = \ell_3 \circ (\ell_2 \circ \ell_1)$$

## Profunctor Optics Composition

In the [[profunctor optics]] framework, lens composition corresponds to [[function composition]] of [[Tambara module]] transformations:

$$\begin{align} &\text{Lens}(A, B) \times \text{Lens}(S, A) \ &\cong (\forall P. \text{Tambara}(P) \Rightarrow P(B,B) \to P(A,A)) \times (\forall P. \text{Tambara}(P) \Rightarrow P(A,A) \to P(S,S)) \ &\to (\forall P. \text{Tambara}(P) \Rightarrow P(B,B) \to P(S,S)) \ &\cong \text{Lens}(S, B) \end{align}$$

## Examples

### Record Field Composition

For nested records:

```haskell
Person = { name: String, address: Address }
Address = { street: String, city: String }
```

The lens to access city: $$\text{cityLens} = \text{addressLens} \circ \text{nameLens}$$

### Concrete Calculation

Given $S = A \times (B \times C)$ and lenses:

- $\ell_1: S \to B \times C$ (second projection)
- $\ell_2: B \times C \to C$ (second projection)

Their composition $\ell_2 \circ \ell_1: S \to C$:

- $\text{view}((a, (b, c))) = c$
- $\text{update}((a, (b, c)), c') = (a, (b, c'))$

## Properties

### Functoriality

The lens composition operation is [[functor|functorial]], preserving:

- [[Identity morphism|Identities]]
- [[Composition]] structure
- [[Naturality]] conditions

### Monoidal Structure

Lens composition respects [[monoidal category|monoidal structure]]: $$(\ell_1 \otimes \ell_2) \circ (\ell_3 \otimes \ell_4) = (\ell_1 \circ \ell_3) \otimes (\ell_2 \circ \ell_4)$$

### Relationship to Kleisli Composition

In the [[Kleisli category]] of the [[state monad]], lens composition corresponds to [[Kleisli composition]]: $$(\ell_2 \circ \ell_1)^* = \ell_2^* \circ \ell_1^*$$

## Computational Complexity

### View Operation

Composing $n$ lenses gives view complexity $O(n)$ for sequential access.

### Update Operation

Update complexity is $O(n)$ but requires careful [[sharing]] to avoid excessive copying in [[persistent data structure|persistent]] implementations.

## Applications

### Nested Data Access

Composition enables deep focusing into complex data structures: $$\text{employee.department.manager.name}$$

### Bidirectional Transformations

In [[database theory]], composition of [[database]] lenses maintains [[ACID properties|consistency]] across nested [[view|views]].

### Functional Programming

Lens composition provides:

- Type-safe nested updates
- Compositional [[API]] design
- [[Immutable]] data manipulation

Lens composition provides the fundamental [[categorical]] structure that makes [[lens|lenses]] a powerful tool for [[bidirectional]] data access, combining mathematical rigor with practical [[software engineering]] benefits.