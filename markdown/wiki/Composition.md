#definition #category-theory #morphism #fundamental

**Composition** is the fundamental operation in [[category theory]] that combines compatible [[morphism|morphisms]] to form new morphisms, satisfying [[associativity]] and [[unit law|unit laws]] that define the structure of a [[category]].

## Formal Definition

In a [[category]] $\mathcal{C}$, **composition** is a [[partial function]]:

$$\circ: \text{Mor}(\mathcal{C}) \times \text{Mor}(\mathcal{C}) \rightharpoonup \text{Mor}(\mathcal{C})$$

For [[morphism|morphisms]] $f: A \to B$ and $g: B \to C$, their **composite** is:

$$g \circ f: A \to C$$

defined when the [[codomain]] of $f$ equals the [[domain]] of $g$.

## Diagrammatic Representation

Composition is visualized as:

<!-- \begin{tikzcd} A \arrow[r, "f"] \arrow[rr, "g \circ f", bend right] & B \arrow[r, "g"] & C \end{tikzcd} -->

## Fundamental Laws

### Associativity

For composable morphisms $f: A \to B$, $g: B \to C$, $h: C \to D$: $$h \circ (g \circ f) = (h \circ g) \circ f$$

### Unit Laws

For any morphism $f: A \to B$:

- **Left unit**: $f \circ \text{id}_A = f$
- **Right unit**: $\text{id}_B \circ f = f$

where $\text{id}_A$ and $\text{id}_B$ are [[identity morphism|identity morphisms]].

## Composition in Higher Categories

### 2-Categories

In a [[2-category]], there are two types of composition:

#### Horizontal Composition

For [[1-morphism|1-morphisms]] $f: A \to B$, $g: B \to C$: $$g \circ f: A \to C$$

#### Vertical Composition

For [[2-morphism|2-morphisms]] $\alpha: f \Rightarrow g$ and $\beta: g \Rightarrow h$: $$\beta \cdot \alpha: f \Rightarrow h$$

### Interchange Law

The fundamental coherence condition relating horizontal and vertical composition: $$(\beta_2 \cdot \alpha_2) \circ (\beta_1 \cdot \alpha_1) = (\beta_2 \circ \beta_1) \cdot (\alpha_2 \circ \alpha_1)$$

### n-Categories

In [[n-category|n-categories]], there are multiple composition operations $\circ_i$ for $1 \leq i \leq n$, satisfying [[interchange law|interchange laws]]: $$f \circ_i (g \circ_j h) = (f \circ_i g) \circ_j (f \circ_i h)$$ when $i \neq j$

## Examples

### Set Category

In $\mathbf{Set}$, composition is [[function composition]]: $$(g \circ f)(x) = g(f(x))$$

### Group Category

In the category $\mathbf{Grp}$, composition of [[group homomorphism|group homomorphisms]] $f: G \to H$ and $g: H \to K$: $$(g \circ f)(x) = g(f(x))$$ preserving group structure.

### Topological Spaces

In $\mathbf{Top}$, composition of [[continuous map|continuous maps]] yields continuous maps by the [[composition of continuous functions|composition theorem]].

### Matrix Multiplication

In the category of finite-dimensional vector spaces, composition corresponds to [[matrix multiplication]]: $$[g \circ f] = [g][f]$$

## Special Cases

### Endomorphism Composition

For [[endomorphism|endomorphisms]] $f, g: A \to A$, composition gives the [[endomorphism monoid]] $\text{End}(A)$.

### Automorphism Composition

[[Automorphism|Automorphisms]] form a [[group]] $\text{Aut}(A)$ under composition.

### Functor Composition

For [[functor|functors]] $F: \mathcal{C} \to \mathcal{D}$ and $G: \mathcal{D} \to \mathcal{E}$: $$(G \circ F)(X) = G(F(X))$$ $$(G \circ F)(f) = G(F(f))$$

## Properties

### Composition Preserves Structure

- [[Monomorphism|Monomorphisms]]: If $f$ and $g$ are [[monomorphism|monic]], then $g \circ f$ is monic
- [[Epimorphism|Epimorphisms]]: If $f$ and $g$ are [[epimorphism|epic]], then $g \circ f$ is epic
- [[Isomorphism|Isomorphisms]]: If $f$ and $g$ are [[isomorphism|iso]], then $g \circ f$ is iso

### Cancellation Laws

- **Left cancellation**: If $g \circ f = g \circ f'$ and $g$ is [[monomorphism|monic]], then $f = f'$
- **Right cancellation**: If $f \circ g = f' \circ g$ and $g$ is [[epimorphism|epic]], then $f = f'$

## Composition as Tensor Product

In [[monoidal category|monoidal categories]], composition can be viewed as a special case of the [[tensor product]]: $$\circ: \text{Hom}(B,C) \otimes \text{Hom}(A,B) \to \text{Hom}(A,C)$$

## Computational Aspects

### Category of Computations

In [[computer science]], composition models:

- [[Program composition]]
- [[Pipeline|Data pipelines]]
- [[Function composition]] in functional programming

### Complexity

Composition complexity depends on the underlying structure:

- Function composition: $O(1)$ time
- Matrix composition: $O(n^3)$ for $n \times n$ matrices
- Graph morphism composition: Potentially exponential

## Weak Composition

In [[weak n-category|weak n-categories]], composition satisfies laws up to coherent [[isomorphism|isomorphisms]] rather than strict equality, leading to:

- [[Coherence theorem|Coherence theorems]]
- [[Higher associativity]]
- [[Homotopy coherent]] structures

## Universal Property

Composition can be characterized by [[universal property|universal properties]] in categories with additional structure, such as [[cartesian closed category|cartesian closed]] or [[monoidal closed category|monoidal closed categories]].

Composition is the cornerstone operation that transforms collections of [[object|objects]] and [[morphism|morphisms]] into the rich mathematical structures studied in [[category theory]], providing the foundation for [[functor|functors]], [[natural transformation|natural transformations]], and all higher categorical constructions.