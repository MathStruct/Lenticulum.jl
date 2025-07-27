#definition #category-theory #higher-category-theory #homotopy-theory

An **n-category** is a [[higher category]] with [[k-morphism|k-morphisms]] for $0 \leq k \leq n$, where all [[k-morphism|k-morphisms]] with $k > n$ are [[invertible morphism|invertible]]. This provides a framework for organizing mathematical structures with multiple levels of [[morphism|morphisms]] and their [[composition|compositions]].

## Formal Definition

An **n-category** $\mathcal{C}$ consists of:

### Data

- A collection of [[object|0-morphisms]] (objects)
- For each $1 \leq k \leq n$, collections of [[k-morphism|k-morphisms]] between parallel $(k-1)$-morphisms
- For $k > n$, all k-morphisms are [[equivalence|equivalences]] (effectively [[identity morphism|identities]])

### Structure

- [[Composition]] operations for k-morphisms at each level $1 \leq k \leq n$
- [[Identity morphism|Identity]] k-morphisms for each level
- [[Associativity]] and [[unit law|unit laws]] holding up to coherent [[isomorphism|isomorphisms]] at higher levels

## Low-Dimensional Cases

### 0-Category

A **0-category** is a [[set]] (or collection) of objects with no non-trivial morphisms.

### 1-Category

A **1-category** is an ordinary [[category]] with:

- [[Object|Objects]] (0-morphisms)
- [[Morphism|Morphisms]] (1-morphisms) between objects
- [[Composition]] and [[identity morphism|identities]] satisfying [[associativity]] and [[unit law|unit laws]] strictly

### 2-Category

A **[[2-category]]** has:

- [[Object|Objects]]
- [[1-morphism|1-morphisms]] between objects
- [[2-morphism|2-morphisms]] between parallel 1-morphisms
- Composition laws with [[associativity]] and unit laws holding up to [[invertible 2-morphism|invertible 2-morphisms]]

### 3-Category

A **3-category** additionally has:

- [[3-morphism|3-morphisms]] between parallel 2-morphisms
- Higher coherence laws involving [[invertible 3-morphism|invertible 3-morphisms]]

## Strictness vs. Weakness

### Strict n-Categories

All [[composition]] and [[associativity]] laws hold as strict equalities. These form the [[category]] $n\text{-}\mathbf{Cat}_{\text{strict}}$.

### Weak n-Categories

[[Composition]] and coherence laws hold up to coherent [[isomorphism|isomorphisms]]. The associativity and unit laws are satisfied up to [[invertible morphism|invertible]] $(n+1)$-morphisms, with higher coherence conditions.

## Examples

### Cat as 2-Category

[[Cat]] is the archetypal 2-category:

- Objects: Small [[category|categories]]
- 1-morphisms: [[Functor|Functors]]
- 2-morphisms: [[Natural transformation|Natural transformations]]

### Homotopy n-Types

The [[homotopy category]] of [[n-type|n-types]] forms an n-category where:

- Objects: [[Topological space|Topological spaces]]
- k-morphisms: [[Homotopy class|Homotopy classes]] of [[map|maps]] $S^{k-1} \to \text{Map}(X,Y)$

### Algebraic Structures

Higher [[algebraic structure|algebraic structures]] like [[A-infinity algebra|A∞-algebras]] and [[E-n algebra|En-algebras]] organize naturally into n-categories.

## Composition Operations

At each level $k$, there are multiple composition operations:

### Source and Target Maps

For k-morphisms $f$, there are $(k-1)$-morphisms:

- $s_i(f)$: source along the i-th direction
- $t_i(f)$: target along the i-th direction

### Composition Along Direction i

$$\circ_i: \mathcal{C}_k \times_{s_i,t_i} \mathcal{C}_k \to \mathcal{C}_k$$

These satisfy [[interchange law|interchange laws]]: $$f \circ_i (g \circ_j h) = (f \circ_i g) \circ_j (f \circ_i h)$$ when $i \neq j$

## Coherence Conditions

### Mac Lane's Coherence

Higher coherence conditions ensure that different ways of [[bracketing]] compositions yield the same result up to canonical [[isomorphism|isomorphisms]].

### Batanin's Approach

Uses [[operad|operads]] to encode the coherence structure systematically.

### Trimble's Definition

Employs [[tetracategory|tetracategories]] and higher [[multicategory|multicategories]].

## ∞-Categories

The limit case where $n = \infty$ gives **[[infinity-category|∞-categories]]**, fundamental objects in [[homotopy theory]] and [[algebraic topology]].

### Models

- [[Quasi-category|Quasi-categories]] (∞-categories as [[simplicial set|simplicial sets]])
- [[Complete Segal space|Complete Segal spaces]]
- [[Category enriched in spaces|Categories enriched in spaces]]

## Applications

### Homotopy Theory

n-categories provide models for [[homotopy n-type|homotopy n-types]] and [[stable homotopy theory]].

### Algebraic Topology

[[Fundamental groupoid|Fundamental groupoids]], [[loop space|loop spaces]], and [[spectrum|spectra]] organize into n-categorical structures.

### Mathematical Physics

[[Topological quantum field theory|TQFTs]], [[string theory]], and [[gauge theory]] use higher categorical structures.

### Computer Science

[[Type theory]], [[concurrency theory]], and [[program semantics]] employ n-categorical models.

## Higher Categorical Logic

### Internal Logic

n-categories support [[internal language|internal languages]] generalizing [[Mitchell-Bénabou language]].

### Homotopy Type Theory

[[Homotopy type theory]] provides [[synthetic]] foundations where types are interpreted as [[infinity-groupoid|∞-groupoids]].

n-categories provide the mathematical framework for organizing and understanding structures with multiple levels of [[morphism|morphisms]], fundamental to modern [[algebraic topology]], [[homotopy theory]], and [[mathematical physics]].