---
aliases: [Cat, 1-Category]
---
A **category** consists of:

- A collection of [[object|objects]]
- For each pair of objects $A, B$, a collection of [[morphism|morphisms]] (or arrows) from $A$ to $B$, denoted $\text{Hom}(A,B)$ or $C(A,B)$
- A [[composition]] operation that assigns to each pair of composable morphisms $f: A \to B$ and $g: B \to C$ a composite morphism $g \circ f: A \to C$

## Axioms

**Associativity**: For morphisms $f: A \to B$, $g: B \to C$, and $h: C \to D$, we have $(h \circ g) \circ f = h \circ (g \circ f)$.

**Identity**: For each object $A$, there exists an [[identity morphism]] $\text{id}_A: A \to A$ such that for any morphism $f: A \to B$, we have $f \circ \text{id}_A = f$ and $\text{id}_B \circ f = f$.

<!-- Basic category structure: \begin{tikzcd} A \arrow[r, "f"] & B \arrow[r, "g"] & C \end{tikzcd} with composite: $g \circ f: A \to C$ -->

Examples include [[Set]] (sets and functions), [[Top]] (topological spaces and continuous maps), [[Grp]] (groups and group homomorphisms), and [[Vec]] (vector spaces and linear maps).
