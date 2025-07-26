#definition #category-theory #optics #computable

A **lens** is a bidirectional data accessor consisting of a pair of [[morphism|morphisms]]: a getter and a setter that satisfy coherence conditions.

## Concrete Definition

For [[object|objects]] $S$ (source), $T$ (target), $A$ (focus), $B$ (updated focus), a lens consists of:

- **Get**: $\text{get}: S \to A$ - extracts the focus
- **Put**: $\text{put}: S \times B \to T$ - updates the focus

## Lens Laws

- **Get-Put**: $\text{put}(s, \text{get}(s)) = s$ for all $s: S$
- **Put-Get**: $\text{get}(\text{put}(s, b)) = b$ for all $s: S, b: B$
- **Put-Put**: $\text{put}(\text{put}(s, b_1), b_2) = \text{put}(s, b_2)$ for all $s: S, b_1, b_2: B$

## Categorical Characterization

A lens is an [[optic]] for the action of a [[Cartesian Category]] with the [[product functor]] $(\times): \mathbf{C} \to [\mathbf{C}, \mathbf{C}]$.

In the [[profunctor]] representation: a lens is an optic that works uniformly for all [[Cartesian profunctor|Cartesian profunctors]].

<!-- \begin{tikzcd} S \arrow[r, "\text{get}"] \arrow[d, "\langle \text{id}, \text{get} \rangle"'] & A \\ S \times A \arrow[r, "\text{id} \times f"'] & S \times B \arrow[u, "\text{put}"'] \end{tikzcd} -->

Lenses form a [[category]] under composition and are fundamental examples of [[Tambara module|Tambara modules]] for the [[cartesian monoidal category|cartesian monoidal structure]].