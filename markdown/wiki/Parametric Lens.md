#definition #category-theory #optics #computable

A **parametric lens** is a [[lens]] equipped with an additional parameter that controls the behavior of the get and put operations.

## Concrete Definition

For [[object|objects]] $S$ (source), $T$ (target), $A$ (focus), $B$ (updated focus), and $P$ (parameter), a parametric lens consists of:

- **Parametric Get**: $\text{get}: P \times S \to A$ - extracts the focus using parameter
- **Parametric Put**: $\text{put}: P \times S \times B \to T$ - updates the focus using parameter

## Parametric Lens Laws

- **Get-Put**: $\text{put}(p, s, \text{get}(p, s)) = s$ for all $p: P, s: S$
- **Put-Get**: $\text{get}(p, \text{put}(p, s, b)) = b$ for all $p: P, s: S, b: B$
- **Put-Put**: $\text{put}(p, \text{put}(p, s, b_1), b_2) = \text{put}(p, s, b_2)$ for all $p: P, s: S, b_1, b_2: B$

## Categorical Characterization

A parametric lens is an [[optic]] that is [[natural transformation|natural]] in the parameter. In the [[profunctor]] representation, it corresponds to a [[dinatural transformation]] that commutes with the parameter.

<!-- \begin{tikzcd} P \times S \arrow[r, "\text{get}"] \arrow[d, "\text{id} \times \langle \text{id}, \text{get} \rangle"'] & A \\ P \times S \times A \arrow[r, "\text{id} \times \text{id} \times f"'] & P \times S \times B \arrow[u, "\text{put}"'] \end{tikzcd} -->

Parametric lenses generalize ordinary [[lens|lenses]] and form a [[category]] under parameter-preserving composition. They are examples of [[indexed optic|indexed optics]].