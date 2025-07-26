A **2-morphism** (also called a **2-cell**) in a [[2-category]] is a [[morphism]] between parallel [[1-morphism|1-morphisms]].

Given two 1-morphisms $f, g: A \to B$ between the same pair of objects in a 2-category, a 2-morphism $\alpha: f \Rightarrow g$ provides a "transformation" or "deformation" from $f$ to $g$.

## Composition

2-morphisms can be composed in two ways:

- **Vertical composition**: If $\alpha: f \Rightarrow g$ and $\beta: g \Rightarrow h$, then $\beta \circ \alpha: f \Rightarrow h$
- **Horizontal composition**: If $\alpha: f \Rightarrow g$ and $\beta: f' \Rightarrow g'$, then $\beta * \alpha: f' \circ f \Rightarrow g' \circ g$

<!-- Vertical composition: \begin{tikzcd} A \arrow[r, "f", bend left=60] \arrow[r, "g"] \arrow[r, "h"', bend right=60] & B \end{tikzcd} $\alpha: f \Rightarrow g$, $\beta: g \Rightarrow h$, $\beta \circ \alpha: f \Rightarrow h$ Horizontal composition: \begin{tikzcd} A \arrow[r, "f", bend left] \arrow[r, "g"', bend right] & B \arrow[r, "f'", bend left] \arrow[r, "g'"', bend right] & C \end{tikzcd} $\alpha: f \Rightarrow g$, $\beta: f' \Rightarrow g'$, $\beta * \alpha: f' \circ f \Rightarrow g' \circ g$ -->

## Identity

Each 1-morphism $f: A \to B$ has an [[identity 2-morphism]] $\text{id}_f: f \Rightarrow f$.

Examples include [[natural transformation|natural transformations]] between [[functor|functors]] in the 2-category [[Cat]].