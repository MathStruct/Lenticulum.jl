A **2-category** is a [[higher category theory|higher categorical]] structure consisting of:

- **Objects** (0-cells)
- **1-morphisms** (1-cells) between objects
- **2-morphisms** (2-cells) between parallel 1-morphisms

The structure satisfies the following axioms:

## Composition

- **Horizontal composition**: 2-morphisms can be composed along 1-morphisms
- **Vertical composition**: 2-morphisms between the same pair of 1-morphisms can be composed
- **1-morphism composition**: 1-morphisms compose associatively with identity 1-morphisms

## Identity structures

- Each object has an [[identity morphism|identity 1-morphism]]
- Each 1-morphism has an [[identity 2-morphism]]

## Coherence conditions

- The [[interchange law]] holds: horizontal and vertical composition of 2-morphisms are compatible
- Composition is associative and unital at both the 1-morphism and 2-morphism levels

<!-- Basic 2-category structure: \begin{tikzcd} A \arrow[r, "f", bend left] \arrow[r, "g"', bend right] & B \end{tikzcd} with 2-morphism: $\alpha: f \Rightarrow g$ -->

Examples include [[Cat]] (the 2-category of categories), [[Gpd]] (the 2-category of [[groupoid|groupoids]]), and any [[monoidal category]] viewed as a one-object 2-category.