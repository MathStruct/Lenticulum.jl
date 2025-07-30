#definition #category-theory #colimits #computable #catlab

A **coproduct** of [[Object|objects]] $A$ and $B$ in a [[category]] is an object $A + B$ together with [[injection|injections]] $\iota_1: A \to A + B$ and $\iota_2: B \to A + B$ satisfying a [[universal property]].

## Universal Property

For any [[Object]] $C$ with [[morphism|morphisms]] $f: A \to C$ and $g: B \to C$, there exists a unique morphism $[f, g]: A + B \to C$ (called the **[[copairing]]**) such that:

- $[f, g] \circ \iota_1 = f$
- $[f, g] \circ \iota_2 = g$

<!-- \begin{tikzcd} A \arrow[r, "\iota_1"] \arrow[dr, "f"'] & A + B \arrow[d, "[f, g]", dashed] & B \arrow[l, "\iota_2"'] \arrow[dl, "g"] \\ & C & \end{tikzcd} -->

## Colimit Characterization

The coproduct is the [[colimit]] of the [[discrete category|discrete]] [[diagram]] consisting of objects $A$ and $B$.

## Binary vs General Coproducts

- **[[Binary coproduct]]**: Coproduct of two objects $A + B$
- **[[Finite coproduct]]**: Coproduct of finitely many objects
- **[[Initial object]]**: Empty coproduct (coproduct of zero objects)

## Examples

- In [[Set]]: Disjoint union of sets
- In [[Top]]: Disjoint union with coproduct topology
- In [[Grp]]: Free product of groups
- In [[Vec]]: Direct sum of vector spaces (also the product)
- In [[Bool]]: Logical OR operation

## Properties

- Coproducts are unique up to unique [[isomorphism]]
- Coproducts are [[associativity|associative]] and [[unitality|unital]] (with [[initial object]])
- Coproducts are [[duality|dual]] to [[product|products]]

Coproducts make a category into a [[cocartesian category]] and provide the [[tensor product]] for [[cocartesian monoidal category|cocartesian monoidal categories]].

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)