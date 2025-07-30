#definition #category-theory #limits #computable #catlab

A **product** of [[Object|objects]] $A$ and $B$ in a [[category]] is an object $A \times B$ together with [[projection|projections]] $\pi_1: A \times B \to A$ and $\pi_2: A \times B \to B$ satisfying a [[universal property]].

## Universal Property

For any [[Object]] $C$ with [[morphism|morphisms]] $f: C \to A$ and $g: C \to B$, there exists a unique morphism $\langle f, g \rangle: C \to A \times B$ (called the **[[pairing]]**) such that:

- $\pi_1 \circ \langle f, g \rangle = f$
- $\pi_2 \circ \langle f, g \rangle = g$

<!-- \begin{tikzcd} & C \arrow[dl, "f"'] \arrow[dr, "g"] \arrow[d, "\langle f, g \rangle", dashed] & \\ A & A \times B \arrow[l, "\pi_1"] \arrow[r, "\pi_2"'] & B \end{tikzcd} -->

## Limit Characterization

The product is the [[limit]] of the [[discrete category|discrete]] [[diagram]] consisting of objects $A$ and $B$.

## Binary vs General Products

- **[[Binary product]]**: Product of two objects $A \times B$
- **[[Finite product]]**: Product of finitely many objects
- **[[Terminal object]]**: Empty product (product of zero objects)

## Examples

- In [[Set]]: Cartesian product of sets
- In [[Top]]: Product topology on $X \times Y$
- In [[Grp]]: Direct product of groups
- In [[Vec]]: Direct sum of vector spaces (also the coproduct)

## Properties

- Products are unique up to unique [[isomorphism]]
- Products are [[associativity|associative]] and [[unitality|unital]] (with [[terminal object]])
- Not all categories have products

Products make a category into a [[cartesian category]] and provide the [[tensor product]] for [[cartesian monoidal category|cartesian monoidal categories]].

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)