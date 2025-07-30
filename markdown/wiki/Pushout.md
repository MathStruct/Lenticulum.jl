#definition #category-theory #colimits #computable #catlab

A **pushout** of a [[span]] $A \xleftarrow{f} C \xrightarrow{g} B$ is the [[colimit]] of this diagram, consisting of an [[Object]] $A +_C B$ with [[morphism|morphisms]] $i_1: A \to A +_C B$ and $i_2: B \to A +_C B$ such that $i_1 \circ f = i_2 \circ g$.

## Universal Property

For any [[Object]] $X$ with morphisms $u: A \to X$ and $v: B \to X$ such that $u \circ f = v \circ g$, there exists a unique morphism $[u, v]: A +_C B \to X$ such that:

- $[u, v] \circ i_1 = u$
- $[u, v] \circ i_2 = v$

<!-- \begin{tikzcd} & C \arrow[dl, "f"'] \arrow[dr, "g"] & \\ A \arrow[d, "i_1"] \arrow[drr, "u", bend left] & & B \arrow[d, "i_2"'] \arrow[dll, "v"', bend right] \\ A +_C B \arrow[rr, "[u, v]", dashed] & & X \end{tikzcd} -->

## Commutative Square

The pushout forms a [[commutative diagram|commutative square]]: $$i_1 \circ f = i_2 \circ g$$

## Alternative Names

- **Amalgamated sum**: Especially in algebra
- **Cocartesian square**: When viewed as a cocartesian morphism
- **Fiber coproduct**: In some contexts

## Examples

- In [[Set]]: Quotient of disjoint union $A \sqcup B$ by identifying $f(c) \sim g(c)$ for all $c \in C$
- In [[Top]]: Quotient topology on the set-theoretic pushout
- In [[Grp]]: Amalgamated free product of groups
- In [[Cat]]: Gluing categories along a common subcategory

## Properties

- Pushouts are unique up to unique [[isomorphism]]
- Pushouts compose: pushout of pushout is pushout
- [[Epimorphism|Epimorphisms]] are stable under pushout
- Pushouts are preserved by [[left adjoint]] functors

## Duality

The [[duality|dual]] of pushout is [[pullback]], which is the [[limit]] of a [[cospan]] diagram.

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)