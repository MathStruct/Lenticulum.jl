#definition #category-theory #limits #computable #catlab

A **pullback** of a [[cospan]] $A \xrightarrow{f} C \xleftarrow{g} B$ is the [[limit]] of this diagram, consisting of an [[Object]] $A \times_C B$ with [[morphism|morphisms]] $p_1: A \times_C B \to A$ and $p_2: A \times_C B \to B$ such that $f \circ p_1 = g \circ p_2$.

## Universal Property

For any [[Object]] $X$ with morphisms $u: X \to A$ and $v: X \to B$ such that $f \circ u = g \circ v$, there exists a unique morphism $\langle u, v \rangle: X \to A \times_C B$ such that:

- $p_1 \circ \langle u, v \rangle = u$
- $p_2 \circ \langle u, v \rangle = v$

<!-- \begin{tikzcd} X \arrow[dr, "u"] \arrow[ddr, "v"', bend right] \arrow[dd, "\langle u, v \rangle", dashed] & & \\ & A \arrow[d, "f"] & \\ A \times_C B \arrow[r, "p_1"] \arrow[d, "p_2"'] & C & \\ B \arrow[ru, "g"'] & & \end{tikzcd} -->

## Commutative Square

The pullback forms a [[commutative diagram|commutative square]]: $$f \circ p_1 = g \circ p_2$$

## Alternative Names

- **Fiber product**: Especially in algebraic geometry
- **Cartesian square**: When viewed as a cartesian morphism
- **Inverse image**: In the context of inverse image functors

## Examples

- In [[Set]]: $A \times_C B = {(a, b) \in A \times B : f(a) = g(b)}$
- In [[Top]]: Subspace topology on the set-theoretic pullback
- In [[Cat]]: Comma category construction

## Properties

- Pullbacks are unique up to unique [[isomorphism]]
- Pullbacks compose: pullback of pullback is pullback
- [[Monomorphism|Monomorphisms]] are stable under pullback
- Pullbacks are preserved by [[right adjoint]] functors

## Duality

The [[duality|dual]] of pullback is [[pushout]], which is the [[colimit]] of a [[span]] diagram.

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)