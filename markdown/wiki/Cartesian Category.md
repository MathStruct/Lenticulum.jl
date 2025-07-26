#definition #category-theory #cartesian #computable #catlab

A **cartesian category** is a [[category]] with all finite [[product|products]], including a [[terminal object]].

## Structure

- **[[Terminal object]]**: Object $1$ such that for every object $A$, there exists a unique [[morphism]] $!_A: A \to 1$
- **[[Binary product]]**: For any objects $A, B$, there exists a product $A \times B$ with [[projection|projections]] $\pi_1: A \times B \to A$ and $\pi_2: A \times B \to B$
- **[[Universal property]]**: For any object $C$ with morphisms $f: C \to A$ and $g: C \to B$, there exists a unique morphism $\langle f, g \rangle: C \to A \times B$ such that $\pi_1 \circ \langle f, g \rangle = f$ and $\pi_2 \circ \langle f, g \rangle = g$

## Cartesian Structure

The [[product]] operation makes every cartesian category into a [[cartesian monoidal category]] with:

- **[[Tensor product]]**: $A \otimes B = A \times B$
- **[[Unit object]]**: $I = 1$ (terminal object)
- **Natural [[associativity]] and [[unitality]]**

## Morphisms

- **[[Pairing]]**: $\langle f, g \rangle: C \to A \times B$
- **[[Projection|Projections]]**: $\pi_1: A \times B \to A$, $\pi_2: A \times B \to B$
- **[[Diagonal]]**: $\Delta_A: A \to A \times A$ given by $\langle \text{id}_A, \text{id}_A \rangle$

<!-- \begin{tikzcd} & C \arrow[dl, "f"'] \arrow[dr, "g"] \arrow[d, "\langle f, g \rangle", dashed] & \\ A & A \times B \arrow[l, "\pi_1"] \arrow[r, "\pi_2"'] & B \end{tikzcd} -->

## Examples

- **[[Set]]**: Category of sets with cartesian products
- **[[Top]]**: Category of topological spaces with product topology
- **[[Grp]]**: Category of groups with direct products

Cartesian categories provide the foundation for [[lens|lenses]] and [[cartesian closed category|cartesian closed categories]].

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)