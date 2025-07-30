#definition #category-theory #computable #catlab

A **diagram** in a [[category]] $\mathcal{C}$ is a [[functor]] $D: \mathcal{J} \to \mathcal{C}$ from a small [[category]] $\mathcal{J}$ (called the **index category** or **shape**) to $\mathcal{C}$.

## Formal Definition

Given categories $\mathcal{J}$ and $\mathcal{C}$, a diagram of shape $\mathcal{J}$ in $\mathcal{C}$ consists of:

- An [[Object]] $D(j) \in \mathcal{C}$ for each object $j \in \mathcal{J}$
- A [[morphism]] $D(f): D(j) \to D(k)$ for each morphism $f: j \to k$ in $\mathcal{J}$
- Preservation of [[composition]] and [[identity morphism|identities]]

## Common Shapes

- **[[Discrete category]]**: Collection of objects with no morphisms
- **[[Walking arrow]]**: Two objects with one non-identity morphism
- **[[Walking cospan]]**: Three objects $A \leftarrow B \rightarrow C$
- **[[Walking span]]**: Three objects $A \rightarrow B \leftarrow C$

## Examples

**Triangle diagram** (shape $\bullet \rightarrow \bullet \leftarrow \bullet$):

<!-- \begin{tikzcd} & B \arrow[dl, "g"'] \arrow[dr, "h"] & \\ A & & C \end{tikzcd} -->

**Square diagram** (shape with four objects and four morphisms):

<!-- \begin{tikzcd} A \arrow[r, "f"] \arrow[d, "g"'] & B \arrow[d, "h"] \\ C \arrow[r, "k"'] & D \end{tikzcd} -->

## Properties

- Diagrams can be [[commutative diagram|commutative]] or non-commutative
- [[Limit|Limits]] and [[colimit|colimits]] are defined relative to diagrams
- Diagrams of different shapes capture different mathematical structures

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/categorical_algebra/)