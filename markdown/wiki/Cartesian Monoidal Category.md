#definition #category-theory #monoidal #cartesian #computable #catlab

A **cartesian monoidal category** is a [[monoidal category]] where the [[tensor product]] is given by the [[product]] and the [[unit object]] is the [[terminal object]].

## Structure

- **[[Tensor product]]**: $A \otimes B = A \times B$ (categorical [[product]])
- **[[Unit object]]**: $I = 1$ (the [[terminal object]])
- **[[Associator]]**: Canonical isomorphism $(A \times B) \times C \cong A \times (B \times C)$
- **[[Left unitor]]**: Canonical isomorphism $1 \times A \cong A$
- **[[Right unitor]]**: Canonical isomorphism $A \times 1 \cong A$

## Additional Structure

Every cartesian monoidal category has:

- **[[Diagonal morphism]]**: $\Delta_A: A \to A \times A$ given by $\langle \text{id}_A, \text{id}_A \rangle$
- **[[Projection|Projections]]**: $\pi_1: A \times B \to A$ and $\pi_2: A \times B \to B$
- **[[Deletion morphism]]**: $!_A: A \to 1$ (unique morphism to terminal object)

## Cartesian Structure

The presence of diagonal and deletion morphisms makes cartesian monoidal categories into [[cartesian category|cartesian categories]], giving them the structure of a [[commutative comonoid]] on every object.

<!-- \begin{tikzcd} A \arrow[r, "\Delta_A"] \arrow[dr, "!_A"'] & A \times A \arrow[d, "\pi_1", shift left] \arrow[d, "\pi_2"', shift right] \\ & A & 1 \end{tikzcd} -->

## Examples

- **[[Set]]**: Sets with cartesian product and singleton set
- **[[FinSet]]**: Finite sets with cartesian product
- **[[Top]]**: Topological spaces with product topology
- **[[Grp]]**: Groups with direct product

## Applications

Cartesian monoidal categories are the natural setting for:

- **[[Lens|Lenses]]**: Optics for the cartesian tensor product
- **[[Cartesian closed category|Cartesian closed categories]]**: With exponential objects
- **[[Simply typed lambda calculus]]**: Computational interpretation

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)