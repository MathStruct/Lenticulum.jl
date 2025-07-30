#definition #category-theory #colimits #computable #catlab

A **colimit** of a [[diagram]] $D: \mathcal{J} \to \mathcal{C}$ is an [[Object]] $\text{colim } D$ together with a family of [[morphism|morphisms]] $\iota_j: D(j) \to \text{colim } D$ (called **[[injection|injections]]**) satisfying a [[universal property]].

## Universal Property

The colimit $\text{colim } D$ with injections $\iota_j: D(j) \to \text{colim } D$ satisfies:

For any [[Object]] $X$ with morphisms $f_j: D(j) \to X$ making all triangles commute (i.e., $f_k \circ D(u) = f_j$ for each morphism $u: j \to k$ in $\mathcal{J}$), there exists a unique morphism $[f_j]: \text{colim } D \to X$ such that $[f_j] \circ \iota_j = f_j$ for all $j$.

## Cocone Structure

A **[[cocone]]** under diagram $D$ with apex $X$ is a family of morphisms $f_j: D(j) \to X$ such that for every morphism $u: j \to k$ in $\mathcal{J}$, we have $f_k \circ D(u) = f_j$.

The colimit is the **initial** object in the [[category]] of cocones.

<!-- \begin{tikzcd} D(j) \arrow[r, "D(u)"] \arrow[dr, "f_j"'] \arrow[d, "\iota_j"] & D(k) \arrow[d, "\iota_k"] \arrow[dl, "f_k"] \\ \text{colim } D \arrow[r, "[f_j]", dashed] & X & \end{tikzcd} -->

## Examples

- **[[Coproduct]]**: Colimit of a [[discrete category|discrete]] diagram
- **[[Coequalizer]]**: Colimit of a parallel pair of morphisms
- **[[Pushout]]**: Colimit of a [[span]] diagram
- **[[Initial object]]**: Colimit of the empty diagram

## Properties

- Colimits are unique up to unique [[isomorphism]]
- A [[category]] **has colimits** if all small diagrams have colimits
- Colimits are preserved by [[left adjoint]] functors
- Colimits are [[duality|dual]] to [[limit|limits]]

## Cocompleteness

A category is **[[cocomplete category|cocomplete]]** if it has all small colimits.

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)