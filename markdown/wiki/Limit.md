#definition #category-theory #limits #computable #catlab

A **limit** of a [[diagram]] $D: \mathcal{J} \to \mathcal{C}$ is an [[object]] $\lim D$ together with a family of [[morphism|morphisms]] $\pi_j: \lim D \to D(j)$ (called **[[projection|projections]]**) satisfying a [[universal property]].

## Universal Property

The limit $\lim D$ with projections $\pi_j: \lim D \to D(j)$ satisfies:

For any [[object]] $X$ with morphisms $f_j: X \to D(j)$ making all triangles commute (i.e., $D(u) \circ f_j = f_k$ for each morphism $u: j \to k$ in $\mathcal{J}$), there exists a unique morphism $\langle f_j \rangle: X \to \lim D$ such that $\pi_j \circ \langle f_j \rangle = f_j$ for all $j$.

## Cone Structure

A **[[cone]]** over diagram $D$ with apex $X$ is a family of morphisms $f_j: X \to D(j)$ such that for every morphism $u: j \to k$ in $\mathcal{J}$, we have $D(u) \circ f_j = f_k$.

The limit is the **terminal** object in the [[category]] of cones.

<!--
\begin{tikzcd}
& X \arrow[dl, "f_j"'] \arrow[dr, "f_k"] \arrow[d, "\langle f_j \rangle", dashed] & \\
D(j) & \lim D \arrow[l, "\pi_j"] \arrow[r, "\pi_k"'] & D(k) \\
& & D(j) \arrow[u, "D(u)"']
\end{tikzcd}
-->

## Examples

- **[[Product]]**: Limit of a [[discrete category|discrete]] diagram
- **[[Equalizer]]**: Limit of a parallel pair of morphisms
- **[[Pullback]]**: Limit of a [[cospan]] diagram
- **[[Terminal object]]**: Limit of the empty diagram

## Properties

- Limits are unique up to unique [[isomorphism]]
- A [[category]] **has limits** if all small diagrams have limits
- Limits are preserved by [[right adjoint]] functors
- Limits are [[duality|dual]] to [[colimit|colimits]]

## Completeness

A category is **[[complete category|complete]]** if it has all small limits.

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)