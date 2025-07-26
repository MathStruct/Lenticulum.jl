#definition #category-theory #limits #computable #catlab

A **terminal object** in a [[category]] $\mathcal{C}$ is an [[object]] $1$ such that for every object $A$ in $\mathcal{C}$, there exists a unique [[morphism]] $!_A: A \to 1$.

## Universal Property

An object $1$ is terminal if and only if the [[hom-set]] $\text{Hom}(A, 1)$ has exactly one element for every object $A$.

## Uniqueness

Terminal objects are unique up to unique [[isomorphism]]. If $1$ and $1'$ are both terminal, then there exists a unique isomorphism $1 \cong 1'$.

## Limit Characterization

The terminal object is the [[limit]] of the empty [[diagram]], making it the empty [[product]].

<!-- \begin{tikzcd} A \arrow[dr, "!_A"] & \\ & 1 \\ B \arrow[ur, "!_B"'] & \end{tikzcd} -->

## Examples

- In [[Set]]: Any singleton set ${*}$
- In [[Grp]]: The trivial group ${e}$
- In [[Ring]]: The zero ring ${0}$
- In [[Top]]: Any singleton topological space
- In [[Poset]]: The greatest element (if it exists)
- In [[Cat]]: The terminal category $\mathbf{1}$ with one object and one morphism

## Properties

- **[[Epimorphism]]**: Every morphism to a terminal object is an epimorphism
- **[[Product]] unit**: Terminal object serves as the unit for products: $A \times 1 \cong A$
- **Preservation**: Terminal objects are preserved by [[right adjoint]] functors

## Duality

The [[duality|dual]] concept is the [[initial object]], where morphisms come **from** the object rather than **to** it.

## Applications

Terminal objects appear in:

- **[[Product]]** constructions as the empty case
- **[[Limit]]** computations as base cases
- **[[Cartesian category|Cartesian categories]]** as the multiplicative unit

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)