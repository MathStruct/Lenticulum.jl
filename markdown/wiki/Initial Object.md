#definition #category-theory #limits #computable #catlab

An **initial object** in a [[category]] $\mathcal{C}$ is an [[Object]] $0$ such that for every object $A$ in $\mathcal{C}$, there exists a unique [[morphism]] $!_A: 0 \to A$.

## Universal Property

An object $0$ is initial if and only if the [[hom-set]] $\text{Hom}(0, A)$ has exactly one element for every object $A$.

## Uniqueness

Initial objects are unique up to unique [[isomorphism]]. If $0$ and $0'$ are both initial, then there exists a unique isomorphism $0 \cong 0'$.

## Colimit Characterization

The initial object is the [[colimit]] of the empty [[diagram]], making it the empty [[coproduct]].

<!-- \begin{tikzcd} 0 \arrow[r, "!_A"] \arrow[dr, "!_B"'] & A \\ & B \end{tikzcd} -->

## Examples

- In [[Set]]: The empty set $\emptyset$
- In [[Grp]]: The trivial group ${e}$
- In [[Ring]]: The zero ring ${0}$
- In [[Top]]: The empty topological space
- In [[Poset]]: The least element (if it exists)

## Properties

- **[[Monomorphism]]**: Every morphism from an initial object is a monomorphism
- **[[Strict initial object]]**: In categories with zero morphisms, initial = terminal implies zero object
- **Preservation**: Initial objects are preserved by [[left adjoint]] functors

## Duality

The [[duality|dual]] concept is the [[terminal object]], where morphisms go **to** the object rather than **from** it.

## Applications

Initial objects appear in:

- **[[Coproduct]]** constructions as the empty case
- **[[Free object]]** constructions as the free object on the empty set
- **[[Colimit]]** computations as base cases

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)