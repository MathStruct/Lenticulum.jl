#definition #category-theory #set-theory #computable #catlab

**Set** is the [[category]] of sets and functions, serving as the fundamental example of a [[category]] and the foundation for much of categorical reasoning.

## Structure

- **[[Object|Objects]]**: Sets (collections of elements)
- **[[Morphism|Morphisms]]**: Functions $f: A \to B$ between sets
- **[[Composition]]**: Function composition $(g \circ f)(x) = g(f(x))$
- **[[Identity morphism|Identities]]**: Identity function $\text{id}_A: A \to A$ given by $\text{id}_A(x) = x$

## Categorical Properties

**Set** is:

- **[[Complete category|Complete]]**: Has all small [[limit|limits]]
- **[[Cocomplete category|Cocomplete]]**: Has all small [[colimit|colimits]]
- **[[Cartesian category|Cartesian]]**: Has all finite [[product|products]]
- **[[Cartesian closed category|Cartesian closed]]**: Has [[exponential object|exponential objects]]
- **[[Topos]]**: Elementary topos with [[subobject classifier]]

## Specific Constructions

- **[[Product]]**: Cartesian product $A \times B$
- **[[Coproduct]]**: Disjoint union $A \sqcup B$
- **[[Terminal object]]**: Singleton set ${*}$
- **[[Initial object]]**: Empty set $\emptyset$
- **[[Pullback]]**: Fiber product ${(a,b) \in A \times B : f(a) = g(b)}$
- **[[Pushout]]**: Quotient of disjoint union by equivalence relation

<!-- \begin{tikzcd} A \arrow[r, "f"] & B \\ C \arrow[r, "g"'] & D \end{tikzcd} Functions in Set -->

## Universal Properties

Set satisfies many [[universal property|universal properties]], making it the archetypal example for understanding categorical concepts.

## Role in Category Theory

- **[[Hom-functor]]**: $\text{Hom}(-,S): \mathcal{C}^{\text{op}} \to \text{Set}$
- **[[Representable functor|Representable functors]]**: Functors naturally isomorphic to hom-functors
- **[[Yoneda lemma]]**: Embedding any category into $[\mathcal{C}^{\text{op}}, \text{Set}]$

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)