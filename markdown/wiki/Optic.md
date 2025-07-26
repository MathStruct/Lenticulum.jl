#definition #category-theory #optics #computable

An **optic** is a bidirectional data accessor that provides a uniform framework for accessing and updating substructures within larger data structures.

## Concrete Definition

An optic from $S$ to $T$ changing $A$ to $B$ consists of:

- **Forward transformation**: $S \to M \bullet A$ for some [[mixed optic|mixed]] structure $M$
- **Backward transformation**: $M \bullet B \to T$

where $\bullet$ denotes the action of the [[monoidal structure]].

## Coend Representation

An optic is formally defined as: Optic(S,T,A,B)=∫MC(S,M∙A)×C(M∙B,T)\text{Optic}(S, T, A, B) = \int^M \mathbf{C}(S, M \bullet A) \times \mathbf{C}(M \bullet B, T)Optic(S,T,A,B)=∫MC(S,M∙A)×C(M∙B,T)

where the [[coend]] is taken over all objects $M$ in the [[monoidal category]].

## Profunctor Representation

An optic corresponds to a [[Tambara module]] for the monoidal structure, giving rise to the **profunctor optic** representation: ∀P.Tambara(P)⇒P(A,B)→P(S,T)\forall P. \text{Tambara}(P) \Rightarrow P(A, B) \to P(S, T)∀P.Tambara(P)⇒P(A,B)→P(S,T)

## Examples

- **[[Lens]]**: Optic for [[cartesian monoidal category|cartesian monoidal structure]]
- **[[Prism]]**: Optic for [[cocartesian monoidal category|cocartesian monoidal structure]]
- **[[Traversal]]**: Optic for [[applicative functor|applicative]] structure
- **[[Grate]]**: Optic for [[closed monoidal category|closed monoidal structure]]

<!-- \begin{tikzcd} S \arrow[r] \arrow[dr] & M \bullet A \arrow[d] & A \arrow[l] \\ & M \bullet B \arrow[r] & B \\ T \arrow[u] & & \end{tikzcd} -->

Optics support modularity, allowing construction of accessors for complex structures by combining simpler ones.