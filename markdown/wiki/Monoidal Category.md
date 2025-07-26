#definition #category-theory #monoidal #computable #catlab

A **monoidal category** is a [[category]] $\mathcal{C}$ equipped with a [[tensor product]] [[bifunctor]] $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$, a [[unit object]] $I$, and [[natural isomorphism|natural isomorphisms]] for [[associativity]] and [[unitality]] satisfying coherence conditions.

## Structure

- **[[Tensor product]]**: Bifunctor $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$
- **[[Unit object]]**: Object $I \in \mathcal{C}$
- **[[Associator]]**: Natural isomorphism $\alpha_{A,B,C}: (A \otimes B) \otimes C \cong A \otimes (B \otimes C)$
- **[[Left unitor]]**: Natural isomorphism $\lambda_A: I \otimes A \cong A$
- **[[Right unitor]]**: Natural isomorphism $\rho_A: A \otimes I \cong A$

## Coherence Conditions

**[[Pentagon axiom]]**: The associator satisfies the pentagon identity relating different ways of associating four objects.

**[[Triangle axiom]]**: The unitors and associator are compatible on objects of the form $I \otimes A \otimes I$.

<!-- \begin{tikzcd} A \otimes B \arrow[r, "\otimes"] & (A \otimes B) \otimes C \arrow[d, "\alpha"] \\ & A \otimes (B \otimes C) \end{tikzcd} -->

## Examples

- **[[Cartesian monoidal category]]**: $(\mathcal{C}, \times, 1)$ with [[product]] as tensor
- **[[Cocartesian monoidal category]]**: $(\mathcal{C}, +, 0)$ with [[coproduct]] as tensor
- **[[Symmetric monoidal category]]**: Monoidal category with [[braiding]]
- **[[Closed monoidal category]]**: Monoidal category with [[internal hom]]

## Applications

Monoidal categories provide the foundation for [[monoidal functor|monoidal functors]], [[optic|optics]], and [[string diagram|string diagrams]].

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/)