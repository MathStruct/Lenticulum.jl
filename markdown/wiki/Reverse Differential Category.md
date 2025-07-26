#definition #category-theory #differential #automatic-differentiation #computable

A **reverse differential category** is a [[category]] equipped with a reverse derivative operation that axiomatizes the categorical structure underlying automatic differentiation and backpropagation in machine learning.

## Structure

A reverse differential category consists of:

- A [[cartesian category]] $\mathcal{C}$ with [[coproduct|coproducts]]
- A **reverse derivative operator** $R[-]: \mathcal{C}(A, B) \to \mathcal{C}(A \times B, A)$
- Satisfying coherence conditions for the reverse mode of automatic differentiation

## Reverse Derivative Operation

For a [[morphism]] $f: A \to B$, the reverse derivative $R[f]: A \times B \to A$ captures:

- **Forward pass**: Computing $f(a)$
- **Backward pass**: Computing the "gradient" with respect to the input

## Key Properties

A category with a reverse derivative also has a forward derivative, but the converse is not true. In fact, we show explicitly what a forward derivative is missing: a reverse derivative is equivalent to a forward derivative with a dagger structure on its subcategory of linear maps.

## Relationship to Forward Derivatives

- **[[Differential category]]**: Has forward derivative $D[-]$
- **Reverse differential category**: Has both forward and reverse derivatives
- **Additional structure**: Reverse derivative requires a [[dagger category|dagger structure]] on linear maps

## Applications

- **[[Automatic differentiation]]**: Categorical semantics for reverse-mode AD
- **[[Machine learning]]**: Backpropagation algorithms
- **[[Optimization]]**: Gradient-based methods

## Examples

- **[[Smooth manifold|Smooth manifolds]]** with reverse-mode differentiation
- **[[Neural network]]** categories with backpropagation
- **[[Probabilistic programming]]** with gradient estimation

<!-- \begin{tikzcd} A \times B \arrow[r, "R[f]"] & A \\ A \arrow[u, "\langle \text{id}, f \rangle"] \arrow[r, "f"'] & B \arrow[u, "\text{id} \times \text{id}"] \end{tikzcd} -->

Reverse differential categories provide the theoretical foundation for understanding automatic differentiation from a categorical perspective.