 #definition #category-theory #computable #catlab

A **monad** is an [[endofunctor]] equipped with two [[natural transformation|natural transformations]] that satisfy certain coherence conditions, providing a categorical framework for composition and sequencing.

## Formal Definition

A **monad** on a [[category]] $\mathcal{C}$ is a triple $(T, \eta, \mu)$ where:

- $T: \mathcal{C} \to \mathcal{C}$ is an [[endofunctor]]
- $\eta: \text{Id}_{\mathcal{C}} \Rightarrow T$ is a [[natural transformation]] called the **unit**
- $\mu: T^2 \Rightarrow T$ is a natural transformation called the **multiplication**

Such that the following [[commutative diagram|commutative diagrams]] hold:

### Associativity

<!-- \begin{tikzcd} T^3 \arrow[r, "T\mu"] \arrow[d, "\mu T"'] & T^2 \arrow[d, "\mu"] \\ T^2 \arrow[r, "\mu"'] & T \end{tikzcd} -->

### Left and Right Unit Laws

<!-- \begin{tikzcd} T \arrow[r, "\eta T"] \arrow[dr, "\text{id}"'] & T^2 \arrow[d, "\mu"] & T \arrow[l, "T\eta"'] \arrow[dl, "\text{id}"] \\ & T & \end{tikzcd} -->

## Component-wise Description

For each [[Object]] $X \in \mathcal{C}$:

- $\eta_X: X \to T(X)$ embeds $X$ into the monad
- $\mu_X: T(T(X)) \to T(X)$ flattens nested applications

The laws become:

1. **Associativity**: $\mu_X \circ T(\mu_X) = \mu_X \circ \mu_{T(X)}$
2. **Left unit**: $\mu_X \circ T(\eta_X) = \text{id}_{T(X)}$
3. **Right unit**: $\mu_X \circ \eta_{T(X)} = \text{id}_{T(X)}$

## Basic Examples

- **Identity monad**: $T = \text{Id}$, $\eta = \mu = \text{id}$
- **Maybe monad** on $\mathbf{Set}$: $T(X) = X \cup {\bot}$ with appropriate unit and multiplication
- **List monad** on $\mathbf{Set}$: $T(X) = X^*$ (finite sequences), $\eta_X(x) = [x]$, $\mu_X$ flattens lists
- **Power set monad**: $T = \mathcal{P}$, $\eta_X(x) = {x}$, $\mu_X$ takes unions

## Kleisli Category

Every monad $(T, \eta, \mu)$ on $\mathcal{C}$ gives rise to the **[[Kleisli category]]** $\mathcal{C}_T$ where:

- Objects are the same as in $\mathcal{C}$
- [[Morphism|Morphisms]] $X \to_T Y$ are morphisms $X \to T(Y)$ in $\mathcal{C}$
- [[Composition]] is given by $g \star f = \mu_Z \circ T(g) \circ f$ for $f: X \to T(Y)$ and $g: Y \to T(Z)$

## Eilenberg-Moore Category

The **[[Eilenberg-Moore category]]** $\mathcal{C}^T$ consists of:

- **Objects**: [[Algebra over a functor|T-algebras]] $(X, h: T(X) \to X)$ satisfying algebra laws
- **[[Morphism|Morphisms]]**: [[Homomorphism|T-algebra homomorphisms]]

## Properties

- **[[Adjunction]]**: Every monad arises from an adjunction, though not uniquely
- **[[Distributive law|Distributive laws]]**: Enable composition of monads
- **[[Monad transformer|Monad transformers]]**: Systematic ways to combine monads

## Applications

Monads are fundamental in:

- **[[Functional programming]]**: Modeling computational effects
- **[[Algebraic geometry]]**: [[Descent theory]] and [[stack|stacks]]
- **[[Homotopy theory]]**: [[Simplicial set|Simplicial sets]] and [[model category|model categories]]
- **[[Logic]]**: [[Modal logic]] and [[intuitionistic logic]]

Monads provide a unified framework for handling [[composition]], [[sequencing]], and [[computational effects]] across mathematics and computer science.