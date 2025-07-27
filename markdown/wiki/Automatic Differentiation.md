#definition #category-theory #differential-category #cartesian-differential #computable

**Automatic differentiation** from a categorical perspective is the systematic computation of [[derivative|derivatives]] using the structure of [[cartesian differential category|cartesian differential categories]], providing a compositional framework for [[differentiation]] that preserves the categorical structure of computations.

## Formal Definition

Let $(\mathcal{C}, \otimes, I, \text{D})$ be a [[cartesian differential category]]. **Automatic differentiation** is a [[functor|functorial]] process that assigns to each [[morphism]] $f: A \to B$ its [[derivative]] $Df: A \otimes A \to B$ such that:

1. **Linearity**: $D$ is linear in the second argument
2. **Product rule**: $D(f \otimes g) = (Df \otimes \text{id}) \circ \delta + (\text{id} \otimes Dg) \circ \delta$
3. **Chain rule**: $D(g \circ f) = Dg \circ (f \otimes \text{id}) \circ Df$

where $\delta: A \to A \otimes A$ is the [[diagonal morphism]].

## Cartesian Differential Categories

A **[[cartesian differential category]]** extends a [[cartesian category]] with:

- A differential combinator $\text{D}: \mathcal{C}(A, B) \to \mathcal{C}(A \otimes A, B)$
- [[Chain rule]] and [[product rule]] as categorical laws
- [[Zero morphism|Zero]] and [[addition]] structure on [[hom-set|hom-sets]]

## Forward-Mode Differentiation

**Forward-mode** corresponds to the [[comonad]] structure on [[tangent bundle|tangent spaces]]:

### Tangent Bundle Functor

$$T: \mathcal{C} \to \mathcal{C}$$ $$T(A) = A \times A$$ $$T(f)(a, \dot{a}) = (f(a), Df(a, \dot{a}))$$

This forms a [[comonad]] with:

- **Counit**: $\epsilon_A: A \times A \to A = \pi_1$
- **Comultiplication**: $\delta_A: A \times A \to (A \times A) \times (A \times A)$

### Forward Differentiation Diagram

<!-- \begin{tikzcd} A \times A \arrow[r, "T(f)"] \arrow[d, "\pi_1"'] & B \times B \arrow[d, "\pi_1"] \\ A \arrow[r, "f"'] & B \end{tikzcd} -->

## Reverse-Mode Differentiation

**Reverse-mode** corresponds to the [[monad]] structure on [[cotangent bundle|cotangent spaces]]:

### Cotangent Bundle Functor

$$T^_: \mathcal{C}^{\text{op}} \to \mathcal{C}$$ $$T^_(A) = A \times (A \to \mathbb{R})$$

For $f: A \to B$, the reverse-mode transformation: $$T^_(f): T^_(B) \to T^*(A)$$ $$(b, \bar{b}) \mapsto (a, \lambda \dot{a}. \bar{b}(Df(a, \dot{a})))$$

where $a$ satisfies $f(a) = b$.

## Dual Numbers Construction

In the category $\mathbf{Ring}$ of [[ring|rings]], automatic differentiation uses [[dual number|dual numbers]]:

$$\mathbb{R}[\epsilon]/(\epsilon^2) = {a + b\epsilon : a, b \in \mathbb{R}, \epsilon^2 = 0}$$

For polynomial $f: \mathbb{R} \to \mathbb{R}$: $$f(a + \epsilon) = f(a) + f'(a)\epsilon$$

This extends to the [[functor]]: $$D: \mathbf{Ring} \to \mathbf{Ring}$$ $$D(R) = R[\epsilon]/(\epsilon^2)$$

## Computational Graph Categories

### Traced Monoidal Categories

Automatic differentiation in [[neural network|neural networks]] uses [[traced monoidal category|traced monoidal categories]] where:

- Objects: [[vector space|Vector spaces]] (layer dimensions)
- Morphisms: [[linear map|Linear maps]] (weight matrices)
- [[Trace]]: [[Backpropagation]] through cycles

### String Diagrams

[[Computation|Computational graphs]] are represented as [[string diagram|string diagrams]]:

<!-- \begin{tikzcd} & \bullet \arrow[dl] \arrow[dr] & \\ \bullet \arrow[dr] & & \bullet \arrow[dl] \\ & \bullet & \end{tikzcd} -->

## Higher-Order Derivatives

### Jet Bundle Categories

**Higher-order** automatic differentiation uses [[jet bundle|jet bundles]]: $$J^n(A, B) = \text{Hom}(\mathbb{R}[\epsilon]/(\epsilon^{n+1}), B^A)$$

The [[functor]] $J^n: \mathcal{C} \to \mathcal{C}$ captures $n$-th order [[Taylor expansion|Taylor approximations]].

### Faà di Bruno Formula

The categorical [[chain rule]] for higher derivatives follows the [[Faà di Bruno formula]]: $$D^n(g \circ f) = \sum_{k=1}^n D^k g \circ B_{n,k}(Df, D^2f, \ldots, D^{n-k+1}f)$$

where $B_{n,k}$ are [[Bell polynomial|Bell polynomials]].

## Smooth Categories

### Cartesian Closed Categories

In [[cartesian closed category|cartesian closed categories]] with [[exponential object|exponentials]], differentiation respects the [[internal hom]]: $$\text{D}(\lambda f) = \lambda(\text{D}f)$$

### Synthetic Differential Geometry

[[Synthetic differential geometry]] provides [[topos|topos-theoretic]] foundations where:

- [[Infinitesimal|Infinitesimals]] are [[nilpotent element|nilpotent]]
- [[Smooth function|Smooth functions]] are [[internal]] to the topos
- Differentiation is automatic from the [[axiom|axioms]]

## Implementation Categories

### Automatic Differentiation Monad

The AD computation forms a [[monad]] $M$ on $\mathbf{Vect}$: $$M(V) = V \times (V \to \mathbb{R})$$ $$\eta_V: V \to M(V) = v \mapsto (v, 0)$$ $$\mu_V: M(M(V)) \to M(V)$$ (flattening nested derivatives)

### Backpropagation Functor

[[Backpropagation]] is a [[contravariant functor]]: $$\text{Back}: \mathbf{Diff}^{\text{op}} \to \mathbf{Lin}$$ mapping [[differentiable function|differentiable functions]] to their [[gradient]] computations.

## Applications

### Machine Learning

- [[Neural network]] training via [[gradient descent]]
- Automatic computation of [[Jacobian matrix|Jacobians]] and [[Hessian matrix|Hessians]]
- [[Variational inference]] and [[optimization]]

### Scientific Computing

- [[Sensitivity analysis]] in [[dynamical system|dynamical systems]]
- [[Optimal control]] problems
- [[Partial differential equation|PDE]] solving with gradient methods

### Programming Languages

- [[Differentiable programming]] languages
- [[Type system|Type systems]] for [[AD]] transformations
- [[Compiler|Compiler]] optimizations for gradient computation

## Properties

### Functoriality

Automatic differentiation preserves:

- [[Composition]] of functions
- [[Product]] and [[sum]] structures
- [[Limit|Limits]] and [[colimit|colimits]] (in appropriate senses)

### Complexity

- **Forward mode**: $O(n)$ for functions $\mathbb{R}^n \to \mathbb{R}$
- **Reverse mode**: $O(m)$ for functions $\mathbb{R} \to \mathbb{R}^m$
- **Mixed mode**: Optimal for functions $\mathbb{R}^n \to \mathbb{R}^m$

Automatic differentiation demonstrates how [[category theory]] provides both theoretical foundations and practical computational frameworks, unifying mathematical concepts with algorithmic implementation in [[differentiable programming]] and [[machine learning]].