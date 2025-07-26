# Lenticulum.jl: A Categorical Machine Learning Framework

**A category theory-inspired ML framework for implicit learning with MLIR compilation**

---

## Abstract

Lenticulum.jl introduces a novel machine learning framework that fundamentally reimagines neural computation through the lens of category theory. Unlike traditional frameworks that model learning as explicit parameterized functions $f_\theta: X \rightarrow Y$, Lenticulum embraces implicit learning via parameterized relations $(x, y) \in \mathcal{R}_\theta(X,Y)$. Built on Julia's multiple dispatch system and compiled via MLIR through Reactant.jl, Lenticulum achieves both mathematical rigor and C-like performance while maintaining categorical compositionality.

The framework centers around **Parametric Lenses** as fundamental building blocks, providing bidirectional transformations that naturally encode both forward computation and gradient backpropagation within a unified categorical structure. This approach enables principled handling of constraints, differential equations, and geometric relationships that traditional explicit learners struggle with.

---

## 1. Motivation

### The Learning-by-Doing Philosophy

Traditional study of category theory often leads to understanding concepts in isolation, only to forget them when they appear in different contexts. Lenticulum.jl embodies a **learning-by-doing** approach: implementing categorical concepts in a practical ML framework to achieve deep, lasting understanding through hands-on experience.

### Limitations of Current Approaches

**Explicit Learning Paradigm**: Most ML frameworks assume learning parameterized functions $f_\theta: X \rightarrow Y$. While computationally convenient, this paradigm struggles with:
- Systems governed by differential equations
- Geometric constraints and symmetries  
- Algebraic relationships between variables
- Multi-modal or relational data structures

**Performance vs. Expressiveness Trade-off**: Existing solutions force a choice between mathematical expressiveness (Haskell-based categorical frameworks) and computational performance (traditional ML frameworks). No current framework provides both categorical rigor and C-like speed.

**Lack of Categorical Foundations**: Most ML frameworks treat composition, parallelism, and transformations as ad-hoc operations rather than principled categorical constructions.

---

## 2. Core Innovation: Implicit Learning

### From Functions to Relations

Lenticulum introduces **implicit learning**, where instead of learning functions directly:

$$f_\theta: X \rightarrow Y$$

We learn parameterized relations:

$$(x, y) \in \mathcal{R}_\theta(X, Y)$$

This is implemented by learning an error function:

$$r_\theta: X \times Y \rightarrow \mathcal{E}$$

where the goal is to minimize:

$$r_\theta(x, y) = 0$$

### Advantages of Implicit Learning

**Constraint Conformity**: Implicit learners naturally encode:
- Conservation laws in physics
- Geometric invariants
- Differential equation constraints
- Algebraic relationships

**Parameter Efficiency**: Relations can represent complex mappings with fewer parameters than equivalent explicit functions.

**Mathematical Naturality**: Many real-world phenomena are inherently relational rather than functional.

---

## 3. Categorical Foundations

### Core Concepts in Lenticulum

**Categories & Morphisms**
- Objects: Data types, vector spaces, probability distributions
- Morphisms: Parametric lenses, layer transformations, relation constraints

**Parametric Lenses**
- Bidirectional transformations with `get` (forward) and `set` (backward) operations
- Satisfy lens laws ensuring categorical consistency
- Natural encoding of backpropagation within categorical framework

**Functors**
- Model architectural transformations (batching, reshaping, feature maps)
- Preserve categorical structure across different domains

**Monoidal Structure**
- Parallel composition of networks
- Tensor operations as categorical products
- Natural handling of multi-input/output architectures

**String Diagrams**
- Visual representation of network architectures
- Intuitive design and reasoning about categorical compositions

### The Lenticulum Category

Lenticulum defines a category **Lens** where:
- **Objects**: Types equipped with parameter spaces
- **Morphisms**: Parametric lenses `ParametricLens{P, S, T}`
- **Composition**: Sequential application of lenses
- **Identity**: Identity lens preserving structure

---

## 4. Technical Architecture

### Julia + MLIR: The Perfect Match

**Multiple Dispatch**: Julia's type system naturally encodes categorical relationships through method specialization based on type signatures.

**MLIR Compilation**: Via Reactant.jl, Lenticulum compiles to MLIR (Multi-Level Intermediate Representation), enabling:
- Near C-like performance
- Automatic parallelization across CPU/GPU/TPU
- Advanced compiler optimizations
- Integration with modern accelerator ecosystems

**Type Safety**: Rich type system enables encoding of categorical laws and constraints at compile time.

### Core Abstractions

```julia
abstract type ParametricLens{P, S, T} end

# Forward pass: extract target from source
get(lens::ParametricLens{P, S, T}, params::P, source::S) -> T

# Backward pass: update source via target
set(lens::ParametricLens{P, S, T}, params::P, source::S, target::T) -> S
```

**DenseLens**: Linear transformations as bidirectional lenses
**CompositeLens**: Categorical composition of multiple lenses  
**ProductLens**: Parallel application via monoidal products
**ImplicitLens**: Relation-based learning via error minimization

---

## 5. Integration Ecosystem

### CatLab.jl Integration
- Leverage existing categorical abstractions
- String diagram visualization
- Automated categorical construction

### Enzyme.jl & Reactant.jl
- Automatic differentiation through categorical structure
- MLIR compilation for performance
- Hardware acceleration support

### BenchmarkTools.jl Testing
- Performance comparison against established frameworks
- Ensuring competitive speed while maintaining categorical rigor

### Future: Lean.jl Integration
- Formal verification of categorical laws
- Type-level proofs of network properties
- Mathematically guaranteed correctness

---

## 6. Applications & Use Cases

### Differential Equations
```julia
# Learning implicit solutions to ODEs
ode_lens = ImplicitLens((x, y, dy) -> r_θ(x, y, dy))
# Minimizes: dy/dx - f(x, y) = 0
```

### Physics-Informed Learning
```julia
# Conservation law constraints
physics_lens = ConstraintLens(conservation_relations)
model = compose(physics_lens, neural_lens)
```

### Geometric Deep Learning
```julia
# Equivariant transformations via functors
equivariant_model = FunctorLens(symmetry_group, base_lens)
```

### Multi-Modal Reasoning
```julia
# Relational learning across modalities
relation_lens = ImplicitLens((text, image, label) -> alignment_error)
```

---

## 7. Roadmap

### Phase 1: Foundations ✓
- [x] Core `ParametricLens` abstraction
- [x] Basic lens implementations (`DenseLens`)
- [x] Integration with Reactant.jl/Enzyme.jl

### Phase 2: Categorical Structure
- [ ] Lens composition and identity
- [ ] Monoidal products and coproducts  
- [ ] Functorial transformations
- [ ] String diagram integration with CatLab.jl

### Phase 3: Implicit Learning
- [ ] `ImplicitLens` implementation
- [ ] Constraint satisfaction mechanisms
- [ ] Differential equation solvers
- [ ] Physics-informed architectures

### Phase 4: Advanced Features
- [ ] Higher-order categorical constructions
- [ ] Formal verification via Lean.jl
- [ ] Industrial-scale optimizations
- [ ] Ecosystem integrations

### Phase 5: Applications
- [ ] Scientific computing benchmarks
- [ ] Geometric deep learning examples
- [ ] Multi-modal architectures
- [ ] Real-world case studies

---

## 8. Why Lenticulum?

**Theoretical Rigor**: Grounded in solid mathematical foundations via category theory, ensuring compositionality and principled design.

**Practical Performance**: MLIR compilation delivers production-ready speed without sacrificing expressiveness.

**Novel Paradigm**: Implicit learning opens new possibilities for constraint-aware, relation-based machine learning.

**Learning Platform**: Serves as a hands-on laboratory for understanding categorical concepts through implementation.

**Community Building**: Bridges the gap between category theorists and ML practitioners, fostering cross-pollination of ideas.

---

## 9. Getting Started

```julia
using Pkg
Pkg.add("https://github.com/yourusername/Lenticulum.jl")

using Lenticulum
using Reactant, Enzyme

# Create a categorical neural network
lens1 = DenseLens(784, 128)
lens2 = DenseLens(128, 10)
model = compose(lens1, lens2)

# Train via categorical backpropagation
params = initialize_parameters(model)
trained_params = categorical_train(model, params, data)
```

---

## 10. Contributing

Lenticulum.jl is designed as a learning platform. Contributions that implement categorical concepts, add theoretical depth, or improve performance are highly welcomed. Each implementation should:

- Include clear categorical motivation
- Provide mathematical background
- Maintain performance standards
- Include comprehensive tests

---

## References

- Fong, B., & Spivak, D. I. (2018). *Seven sketches in compositionality: An invitation to applied category theory*
- Cruttwell, G. S., et al. (2022). *Categorical foundations of gradient-based learning*
- Milewski, B. (2019). *Category Theory for Programmers*
- Various papers on categorical machine learning and differentiable programming

---

**Lenticulum.jl**: *Where Category Theory Meets Machine Learning*

*"Learning by doing, one lens at a time."*
