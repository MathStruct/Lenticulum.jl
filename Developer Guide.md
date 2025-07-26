# Lenticulum.jl Developer Guide

Welcome to Lenticulum.jl! This guide outlines the development principles and code organization for our categorical machine learning framework.

## Project Philosophy

Lenticulum.jl is built around **categorical abstractions first**. Every design decision should be motivated by category theory, with concrete implementations following naturally from abstract categorical structures.

## Code Organization

### `/src/AbstractTypes/` - The Categorical Foundation

The `AbstractTypes` module contains the fundamental categorical abstractions that define our framework's mathematical structure.

#### File Organization Rules

**One File Per Abstract Type**

```
/src/AbstractTypes/
├── ParametricLens.jl     # abstract type ParametricLens{P,S,T}
├── CategoryObject.jl     # abstract type CategoryObject
├── Functor.jl           # abstract type Functor{C,D}
└── NaturalTransformation.jl
```

Each file must be named **exactly** like the abstract type it defines. This strict naming convention enables future GraphRAG integration for automated code analysis and categorical relationship discovery.

#### Content Guidelines

**Abstract Methods Only**

```julia
# ✅ Good - in ParametricLens.jl
abstract type ParametricLens{P, S, T} end

"""
Forward transformation: extract target from source
"""
function get end

"""
Backward transformation: update source via target  
"""
function set end
```

**Model Type Hierarchies**

```julia
# ✅ Good - showing categorical relationships
abstract type CategoryObject end
abstract type VectorSpace{T} <: CategoryObject end
abstract type FiniteVectorSpace{T, N} <: VectorSpace{T} end
```

**CatLab.jl Integration** When equivalent types exist in CatLab.jl, reference them unless there's a specific reason not to:

```julia
# ✅ Good - leveraging existing categorical infrastructure
using Catlab.CategoricalAlgebra: FiniteCategory
# Our extension for ML-specific needs
abstract type MLCategory <: FiniteCategory end
```

#### What Belongs Here

- Abstract type definitions with categorical meaning
- Method signatures (no implementations)
- Type hierarchies reflecting categorical relationships
- Documentation explaining categorical motivation
- References to equivalent CatLab.jl types

#### What Does NOT Belong Here

- Concrete type implementations
- Method implementations
- Instance constructors
- Performance optimizations

### `/src/Implementations/` - Concrete Realizations

Concrete implementations of abstract types belong in implementation-specific subfolders:

```
/src/Implementations/
├── Lenses/
│   ├── DenseLens.jl      # Dense <: ParametricLens
│   ├── ConvLens.jl       # Conv <: ParametricLens  
│   └── CompositeLens.jl  # composition operations
├── Functors/
│   ├── BatchFunctor.jl
│   └── FeatureMapFunctor.jl
└── Categories/
    ├── VectCategory.jl
    └── DiffCategory.jl
```

#### Implementation Structure

Concrete types should decompose into categorical components:

```julia
# Dense layer as categorical triple
struct Dense{T} <: ParametricLens{DenseParams{T}, Vector{T}, Vector{T}}
    linear::LinearMap{T}
    bias::BiasMap{T}  
    activation::ActivationMap{T}
end
```

### Additional Reasonable Guidelines

#### `/src/Utils/` - Categorical Utilities

```
/src/Utils/
├── Composition.jl    # categorical composition helpers
├── StringDiagrams.jl # CatLab.jl integration for visualization
└── TypeChecking.jl   # categorical law verification
```

#### `/src/Compilation/` - MLIR Integration

```
/src/Compilation/
├── ReactantBridge.jl # Reactant.jl integration
├── Optimization.jl   # categorical structure-preserving optimizations
└── Lowering.jl       # categorical → MLIR lowering passes
```

#### `/test/` - Categorical Law Testing

```
/test/
├── AbstractTypes/    # test categorical laws hold
├── Implementations/  # test concrete behavior
└── Integration/      # test CatLab.jl compatibility
```

## Development Principles

### 1. Category Theory First

Every abstraction should have clear categorical motivation. Ask:

- What category are we working in?
- What are the objects and morphisms?
- What laws must be satisfied?

### 2. Separation of Concerns

- **Abstract types**: Define categorical structure
- **Concrete types**: Provide computational implementations
- **Methods**: Bridge between categorical and computational

### 3. CatLab.jl Harmony

Leverage existing categorical infrastructure when possible. Only define new abstractions when:

- ML-specific extensions are needed
- Performance requirements demand specialized implementations
- Categorical laws need additional structure

### 4. GraphRAG Preparation

The strict file naming enables automated analysis:

- Type dependency graphs
- Categorical relationship extraction
- Automated documentation generation
- Semantic code search

### 5. Performance Consciousness

While abstractions come first, implementations must be performant:

- Use Julia's type system for zero-cost abstractions
- Leverage multiple dispatch for categorical polymorphism
- Ensure MLIR compilation compatibility

## Contributing Workflow

1. **Identify Categorical Need**: What categorical structure is missing?
2. **Check CatLab.jl**: Does equivalent abstraction exist?
3. **Define Abstract Type**: Create file in `/src/AbstractTypes/`
4. **Document Categorical Laws**: What properties must hold?
5. **Implement Concrete Types**: Create in appropriate `/src/` subfolder
6. **Test Categorical Properties**: Verify laws in `/test/`
7. **Update Knowledge Vault**: Add to Obsidian documentation

## Code Style

### Type Definitions

```julia
# ✅ Explicit type parameters with categorical meaning
abstract type ParametricLens{P<:Parameters, S<:Source, T<:Target} end

# ✅ Clear categorical documentation
"""
A parametric lens represents a bidirectional transformation in the 
category of parametric maps, satisfying the lens laws:
1. SetGet: get(set(lens, p, s, t), p, s) == t
2. GetSet: set(lens, p, s, get(lens, p, s)) == s  
3. SetSet: set(lens, p, set(lens, p, s, t1), t2) == set(lens, p, s, t2)
"""
```

### Method Signatures

```julia
# ✅ Abstract method with categorical meaning
"""
Categorical getter: extract target from source via parametric morphism
"""
function get(lens::ParametricLens{P,S,T}, params::P, source::S)::T where {P,S,T} end
```

## Exception: Glue Code

We acknowledge that some code will be "glue" connecting to external libraries or handling implementation details. Such code should:

- Be clearly marked as glue/utility code
- Live in `/src/Utils/` or appropriate integration folders
- Have minimal categorical pretensions
- Focus on practical interoperability

The goal is categorical purity where it matters, pragmatic flexibility where it doesn't.

---

*Remember: We're building a framework where category theory isn't just theoretical decoration—it's the fundamental organizing principle that makes complex ML architectures composable, verifiable, and performant.*
