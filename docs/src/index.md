```@meta
CurrentModule = Lenticulum
```

# Lenticulum.jl

Implicit machine learning on factor graphs: learning **relations**
``R_\theta \subseteq X_1\times\cdots\times X_n`` rather than functions
``f_\theta : X \to Y``.

!!! warning "Under development"
    Interfaces are not stable. Several packages document known gaps rather than hiding them;
    the per-package pages say what is missing.

## The idea in one page

An ordinary neural network is a function. You feed it an input, you get an output, and the
direction is fixed when you build the layer. Lenticulum's unit of computation is a **factor**:
a relation among several named channels, with no distinguished input or output. Which channels
are inputs is decided *per call*, and the same factor can be run in whichever direction the
graph needs.

```julia
# a Lux layer knows which side is the input
Dense(3 => 5)

# a factor does not — you tell it, when you use it
GaussianFactor(1 => 1; noise = Q, channels = (:x, :y))
```

Factors are wired into a **factor graph** — a bipartite graph of factors and variables — and
inference is **message passing** rather than a forward pass. That is the whole design:

| | Lux.jl | Lenticulum.jl |
|---|---|---|
| unit | a layer: a function | a factor: a relation |
| direction | fixed at construction | chosen per message |
| wiring | a DAG | any connected graph |
| running it | one forward pass | scheduled messages until convergence |
| backward pass | a gradient | a posterior belief |

## The packages

The project is one umbrella package over five smaller ones. Most users need the top two rows.

| package | what it gives you |
|---|---|
| [`LenticulumCore`](@ref lenticulumcore) | what a **factor** is: channels, polarities, beliefs, energies |
| [`Mycelium`](@ref mycelium) | how factors are **wired and scheduled**: graphs, messages, free energy |
| [`Lenticulum`](@ref lenticulum) | the **linear-Gaussian** factors, and Gaussian beliefs |
| [`VariationalDiffusion`](@ref variationaldiffusion) | a **diffusion model** as a factor (VP-SDE, RED-Diff) |
| [`ImplicitLayers`](@ref implicitlayers) | **DEQs and neural ODEs** as factors |
| [`Adversarial`](@ref adversarial) | **implicit generative models** — generators and density ratios |

`LenticulumCore` and `Mycelium` between them define the framework; the other three are factor
libraries built on it, and each can be ignored if you do not need that model family.

## Installation

Not registered. From a clone:

```julia
using Pkg
Pkg.develop(path = "/path/to/Lenticulum.jl")

# the factor libraries live under lib/ and are separate packages
Pkg.develop(path = "/path/to/Lenticulum.jl/lib/Mycelium.jl")
Pkg.develop(path = "/path/to/Lenticulum.jl/lib/ImplicitLayers.jl")
```

Every package depends on **LuxCore**, not Lux, so wrapping a Lux model costs nothing and
pulls in nothing. Neither `Lux` nor any automatic-differentiation package is a dependency of
anything here.

## Where to go next

- **[Getting started](@ref getting-started)** — a complete worked example you can run:
  a robot's trajectory estimated from odometry and one GPS reading.
- **[Vocabulary](@ref vocabulary)** — the six words you need to read the API. Short.
- The per-package pages, for the reference documentation.

## Where the theory is

**Not here.** This repository is also an [Obsidian](https://obsidian.md) vault, and the
mathematics — the categorical foundations, the papers, the derivations, and an honest account
of what does not work — lives in `markdown/`, starting from `markdown/Index.md`. Per-file
implementation notes sit next to the source they describe, as `*.md` beside `*.jl`.

These docs describe **the code**. If a docstring below cites something in double brackets like
`[[Bethe Free Energy]]`, that is a link into the vault, not a broken link on this site.
