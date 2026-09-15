# Lenticulum.jl

[![CI](https://github.com/MathStruct/Lenticulum.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/MathStruct/Lenticulum.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Docs](https://github.com/MathStruct/Lenticulum.jl/actions/workflows/Docs.yml/badge.svg?branch=master)](https://MathStruct.github.io/Lenticulum.jl/dev/)

**Implicit** machine learning: learning **relations** instead of functions.
Under development.

| | API documentation | Theory vault |
|---|---|---|
| **read it at** | [MathStruct.github.io/Lenticulum.jl/dev](https://MathStruct.github.io/Lenticulum.jl/dev/) | […/dev/vault](https://MathStruct.github.io/Lenticulum.jl/dev/vault/) |
| **it describes** | the Julia packages: what they do, how to call them | the mathematics, the papers, the design decisions, and what does not work yet |
| **source** | `docs/` (Documenter) | this repository, which is an [Obsidian](https://obsidian.md) vault (Quartz) |

## The idea

**Implicit** i.e. replacing learning functions by learning relations. See [Implicit-Layer-Tutorial](https://implicit-layers-tutorial.org/)
| | Explicit Machine Learning | Implicit Learning |
|---|---|---|
| **Approximator** | functions: $f_\theta:X\rightarrow Y$ | relations: $R_\theta\subset X_1\times ...\times X_n$ |

How can we learn this?
- Introduce Error/Energy space $E$ (assume multivariate)
- Learn with the function:

$$
r_\theta: X_1\times ...\times X_n \rightarrow E
$$

where:

$$
(x_1,...,x_n)\in R_\theta : \Longleftrightarrow  r_\theta(x_1,...,x_n) \approx 0
$$

Which of the $x_i$ are inputs is not fixed when the relation is
written; it is chosen when the relation is *used*.

The table below is an **illustrating example**, not a definition. Polynomials are the cleanest
object in analysis with a direct implicit extension — the graph of a polynomial map is an
algebraic variety — so the pair (polynomials, varieties) is the one where every row can be
checked by a theorem rather than by analogy. Implicit learning is not *equal* to algebraic
varieties, any more than explicit learning is equal to polynomials; the general case is
functions versus relations, and the packages below wrap DEQs, neural ODEs, diffusion priors
and discriminators as relations that are not varieties at all. The table is quoted throughout
the vault, so it is kept as originally written; the vault's
[The Table Revisited](markdown/Bridges/The%20Table%20Revisited.md) reads it row by row.

| Aspect | Explicit | Implicit |
|---|---|---|
| **Approximator** | multivariate polynomials | algebraic varieties |
| **Inference** | Forward evaluation | Rootfinding |
| **Backpropagation** | Reverse mode automatic differentiation | Implicit function theorem |
| **Universal approximation theorem** | compact continuous functions via Weierstraß theorem | compact smooth manifolds via Nash–Tognoli theorem |
| **Well-posedness** | Always single-valued | May be multi-valued or have no solution/output only closest point to variety, instead of point on variety |
| **Loss formulation** | $\|f_\theta(x) - y\|^2$ | $\|r_\theta(x_1,..., x_n)\|^2$ |
| **Symmetry handling** | fixed unidirectional output direction | Symmetric: no distinguished input/output |
| **Computational cost of inference** | Cheap | Expensive (Newton's method, etc) |
| **Resulting Layer connections** | Directed Acyclic Graph | Arbitrary connected graph |

## What is here

One umbrella package over five smaller ones. Every package depends on **LuxCore** — not Lux —
so any Lux model wraps as a factor without pulling in Lux, Zygote or Optimisers.

| package | gives you |
|---|---|
| `LenticulumCore` | what a **factor** is: channels, polarities, beliefs, energies |
| `Mycelium` | how factors are **wired and scheduled**: graphs, messages, free energy |
| `Lenticulum` | the **linear-Gaussian** factors and Gaussian beliefs — everything has a closed form to check against |
| `VariationalDiffusion` | a **diffusion model** as a factor (VP-SDE, RED-Diff) |
| `ImplicitLayers` | **deep equilibrium networks and neural ODEs** as factors |
| `Adversarial` | **implicit generative models**: generators and density-ratio factors |

`LenticulumCore` and `Mycelium` define the framework; the other three are factor libraries
on top of it, and each can be ignored if you do not need that model family.

## How it differs from Lux.jl

Lenticulum is built *on* Lux (via LuxCore) and can use Lux models; Lux cannot use Lenticulum.
Three differences:

- A Lux layer is a [parametric lens](https://arxiv.org/html/2103.01931v2#S2): a forward `get`
  and a backward `put`, fixed at construction. A Lenticulum factor is a
  [parameterized statistical game](https://arxiv.org/html/2503.18608v2#S5) — it only *becomes*
  a lens once you choose which channels are inputs, and its backward pass is a posterior rather
  than a gradient.
- Lux wires layers into a DAG. Lenticulum wires factors into a factor graph — bipartite,
  undirected, and allowed to have cycles.
- In Lux, "run the network" is one forward pass. Here it is **message passing**: schedule
  messages between factors until they converge. On a tree that is exact in two sweeps; off a
  tree it is loopy belief propagation, with everything that implies.

## Factor graphs

A factor graph is a bipartite graph of **variables** (wires with no content of their own) and
**factors** (everything with content — including data, priors, losses and optimisers, which are
all nodes rather than special machinery). A factor has named **channels**; passing a message
through it means choosing a **polarity** — which channels are observed, which is being solved
for — and only then is a forward/backward pair assembled.

A complete worked example — a robot trajectory from odometry and one GPS reading, with the exact
posterior *and* the exact marginal likelihood falling out — is the
[getting-started page](https://MathStruct.github.io/Lenticulum.jl/dev/getting-started/).

## Using it

Not registered. From a clone:

```julia
using Pkg
Pkg.develop(path = "/path/to/Lenticulum.jl")
Pkg.develop(path = "/path/to/Lenticulum.jl/lib/Mycelium.jl")   # and the others under lib/
```

Tests, per package:

```sh
julia --project=.                           -e 'using Pkg; Pkg.test()'
julia --project=lib/Mycelium.jl             -e 'using Pkg; Pkg.test()'
# ... likewise for lib/LenticulumCore.jl, lib/VariationalDiffusion.jl,
#     lib/ImplicitLayers.jl, lib/Adversarial.jl
```

Documentation, both halves at once:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl           # API docs + the vault, into docs/build/
docs/vault/build.sh --serve                 # just the vault, live-previewed
```

The vault build needs Node ≥ 22; without it the API docs still build on their own. See
[`docs/vault/README.md`](docs/vault/README.md).

## The vault

This repository is an Obsidian vault. Open the root folder in Obsidian and start from
[`Start here.md`](Start%20here.md); the map of content is
[`markdown/Index.md`](markdown/Index.md). Implementation notes sit beside the source they
describe — `messages.md` next to `messages.jl` — and record what each file does not do as
carefully as what it does.

The vault is the larger half of the project. It is where the honest account lives: which claims
are verified against closed forms, which are approximations, and which are open problems.

## Status

A prototype, by one author, with ~740 tests. The exactness results live on the linear-Gaussian
fragment; everything else is approximate and says so. For what to use *instead* when you need
only one of the things this combines, see the vault's *Related Julia Projects*.
