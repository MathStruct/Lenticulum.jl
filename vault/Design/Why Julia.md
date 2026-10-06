#design #comparison

> Why this project is written in Julia, and not in C++, Rust, PyTorch, JAX or Mojo, and why it
> is not an extension of PyTorch or JAX. Short answer: in Julia, **ordinary code is the
> computation graph**. A solver loop, a custom factor, a new belief type or a closure written by
> a user is compiled to native code for its concrete types and can be differentiated, with no
> tracing restrictions, no graph breaks and no second language. Lenticulum's workload (many
> small heterogeneous factors, solvers inside differentiable programs, query directions chosen
> at run time) is exactly where that matters. The costs are real and listed at the end.

> Sources: Bezanson, Edelman, Karpinski & Shah, *Julia: A Fresh Approach to Numerical Computing*, SIAM Review 2017; Bezanson et al., *Julia: dynamism and performance reconciled by design*, OOPSLA 2018; Paszke et al., *PyTorch*, NeurIPS 2019; Ansel et al., *PyTorch 2*, ASPLOS 2024; Frostig, Johnson & Leary, *Compiling machine learning programs via high-level tracing*, SysML 2018; Bradbury et al., *JAX* (software); Moses & Churavy, *Enzyme*, NeurIPS 2020; Innes, *Don't Unroll Adjoint*, 2018 (Zygote); Revels, Lubin & Papamarkou, *Forward-Mode Automatic Differentiation in Julia*, 2016; full entries in [[Bibliography]]. Evidence from this repository: `lib/VariationalDiffusion.jl/ext/`, `energy_network.jl`, `implicit.jl`, `LenticulumCore/channels.jl`, `Mycelium/messages.jl`, the documentation's tutorials
>
> Bibliography: [[Bibliography#^bezanson2017julia|Bezanson et al. 2017]] · [[Bibliography#^bezanson2018dynamism|Bezanson et al. 2018]] · [[Bibliography#^paszke2019pytorch|Paszke et al. 2019]] · [[Bibliography#^ansel2024pytorch2|Ansel et al. 2024]] · [[Bibliography#^frostig2018jax|Frostig et al. 2018]] · [[Bibliography#^bradbury2018jax|Bradbury et al. 2018]] · [[Bibliography#^innes2019zygote|Innes 2018]] · [[Bibliography#^revels2016forwarddiff|Revels et al. 2016]]
>
> Theory (CT-ML wiki): [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens)

## 1. The central point: the program is the graph

PyTorch builds a graph of tensor operations by running Python, one operation at a time; each
operation crosses from the interpreter into C++ kernels. JAX traces a Python function into a
static graph and compiles it with XLA; the function must be traceable (pure, static shapes,
control flow written with `lax.cond` and `lax.while_loop`). PyTorch 2's `torch.compile`
recovers a graph from Python bytecode and falls back to Python at a *graph break* when it meets
something it cannot capture. In all three, user code lives in one of two worlds: the fast world
of the framework's operations, or the slow (or untraceable) world of Python around it.

Julia has one world. Every method is compiled, on first call, for the concrete types of its
arguments (Bezanson et al. 2017, 2018). A user's loop, closure or data structure is compiled
exactly like the library's own code, so adding it to a computation costs nothing beyond what
the same code would cost written by the library authors. Automatic differentiation works on
that same code: Zygote and Enzyme differentiate ordinary Julia (Enzyme at the level of the
compiled LLVM IR), ForwardDiff through dual numbers that flow through any generic code.

> [!important] What this means in practice
> A Newton solver with a data-dependent number of iterations, inside a model that is trained
> by gradient descent, is a few dozen lines of ordinary Julia here (`implicit_infer`), and its
> derivative is an implicit-function-theorem adjoint written as ordinary Julia
> (`implicit_pullback`). In JAX the loop would be a `lax.while_loop` with a `custom_vjp`; in
> eager PyTorch every iteration pays interpreter overhead; under `torch.compile` it is a graph
> break.

## 2. Evidence from this repository

**Multiple dispatch decides behaviour by type, across packages.**
- `Mycelium.combine` pools two beliefs; which algorithm runs is decided by *both* argument types
  (Gaussian with Gaussian adds natural parameters, Dirac absorbs, Trivial is the unit). A new
  belief type adds methods; the message-passing code does not change ([[Belief Algebra]]).
  Single dispatch (Python methods, C++ virtual functions) dispatches on one argument only;
  double dispatch has to be simulated.
- One `NoisePredictor` wrapper serves a Lux network, the closed-form `GaussianMixtureEps` and the
  energy-parametrised `EnergyNetwork`: the type parameter of the wrapped model selects the
  methods for `epsilon`, its Jacobian and its parameter derivative, and every function
  downstream (the implicit learner, the factor, RED-Diff) accepts all three unchanged
  ([[energy_network]] §2).

**Polarity lives in the type domain, so each query direction is specialised by the compiler.**
`Polarity{names,…}` carries the channel names as type parameters ([[Channels and Polarity]]).
Choosing which channels are inputs is a run-time decision, and the compiler still specialises
the code for it. In JAX, a new query direction means a new trace and a new compilation of the
whole function; in Julia only the methods that depend on it are compiled, once, and cached.

**Any AD backend, any device, through two small extensions.** The implicit learner needs
network derivatives (an input Jacobian, a parameter VJP, for energy networks also second
derivatives). One 124-line extension through DifferentiationInterface gives Zygote,
ForwardDiff, Enzyme and Mooncake; one 76-line extension gives Reactant, i.e. XLA compilation
with Enzyme inside, the JAX-style path *when* it fits ([[backends]]). Nested derivatives for
the energy networks (forward-mode over Zygote, through Lux layers) worked without special
support ([[energy_network]] §4).

**The scientific ecosystem is native.** The tutorials use, side by side and without glue code:
Lux for networks, Optimisers, three AD systems, an ODE integrator written in a few lines,
`ImplicitLayers`' root solvers, CairoMakie, SymbolicRegression.jl (tutorial 7), and Literate +
Documenter for the documentation itself. The related EIT work uses Ferrite.jl for finite
elements and Krylov.jl for linear solvers in the same way. In Python several of these would be
separate C++ or Fortran libraries with their own array types and no shared AD.

## 3. Why not just extend PyTorch or JAX?

The honest version of the question: both have far larger communities, pretrained models,
mature GPU support and better tooling for large static models. For a big image or language
model, they are the better choice. Lenticulum's workload is different:

| Lenticulum needs | PyTorch / JAX friction |
|---|---|
| many **small**, heterogeneous factors, each called many times per query | per-operation Python overhead dominates small tensors (PyTorch); a heterogeneous graph of different factor types is awkward to trace as one static program (JAX) |
| solvers with data-dependent iteration counts **inside** differentiable models | `lax.while_loop` + `custom_vjp` (JAX); graph breaks or eager overhead (PyTorch) |
| the direction of a query chosen **at run time** | a retrace and recompile per direction (JAX) |
| new belief and factor types without editing the core | class hierarchies with single dispatch; double dispatch simulated |
| scientific building blocks (ODE, FEM, symbolic regression, Krylov) **in the same AD world** | separate native libraries, usually not differentiable end to end |

Extending them would mean writing the core in Python around their tensors and fighting their
execution models exactly where this project is unusual. Reactant gives the JAX execution model
inside Julia for the parts where a static compiled graph is the right tool, which is the best of
both rather than a choice between them.

## 4. Why not C++ or Rust?

Both give native speed and Rust gives memory safety. But research code in this project is
written and rewritten interactively, at a REPL, with plots, every day; templates and traits do
not give open multiple dispatch across independently written packages; and the
automatic-differentiation and scientific-computing stacks are thinner (Enzyme itself works at
the LLVM level and can differentiate C++ and Rust, but the surrounding ecosystem of solvers and
models is not there). Julia gets the native speed through its compiler without leaving the
high-level language: the "two-language problem" is the problem it was designed to remove
(Bezanson et al. 2017).

## 5. Why not wait for Mojo?

Mojo aims at the same two-language problem, with a Python-like surface and a compiler for
performance. As of writing it is young: the scientific ecosystem this project uses (AD over
arbitrary code, ODE and FEM solvers, symbolic regression, probabilistic programming,
documentation tooling) does not exist there yet, and waiting for it would mean not doing the
research. If it matures, the ideas in this vault carry over; the code would not.

## 6. The costs, honestly

- **Compile latency.** The first call of anything compiles it. A fresh documentation build takes
  minutes, most of it compilation; the project caches rendered tutorials for that reason.
- **AD is powerful but uneven.** Measured here: Zygote rejects array mutation (the time embedding
  and parameter rebuild had to be rewritten), Enzyme needs runtime activity on Lux layers on
  CPU, and Enzyme's forward-over-reverse fails on Lux layers, so energy networks use
  ForwardDiff-over-Zygote instead ([[backends]] §7, [[energy_network]] §4).
- **Type stability is the programmer's job.** Code that hides types from the compiler is slow;
  the fast path needs some discipline ([[Parallelism and Compilation]]).
- **A smaller community.** Fewer pretrained models, fewer people to hire, fewer answers online.
  For this project's small models and novel algorithms that matters little; for an LLM-scale
  experiment it would matter a lot.

## 7. The verdict

The deciding property is not speed alone, nor multiple dispatch alone, nor the ecosystem
alone, but their combination: **a user's code becomes part of the compiled, differentiable
program at no extra cost, and dispatches into everyone else's code by type.** A framework whose
users define the factors, the beliefs and the solvers needs exactly that.

Related: [[Related Julia Projects]], [[Parallelism and Compilation]], [[The Type Discipline of a Factor Graph]],
[[Channels and Polarity]], [[Belief Algebra]], [[backends]], [[energy_network]], [[Bibliography]]
