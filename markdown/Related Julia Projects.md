# Related Julia Projects

> Where this project sits in the Julia ecosystem, and — more usefully — **when you should use
> something else**.
>
> Almost everything below is mature, well engineered and widely used. This project is a
> prototype. The honest summary is at the bottom in §10; read that first if you are deciding
> what to build on.

## 1. The one-line map

| if you want | use |
|---|---|
| neural networks with explicit parameters | **Lux.jl** |
| a probabilistic model and posterior samples | **Turing.jl** |
| a factor graph with message passing and a free energy | **RxInfer.jl** |
| non-Gaussian, multimodal factor graphs (SLAM) | **IncrementalInference.jl / Caesar.jl** |
| acausal physical equations, compiled and solved | **ModelingToolkit.jl** |
| implicit layers (DEQ, neural ODE) | **DeepEquilibriumNetworks.jl**, **DiffEqFlux.jl** |
| compositional systems with categorical structure | **Catlab.jl / AlgebraicDynamics.jl** |
| **all of the above at once**, on relations that may be learned | nothing — which is why this exists |

## 2. Four axes, and nobody covers all four

| | relations (no fixed direction) | beliefs + free energy | learned components | categorical structure |
|---|---|---|---|---|
| Lux.jl | — | — | ✓ | implicit (it *is* a parametric lens) |
| Turing.jl / Gen.jl | — | ✓ | ✓ | — |
| **RxInfer.jl** | partly | **✓** | — | Forney-style |
| IncrementalInference.jl | ✓ | ✓ (non-parametric) | — | — |
| ModelingToolkit.jl | **✓** | — | — | — |
| Catlab / AlgebraicDynamics | **✓** | — | — | **✓** |
| *this project* | ✓ | ✓ | ✓ | ✓ |

The last row is a claim about *scope*, not about quality. Every other row does its columns far
better.

## 3. Lux.jl — and it is a dependency

Covered in depth by [[Lux as a Parametric Lens]]. The short version: a Lux layer is a
**function** whose direction is fixed at construction; a factor here is a **relation** whose
direction is chosen per call. Everything else — `initialparameters`, `initialstates`, `setup`,
the `ps`/`st` separation, the container-layer trick for parameter trees — is mirrored
deliberately, name for name.

Every package here depends on **LuxCore**, not Lux, so any Lux model wraps as a factor without
dragging in Lux, Zygote or Optimisers. `Flux.jl` differs mainly in implicit parameter handling
and is not otherwise a different comparison.

## 4. Turing.jl, Gen.jl — probabilistic programming

A PPL specifies a **directed generative program** (`x ~ Normal(...)`) and infers by sampling —
HMC/NUTS in Turing, programmable inference and traces in Gen. That is a different object from
a graph of relations, and [[Probabilistic Types]] §2 says why in type terms: a PPL types
programs in the Kleisli category of the probability monad, and relations do not form one —
`combine` is a merge, which no monad supplies.

Practically: **if your model is a generative story, write it in Turing.** It will be faster,
better tested and better documented. This project is for when the model is a set of
*constraints* with no natural generative order.

`Gen.jl` is worth a separate look for one reason: its **programmable inference** — custom
proposals, involutive MCMC — is the closest existing answer to "the inference procedure should
itself be a first-class, composable object", which is what
[[The Inferencer and the Optimizer]] proposes to take further.

## 5. RxInfer.jl and ForneyLab.jl — the nearest neighbour

This is the closest existing project by a considerable distance, and it should be the first
thing anyone compares against.

RxInfer does **reactive message passing on Forney-style factor graphs**: sum-product and
variational message passing, a constrained Bethe free energy, model specification via
`GraphPPL.jl`, the `ReactiveMP.jl` engine, and explicit support for building **active
inference** agents that minimise free energy. `ForneyLab.jl` is its predecessor and generated
inference algorithms by message passing on FFGs.

Almost every structural idea in `Mycelium` has a counterpart there: bipartite graphs, local
message rules, a free energy that scores the model, schedules, and the tree/loopy distinction.

> [!important] The difference is what a factor is allowed to be
> RxInfer's factors are **probability distributions from known families**, and its speed comes
> from exploiting local conjugacy. It does inference on a *specified probabilistic model*,
> extremely well.
>
> A factor here may be an arbitrary residual, a root-finder, an ODE solve, a diffusion prior or
> a neural network — objects with no density, no conjugate structure, and often no direction.
> That is a more general target and it is much less finished.

Two consequences worth being blunt about:

- **For anything RxInfer covers, use RxInfer.** State-space models, conjugate structure,
  real-time filtering — it is built for that and this is not.
- **Its active-inference work is the demonstration** that the cooperative two-part
  architectures of [[The Two-Part Diagram]] §5 really do work as factor graphs. That is a
  result this project benefits from and did not produce.

## 6. IncrementalInference.jl / Caesar.jl — they solved the `combine` gap

`messages.md` §1 has recorded since the beginning that pooling two sample-based beliefs is the
package's main blocker. **This stack solved it**, and by a different route than either of the
ones the vault has proposed.

`IncrementalInference.jl` implements **Multi-Modal iSAM**: non-Gaussian, multimodal factor
graph inference over the Bayes (junction) tree, with **nonparametric belief propagation** —
beliefs as kernel density estimates (`KernelDensityEstimate.jl`), and the *product* of belief
functions estimated by a multiscale Gibbs sampling strategy. `Caesar.jl` and `RoME.jl` are the
SLAM-facing layers on top.

> [!important] Read this before implementing `combine` for `SampleBelief`
> [[Implicit Generative Models]] §5 proposes density-ratio estimation and
> [[Training Energy-Based Models]] §2.2 proposes noise-contrastive estimation. Both are
> plausible. Neither is tested at scale, and **there is a Julia package that has been doing the
> KDE-and-Gibbs version in production robotics for years.**
>
> It is also the closest thing in the ecosystem to this project's ambitions on the *belief*
> axis, and it gets there without needing learned factors or a categorical story.

It is worth noting what it does *not* do: factors are measurement models, not learned
relations, and there is no notion of running a factor in an arbitrary direction chosen at call
time.

## 7. ModelingToolkit.jl and Symbolics.jl — acausal, and symbolic

Two notes already: [[ModelingToolkit as an Acausal Relation]] and
[[The Structural Gap to ModelingToolkit]]. The conclusions, compressed:

- MTK's **incidence graph is a factor graph** — equations as factors, variables as variables —
  so its structural analysis and Mycelium's scheduling operate on the same object.
- MTK has structure this project cannot have: **models are symbolic terms**, so they can be
  inspected and rewritten (`mtkcompile`: alias elimination, tearing, Pantelides index
  reduction). Factors here are opaque objects, and making them terms would cost the ability to
  put a neural network inside one.
- The realistic relationship is a **division of labour**, not a merge:
  [[Time as a Base]] §8 — MTK owns compile time (symbolic, structural, index reduction);
  this owns run time (beliefs, messages, free energy); the interface is the compiled, index-1
  system.

## 8. DeepEquilibriumNetworks.jl, DiffEqFlux.jl, NonlinearSolve.jl

`ImplicitLayers` covers the same models and **depends on none of them** — it needs the inner
Lux network, not the assembled layer, and its solvers are derivative-free (Broyden, as
`DeepEquilibriumNetworks.jl` uses) rather than borrowed.

The distinction is in [[DEQ as a Relation]]: a `DeepEquilibriumNetwork` gives you a Lux layer,
i.e. a function with the solve sealed inside and one direction. Keeping the *residual* instead
gives two. Wrapping their assembled layer is supported (`LuxFactor`) and yields exactly one
polarity — which is the point of the comparison.

For anything requiring stiff solvers, adaptivity, GPU support or real sensitivity analysis,
these packages are the answer and this one is not.

## 9. Catlab.jl and AlgebraicDynamics.jl — the structure, already implemented

This is the comparison the vault had been missing, and it is uncomfortable in a useful way.

`Acausal Composition is a Hypergraph Category` derives, from first principles, that composing
acausal systems needs a **hypergraph category** — variables as Frobenius junctions, arbitrary
degree, no direction. AlgebraicJulia *implements that*, with the same vocabulary:

| the vault's derivation | AlgebraicDynamics / Catlab |
|---|---|
| hypergraph category | algebras of the operad of **undirected wiring diagrams** |
| a factor's channels | a resource sharer's **ports** |
| a shared variable (Frobenius spider) | a **junction**; ports on one junction are identified |
| collapsing a subgraph into a factor | **`oapply`** — apply a composition pattern to primitives |
| specifying the wiring | the **`@relation`** macro |

> [!important] The hierarchy gap has a mature implementation
> [[The Structural Gap to ModelingToolkit]] §4 records that subsystems are not first-class
> here — there is no "collapse this subgraph into a factor" operation, only a formalisation
> (decorated cospans) in [[Acausal Composition is a Hypergraph Category]] §6.
>
> **`oapply` is that operation**, and it works. If the hierarchy story is ever built, it should
> be built by reading Catlab rather than from the cospan literature directly.

What AlgebraicJulia does not have is the probabilistic layer: no beliefs, no inversions, no
free energy, no learned components. Its systems are specified, not fitted. So the relationship
is complementary rather than competitive — they have the structure, this has the statistics,
and neither has both.

## 10. Where this project is worse

Stated plainly, because §2's table could be read as a boast.

- **Maturity.** Six packages, ~740 tests, one author, no users. Everything above is production
  software.
- **Scale.** Dense linear algebra throughout; finite-difference Jacobians; no GPU support; no
  sparse solvers. `ImplicitLayers`'s `fd_jacobian` is honestly labelled a test-scale tool.
- **No training.** Every factor library here *consumes* a trained network. There is no
  automatic-differentiation dependency anywhere and therefore no way to fit anything.
- **Loopy graphs are wrong in a specific way** — exact means, incorrect variances — which
  RxInfer and IncrementalInference both handle better.
- **The objective is not well defined on mixed graphs**, per [[Energy-Based Factor Graphs]]
  §3.2. This is the deepest of the gaps and it is unresolved.

## 11. So what is the niche

The intersection of four things, none of which is individually unserved:

> **relations** with no fixed direction (MTK, Catlab) — carrying **beliefs and a free energy**
> (RxInfer, IncrementalInference) — where a factor may be a **learned** implicit model (Lux,
> DiffEqFlux) — with the **categorical structure** made explicit (Catlab).

The concrete target is *probabilistic acausal modelling*: a physical network written as
equations, some of whose components are fitted, inferred jointly with calibrated uncertainty
and a model-comparison score. MTK gives the equations and no posterior; RxInfer gives the
posterior and needs a generative model; neither expresses a physical law that holds only
approximately with a learned component beside it.

If you need only one of the four axes, use the package that owns it.

## Sources

- [Lux.jl](https://lux.csail.mit.edu/) · [Turing.jl](https://turinglang.org/) ·
  [Gen.jl](https://www.gen.dev/)
- [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl) ·
  [ForneyLab.jl](https://github.com/biaslab/ForneyLab.jl)
- [IncrementalInference.jl](https://github.com/JuliaRobotics/IncrementalInference.jl) ·
  [Caesar.jl](https://juliarobotics.org/Caesar.jl/latest/)
- [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) ·
  [DeepEquilibriumNetworks.jl](https://docs.sciml.ai/DeepEquilibriumNetworks/stable/) ·
  [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/)
- [Catlab.jl](https://algebraicjulia.github.io/Catlab.jl/dev/) ·
  [AlgebraicDynamics.jl](https://algebraicjulia.github.io/AlgebraicDynamics.jl/dev/) ·
  [the resource-sharing post](https://blog.algebraicjulia.org/post/2021/01/resource_sharers/)

Related: [[Lux as a Parametric Lens]], [[ModelingToolkit as an Acausal Relation]],
[[The Structural Gap to ModelingToolkit]], [[Acausal Composition is a Hypergraph Category]],
[[The Two-Part Diagram]], [[The Inferencer and the Optimizer]], [[Probabilistic Types]],
[[DEQ as a Relation]], [[Implicit Generative Models]], [[Training Energy-Based Models]],
[[Energy-Based Factor Graphs]], [[Time as a Base]]
