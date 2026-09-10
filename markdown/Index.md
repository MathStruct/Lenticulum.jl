# Index — Lenticulum concept vault

This is the map of content for the theory behind Lenticulum.jl. **Three** papers are the
backbone, and one bridge layer connects them to Julia:

- **[Categorical Foundations of Gradient-Based Learning](https://arxiv.org/html/2103.01931v2)**
  (Cruttwell, Gavranović, Ghani, Wilson, Zanasi) — how *ordinary* deep learning is a
  **parametric lens**. This is the theory Lux.jl already implements without saying so.
- **[AutoBayes](https://arxiv.org/html/2503.18608v2)** (St Clere Smithe & Perin) — how
  *Bayesian* inference is a **statistical game**, which is a parametric lens whose
  backward pass is a posterior rather than a gradient. This is what Lenticulum.jl
  implements.
- **[A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)**
  (LeCun, Chopra, Hadsell, Ranzato, Huang) — how learning works with **no normalisation at
  all**: an energy, an argmin, and a factor graph. Read after the other two, but it is the
  layer [[README]]'s own table was written from, and the one most of the implementation
  actually runs in.

Read in this order.

## 1. The classical story (what Lux.jl is)

1. [[Para]] — parameters as a categorical construction
2. [[Lens]] — forward/backward pairs, `get` and `put`
3. [[Parametric Lens]] — Definition 2.5, the object Lux layers really are
4. [[Cartesian Reverse Differential Category]] — where the gradient comes from
5. [[Learning Components as Parametric Lenses]] — model, loss, learning rate, optimiser
6. [[Lux as a Parametric Lens]] — **the concrete Lux.jl ↔ lens dictionary**

## 2. The Bayesian story (what Lenticulum.jl is)

7. [[Open Model]] — Definition 1: a kernel with an explicit latent space
8. [[Composition of Open Models]] — Definitions 4–7
9. [[Copiers Cups and Caps]] — how the DAG restriction is lifted
10. [[Bayesian Inversion]] — Bayes' law as a chain rule
11. [[Bayesian Lens]] — Definitions 9–11
12. [[Composition of Bayesian Lenses]] — Definition 12, Theorem 13
13. [[Variational Free Energy]] — Definition 17, Proposition 18: the energy/entropy split
14. [[Statistical Game]] — Definition 20, **the definition Lenticulum factors satisfy**
15. [[Composition of Statistical Games]] — Definition 22, Theorem 23 (the chain rule)
16. [[Parameterized Statistical Game]] — Definition 27, what actually gets optimised
17. [[Composition of Gradients]] — Definitions 28–29, Remark 30: laxness
18. [[Examples from the Paper]] — Appendix A, worked through

## 2b. The energy story (what is left when you stop normalising)

LeCun's tutorial and the modern EBM literature. Structurally *below* AutoBayes: drop the
partition function and a statistical game becomes an energy-based factor graph.

19. [[Energy-Based Learning]] — the framework; why [[README]]'s table is this paper's table;
    **loss functionals**, the concept Lenticulum has no slot for; and the collapse problem,
    which `gaussian.jl` already guards against under the name `complexity`
20. [[Energy-Based Factor Graphs]] — LeCun §6 is `Mycelium` without the beliefs. **min-sum is
    the $T\to0$ limit of sum-product**, so a `DiracBelief` message is a min-sum message and
    three packages' "missing entropy" is the Bethe form degenerating correctly
21. [[Training Energy-Based Models]] — Song & Kingma, Du & Mordatch. Two of the three standard
    EBM training methods are already implemented here under other names: **score matching is
    `VariationalDiffusion`**, **NCE is `Adversarial.RatioFactor`** — and NCE with a known noise
    distribution is the `belief_logdensity` [[messages]] §1 has always wanted

## 2c. Types

What kind of type theory this is, and what the code's own types are doing.

22. [[Probabilistic Types]] — "probabilistic type" names three different things. Beliefs are
    **not** the probability monad (the merge operation is not monadic), but
    ``r_\theta \approx 0`` **is** a graded type judgment with the energy as its grade — and the
    grade lives in a semiring, which is the temperature
23. [[The Type Discipline of a Factor Graph]] — a factor graph is neither linear nor cartesian
    but **Frobenius**; putting polarity in the type domain already caught a real bug; and five
    separately-recorded gaps turn out to be **one typing problem**

## 3. The bridge to Lenticulum.jl

24. [[Channels and Polarity]] — reconciling AutoBayes' X/⟦c⟧/Y with the README's
    $P_{in} + P_{out} + P_{latent} = \mathrm{Id}$
25. [[Scalar and Multivariate Energy]] — **the design decision that is ours, not the
    paper's**: two energies, and the chain rule adapted to the multivariate one
26. [[Implicit Learners]] — the three model families, and why the multivariate energy
    is what makes them work
27. [[AutoBayes to Lenticulum]] — the full naming dictionary paper → Julia
28. [[ModelingToolkit as an Acausal Relation]] — **the other bridge**: MTK is the implicit
    column of [[README]]'s table minus the probability; Willems' behaviors; why MTK's
    incidence graph *is* a factor graph, and why MTK solves where Mycelium propagates
29. [[Acausal Composition is a Hypergraph Category]] — how much structure it takes to wire an
    arbitrary graph rather than a DAG; a variable node is a **Frobenius spider**, and improper
    Gaussian beliefs are what make that work
30. [[The Structural Gap to ModelingToolkit]] — the question in the other direction: **what
    does MTK contain that Lenticulum does not?** Five ranked items; two are not addable
31. [[Time as a Base]] — **the design note**: what an MTK × Lenticulum extension with a time
    dimension would be. A base change, not a redesign — variables carry trajectories, and a
    belief over a trajectory *is* a chain factor graph
32. [[Three Senses of Implicit]] — the word means **three independent things**: implicit
    likelihood, implicit computation, implicit relation. Only the third buys you a polarity
33. [[Depth in Implicit Learning]] — is there a "no deep learning theorem"? **Yes on the
    linear-Gaussian fragment and no elsewhere** — it is the $d=1$ case of a degree bound the
    vault already derived; and depth-as-computation and depth-as-expressivity come apart
34. [[Prolog and Logic Programming]] — "the Prolog of machine learning", scored honestly.
    **Modes are polarities exactly**; Prolog is the *boolean* semiring of the same framework;
    and the shortfall is one thing — **schemas versus ground instances**
## 3b. The algebraic family, worked out

The first of the three [[Implicit Learners]] families in full detail — the case where every
abstract slot of the framework becomes computable, so the framework can be checked.
Entry point: [[Algebraic Implicit Learners]].

- *model class*: [[Varieties Ideals and Real Nullstellensatz]], [[The Veronese Parametrisation]],
  [[Universal Approximation by Nash-Tognoli]]
- *learning*: [[Fitting is a Nullspace Problem]], [[The Parameter is a Grassmannian]],
  [[Algebraic versus Geometric Distance]]
- *inference*: [[Inference as Root Finding]], [[Branches and the Discriminant]],
  [[Backpropagation by the Implicit Function Theorem]]
- *structure*: [[Composition is Elimination]], [[Differential Algebra and DAE Factors]],
  [[Algebraic Statistics Bridge]]
- *verdict*: [[The Algebraic Factor as a Statistical Game]],
  [[Open Problems in Algebraic Implicit Learning]]

## 3c. Factor graphs and message passing

`Mycelium.jl` — how factors are wired and in what order they talk. The layer with no
counterpart in Lux.

- [[Factor Graphs]] — bipartite structure; the **two** acyclicity notions (`istree` vs `isdag`)
- [[Everything is a Factor]] — data, priors, losses and optimisers as graph nodes, forced by Remark 24
- [[Messages are Inversions]] — a factor → variable message **is** ``c'_\pi``; both exclusion principles
- [[Polarity Resolution]] — target + available messages → a `Polarity`; the two legality checks
- [[Schedules]] — the schedule zoo; tree exactness; pruning by edge direction
- [[Bethe Free Energy]] — Theorem 23 generalised to graphs; energies add, entropies get a counting correction
- [[Loopy Message Passing]] — what breaks off a tree, and the monodromy problem
- [[The Linear Gaussian Chain]] — **the worked example**: GTSAM's `OdometryExample`, exact
  marginals against the joint information matrix, and what this library is *not*

## 3d. The equilibrium family, worked out

The second of the three [[Implicit Learners]] families — SciML's deep equilibrium networks
and neural ODEs, wrapped as factors. Entry point: [[The Equilibrium Family]].

- *the fixed point*: [[DEQ as a Relation]] — a DEQ is defined by a relation and shipped as a
  function; the reverse solve; the contraction caveat made testable; why the IFT brings
  automatic differentiation back
- *the flow*: [[NeuralODE as an Invertible Factor]] — bidirectional for free, because a flow
  is a diffeomorphism; the density correction, and why it is currently dead code

## 3e. The diffusion family, worked out

The third of the three [[Implicit Learners]] families — the case where the inversion is
neither exact nor a root-find but a **proximal solve**. Entry point:
[[The Diffusion Family]].

- *the forward process*: [[The VP-SDE]] — Song et al. 2021; the perturbation kernel, the score
  identity, Tweedie's denoiser, and the `tmin` floor nobody documents
- *the inversion*: [[RED-Diff as a Statistical Game]] — variational inference with a point-mass
  posterior; the stop-gradient; and **λ is derivable, not merely tunable**
- *the factor*: [[The Diffusion Factor]] — $P_{in}+P_{out}+P_{latent}=\mathrm{Id}$ becomes a
  `Polarity`; what a Dirac-valued message does to a factor graph
- *alternatives*: [[ProxDM and Proximal Alternatives]] — DPS, ΠGDM, ProxDM, plug-and-play, and
  why RED-Diff was implemented first

## 3f. The adversarial family, worked out

Implicit **generative** models — the case where what is missing is the *density* rather than
the direction, so every factor is unidirectional. Entry point:
[[Implicit Generative Models]].

- *the paper*: [[Implicit Generative Models]] — Mohamed & Lakshminarayanan; learning by
  comparison; the four estimators; and why a density ratio is the answer to [[messages]] §1
- *the diagram*: [[GANs as Two Factors]] — two parameter sets, three nodes, and **one sign the
  Bethe free energy cannot hold**; the missing structure is an open game
- *the vocabulary*: [[Three Senses of Implicit]] — filed under §3 above, and the reason this
  family has no polarities

## 3g. Two-part architectures, and the meta-graph

Where GANs, reinforcement learning and control theory draw the same diagram — and what the
project would look like with factor graphs attached *to* the factor graph.

- [[The Two-Part Diagram]] — the shared shape is the **wiring**, which is a trace and already
  solved. What differs is the **objective**: one function (EM, VAE, active inference, LQG) fits
  the Bethe free energy; minimax (GANs) and general bilevel (actor–critic) do not
- [[The Inferencer and the Optimizer]] — **the design note**: meta-graphs over the base graph's
  messages and parameters. Half of it already exists as `OptimiserFactor` and
  `AmortisedInversion`; it is loop-free because it is coordinate descent on one objective; and
  it is the **separation principle**, exact for linear-Gaussian and failing into *exploration*

## 4. Implementation

Implementation notes live next to the code, per [[Start here]]:

**LenticulumCore.jl** — [[LenticulumCore]], [[abstract_types]], [[channels]], [[energy]],
[[open_model]], [[lens]], [[statistical_game]]

**Mycelium.jl** — [[Mycelium]], [[graph]], [[polarity_resolution]], [[messages]],
[[schedules]], [[passing]], [[free_energy]], [[factors]]

**Lenticulum.jl** — [[constraint]] (`beliefs.md` and `gaussian.md` are not yet written)

**VariationalDiffusion.jl** — [[VariationalDiffusion]], [[schedule]], [[predictor]],
[[reddiff]], [[factor]]

**ImplicitLayers.jl** — [[ImplicitLayers]], [[solve]], [[deq]], [[flow]], [[neuralode]],
[[luxfactor]]

**Adversarial.jl** — [[Adversarial]], [[generator]], [[ratio]]

## Where this would be useful

Six domains with the same shape — a network of relations, sparse asynchronous observations,
some parts known and some fitted, and a residual that means something in the domain itself.

- [[Motivating Examples]] — the overview, and the six properties they share
- [[SLAM and Sensor Fusion]] · [[Trading and Financial Markets]] ·
  [[Energy Markets and Power Grids]]
- [[Metabolomics and Proteomics]] · [[Molecular Dynamics]] ·
  [[Climate and Dynamical Systems]]
- [[Language Models]] — a **product** of diffusion experts, not a mixture; why the framing
  clarifies and the machinery does not

## Orientation

- [[Related Julia Projects]] — where this sits in the Julia ecosystem, and **when to use
  something else**. The nearest neighbour is `RxInfer.jl`; `IncrementalInference.jl`
  has already solved the `combine` gap by kernel-density BP; and Catlab's `oapply` is the
  subgraph-as-factor operation the vault records as missing
- [[Parallelism and Compilation]] — parallelism, GPUs and XLA/MLIR in a *dynamic* SLAM
  setting. There is no junction tree and no parallelism today; a sweep is **quadratic in the
  number of factors** (measured); and the compile-vs-dynamic tension dissolves if you
  **compile the factor types, not the graph**
- [[PhD Proposal]] and [[PhD Proposal v2]] — the same programme argued two ways: problem-first
  and capability-first

## Existing notes

- [[README]] — the pitch
- [[ImplicitREDDiff]] — the diffusion-based implicit learner
