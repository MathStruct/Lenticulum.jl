#overview

> Every note of the Lenticulum vault, in reading order. How the vault is organised and what
> its conventions are: [[Start Here]]. The general category theory — Para, lenses, Markov
> categories, Bayesian lenses, statistical games — is in the
> [CT-ML wiki](https://mathstruct.org/CategoryTheory-ML-Wiki/); links to it below are ordinary
> web links, links within this vault are wikilinks.

**Three** papers are the backbone, and one bridge layer connects them to Julia:

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


**References:** [[Bibliography]], every work cited in the vault, the implementation notes and the
tutorials, generated from one `.bib` file that the documentation also uses.

## 1. The classical story (what Lux.jl is)

1. [Para](https://mathstruct.org/CategoryTheory-ML-Wiki/Para-Construction) — parameters as a categorical construction
2. [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) — forward/backward pairs, `get` and `put`
3. [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) — Definition 2.5, the object Lux layers really are
4. [Cartesian Reverse Differential Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category) — where the gradient comes from
5. [Learning Components as Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses) — model, loss, learning rate, optimiser
6. [[Lux as a Parametric Lens]] — **the concrete Lux.jl ↔ lens dictionary**

## 2. The Bayesian story (what Lenticulum.jl is)

7. [[Open Models and Latent Channels]] — Definitions 1–8: a kernel with an explicit latent
   space, and how the DAG restriction is lifted
8. [Composition of Open Models](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model#composition-without-integration-definition-4) — Definitions 4–7
9. [[Inversions and Bayesian Lenses]] — Definitions 9–11: Bayes' law as a chain rule, and the
   inversion as a free choice
10. [Composition of Bayesian Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens#the-chain-rule-bayesian-inversion-is-functorial) — Definition 12, Theorem 13
11. [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) — Definition 17, Proposition 18: the energy/entropy split
12. [[Factors are Parameterized Statistical Games]] — Definitions 20 and 27: **the definition
    Lenticulum factors satisfy**, and what actually gets optimised
13. [Composition of Statistical Games](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#composition-energies-add-entropies-chain-definition-22) — Definition 22, Theorem 23 (the chain rule)
14. [Composition of Gradients](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients) — Definitions 28–29, Remark 30: laxness
15. [[AutoBayes Examples as Factor Graphs]] — Appendix A, worked through

## 2b. The energy story (what is left when you stop normalising)

LeCun's tutorial and the modern EBM literature. Structurally *below* AutoBayes: drop the
partition function and a statistical game becomes an energy-based factor graph.

16. [[Energy-Based Learning]] — the framework; why [[README]]'s table is this paper's table;
    **loss functionals**, the concept Lenticulum has no slot for; and the collapse problem,
    which `gaussian.jl` already guards against under the name `complexity`
17. [[Energy-Based Factor Graphs]] — LeCun §6 is `Mycelium` without the beliefs. **min-sum is
    the $T\to0$ limit of sum-product**, so a `DiracBelief` message is a min-sum message and
    three packages' "missing entropy" is the Bethe form degenerating correctly
18. [[Training Energy-Based Models]] — Song & Kingma, Du & Mordatch. Two of the three standard
    EBM training methods are already implemented here under other names: **score matching is
    `VariationalDiffusion`**, **NCE is `Adversarial.RatioFactor`** — and NCE with a known noise
    distribution is the `belief_logdensity` [[messages]] §1 has always wanted

## 2c. Types

What kind of type theory this is, and what the code's own types are doing.

19. [[Probabilistic Types]] — "probabilistic type" names three different things. Beliefs are
    **not** the probability monad (the merge operation is not monadic), but
    $r_\theta \approx 0$ **is** a graded type judgment with the energy as its grade — and the
    grade lives in a semiring, which is the temperature
20. [[The Type Discipline of a Factor Graph]] — a factor graph is neither linear nor cartesian
    but **Frobenius**; putting polarity in the type domain already caught a real bug; and five
    separately-recorded gaps turn out to be **one typing problem**

## 3. The bridge to Lenticulum.jl

21. [[Channels and Polarity]] — reconciling AutoBayes' X/⟦c⟧/Y with the README's
    $P_{in} + P_{out} + P_{latent} = \mathrm{Id}$
22. [[Scalar and Multivariate Energy]] — **the design decision that is ours, not the
    paper's**: two energies, and the chain rule adapted to the multivariate one
23. [[Implicit Learners]] — the three model families, and why the multivariate energy
    is what makes them work
24. [[AutoBayes to Lenticulum]] — the full naming dictionary paper → Julia
25. [[ModelingToolkit as an Acausal Relation]] — **the other bridge**: MTK is the implicit
    column of [[README]]'s table minus the probability; Willems' behaviors; why MTK's
    incidence graph *is* a factor graph, and why MTK solves where Mycelium propagates
26. [[Acausal Composition is a Hypergraph Category]] — how much structure it takes to wire an
    arbitrary graph rather than a DAG; a variable node is a **Frobenius spider**, and improper
    Gaussian beliefs are what make that work
27. [[The Structural Gap to ModelingToolkit]] — the question in the other direction: **what
    does MTK contain that Lenticulum does not?** Five ranked items; two are not addable
28. [[Time as a Base]] — **the design note**: what an MTK × Lenticulum extension with a time
    dimension would be. A base change, not a redesign — variables carry trajectories, and a
    belief over a trajectory *is* a chain factor graph
29. [[Three Senses of Implicit]] — the word means **three independent things**: implicit
    likelihood, implicit computation, implicit relation. Only the third buys you a polarity
30. [[Depth in Implicit Learning]] — is there a "no deep learning theorem"? **Yes on the
    linear-Gaussian fragment and no elsewhere** — it is the $d=1$ case of a degree bound the
    vault already derived; and depth-as-computation and depth-as-expressivity come apart
31. [[Prolog and Logic Programming]] — "the Prolog of machine learning", scored honestly.
    **Modes are polarities exactly**; Prolog is the *boolean* semiring of the same framework;
    and the shortfall is one thing — **schemas versus ground instances**
32. [[Geometric Deep Learning and Physical Laws]] — equivariance constrains the **map**, a
    law constrains the **configuration**; $\dot x=-x$ is equivariant and conserves nothing.
    The measured case: **direct-force equivariant potentials are unstable in MD**
    (Bigi–Langer–Ceriotti, ICML 2025). And the converse — this project has no equivariance
33. [[The Table Revisited]] — the README's table is an **illustrating example** (polynomials → varieties,
    the one pair where every slot becomes a theorem), read row by row. Holds up; the one
    real caveat is the **Loss row collapses without a contrastive term**; "symmetric" means
    *acausal*; the whole table is the $T\to0$ limit of the AutoBayes layer

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
- *kernels*: [[Kernel Methods for Implicit Learning]] — RBF and GP implicit surfaces, kernel PCA, density
  ridges (the first tutorial's model is a kernel density estimate), kernel exponential families
- *later*: [[Symbolic Implicit Learning]] — relations as formulas found by search (noted, not built)

## 3c. Factor graphs and message passing

`Mycelium.jl` — how factors are wired and in what order they talk. The layer with no
counterpart in Lux.

- [[Beliefs]] — what travels on an edge: [[Trivial Belief]], [[Dirac Belief]], [[Gaussian Belief]],
  [[Sample Belief]], [[Categorical Belief]], [[Mixture Belief]], and how they pool (`combine`)
- [[Belief Algebra]] — every other operation (addition, mixture, logic, projection, tempering) as a
  factor whose messages are the operation; what exists, what is missing, what to build first
- [[Factor Graphs]] — bipartite structure; the **two** acyclicity notions (`istree` vs `isdag`)
- [[Everything is a Factor]] — data, priors, losses and optimisers as graph nodes, forced by Remark 24
- [[Messages are Inversions]] — a factor → variable message **is** $c'_\pi$; both exclusion principles
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

- *the origin*: [[ImplicitREDDiff]] — the founding note of the project: relations instead of
  functions, input and output as masks chosen at inference time, and the energy the code computes
- *the forward process*: [[The VP-SDE]] — Song et al. 2021; the perturbation kernel, the score
  identity, Tweedie's denoiser, and the `tmin` floor nobody documents
- *the inversion*: [[RED-Diff as a Statistical Game]] — variational inference with a point-mass
  posterior; the stop-gradient; and **λ is derivable, not merely tunable**
- *the factor*: [[The Diffusion Factor]] — $P_{in}+P_{out}+P_{latent}=\mathrm{Id}$ becomes a
  `Polarity`; what a Dirac-valued message does to a factor graph
- *the implicit learner*: [[Implicit Diffusion Learners]] — the stop-gradient field is the exact
  gradient of a smoothed log-density; inference is a proximal point; the smoothing scale decides
  the relation · [[Inference Signatures]] · [[Backpropagation through Implicit Inference]] — the
  Lagrangian worked out, and a parabola learned from a circle by the adjoint alone ·
  [[Deterministic Relaxation]] — a DEQ whose layer is the denoiser ·
  [[The Implicit Diffusion Factor as a Statistical Game]]
- *composition*: [[Composing Diffusion Factors]] — intersections of learned relations by adding
  fields (a product of experts); conditioning as a special case; exactness at $t > 0$
- *verdict*: [[Open Problems in Implicit Diffusion Learning]] — theoretical problems (open or
  intrinsic) separated from missing implementation, each with its evidence
- *alternatives*: [[ProxDM and Proximal Alternatives]] — DPS, ΠGDM, ProxDM, plug-and-play, and
  why RED-Diff was implemented first; ProxDM is now implemented ([[proxdm]])
- *a conservative score*: [[energy_network]] — the network outputs an energy, $\varepsilon = \sigma_t\nabla E$;
  the learned relation gets a scalar energy, trained without nested AD in the loop
- *small networks, any AD backend*: [[backends]] — Zygote, Enzyme, ForwardDiff or Reactant
  through one field; a 5k-parameter MLP trained, inferred with and differentiated through

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

Implementation notes live next to the code, per [[Start Here]]:

**LenticulumCore.jl** — [[LenticulumCore]], [[abstract_types]], [[channels]], [[energy]],
[[open_model]], [[lens]], [[statistical_game]]

**Mycelium.jl** — [[Mycelium]], [[graph]], [[polarity_resolution]], [[messages]],
[[schedules]], [[passing]], [[free_energy]], [[factors]]

**Lenticulum.jl** — [[constraint]] (`beliefs.md` and `gaussian.md` are not yet written)

**VariationalDiffusion.jl** — [[VariationalDiffusion]], [[schedule]], [[predictor]],
[[reddiff]], [[factor]], [[analytic]], [[implicit]], [[implicit_factor]], [[energy_network]], [[proxdm]], [[backends]]

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
- [[RxInfer as a Backend]] — the nearest neighbour as an engine: gains for the probabilistic part,
  five things a replacement would lose, and a connector with learned relations as RxInfer nodes
- [[Why Julia]] — why not C++, Rust, PyTorch, JAX or Mojo: in Julia ordinary code *is* the compiled,
  differentiable computation graph, with evidence from this repository and the costs stated
- [[Parallelism and Compilation]] — parallelism, GPUs and XLA/MLIR in a *dynamic* SLAM
  setting. There is no junction tree and no parallelism today; a sweep is **quadratic in the
  number of factors** (measured); and the compile-vs-dynamic tension dissolves if you
  **compile the factor types, not the graph**
- [[README]] — the pitch
- [PhD Proposal](https://github.com/MathStruct/Lenticulum.jl/blob/master/meta/PhD%20Proposal.md)
  and [PhD Proposal v2](https://github.com/MathStruct/Lenticulum.jl/blob/master/meta/PhD%20Proposal%20v2.md)
  — the same programme argued two ways: problem-first and capability-first. Working material
  in `meta/`, not part of the published vault
