# Index — Lenticulum concept vault

This is the map of content for the theory behind Lenticulum.jl. Two papers are the
backbone, and one bridge layer connects them to Julia:

- **[Categorical Foundations of Gradient-Based Learning](https://arxiv.org/html/2103.01931v2)**
  (Cruttwell, Gavranović, Ghani, Wilson, Zanasi) — how *ordinary* deep learning is a
  **parametric lens**. This is the theory Lux.jl already implements without saying so.
- **[AutoBayes](https://arxiv.org/html/2503.18608v2)** (St Clere Smithe & Perin) — how
  *Bayesian* inference is a **statistical game**, which is a parametric lens whose
  backward pass is a posterior rather than a gradient. This is what Lenticulum.jl
  implements.

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

## 3. The bridge to Lenticulum.jl

19. [[Channels and Polarity]] — reconciling AutoBayes' X/⟦c⟧/Y with the README's
    $P_{in} + P_{out} + P_{latent} = \mathrm{Id}$
20. [[Scalar and Multivariate Energy]] — **the design decision that is ours, not the
    paper's**: two energies, and the chain rule adapted to the multivariate one
21. [[Implicit Learners]] — the three model families, and why the multivariate energy
    is what makes them work
22. [[AutoBayes to Lenticulum]] — the full naming dictionary paper → Julia
23. [[ModelingToolkit as an Acausal Relation]] — **the other bridge**: MTK is the implicit
    column of [[README]]'s table minus the probability; Willems' behaviors; why MTK's
    incidence graph *is* a factor graph, and why MTK solves where Mycelium propagates
24. [[Acausal Composition is a Hypergraph Category]] — how much structure it takes to wire an
    arbitrary graph rather than a DAG; a variable node is a **Frobenius spider**, and improper
    Gaussian beliefs are what make that work
25. [[The Structural Gap to ModelingToolkit]] — the question in the other direction: **what
    does MTK contain that Lenticulum does not?** Five ranked items; two are not addable
26. [[Time as a Base]] — **the design note**: what an MTK × Lenticulum extension with a time
    dimension would be. A base change, not a redesign — variables carry trajectories, and a
    belief over a trajectory *is* a chain factor graph

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

## Existing notes

- [[README]] — the pitch
- [[ImplicitREDDiff]] — the diffusion-based implicit learner
