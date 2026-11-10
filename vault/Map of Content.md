#overview

> Every note of the vault, grouped by the three paths the library is built around and, within
> each, in reading order. Conventions and layout: [[Start Here]]. The state of the code after the
> refactor of November 2026, and what is parked: [[Refactor 2026-11]]. General category theory
> (Para, lenses, Markov categories, statistical games) is in the
> [CT-ML wiki](https://mathstruct.org/CategoryTheory-ML-Wiki/).

**References:** [[Bibliography]], generated from `docs/src/refs.bib`.

## Foundations

What a factor is, independent of any family: channels, polarity, energies, inversions.

- [[Implicit Learners]] — the three model families, and why the multivariate energy is what makes them work
- [[Factors are Parameterized Statistical Games]] — Definitions 20 and 27: the definition Lenticulum factors satisfy, and what actually gets optimised
- [[Channels and Polarity]] — reconciling AutoBayes' X/⟦c⟧/Y with the README's $P_{in} + P_{out} + P_{latent} = \mathrm{Id}$
- [[Inversions and Bayesian Lenses]] — Definitions 9–11: Bayes' law as a chain rule, and the inversion as a free choice
- [[Open Models and Latent Channels]] — Definitions 1–8: a kernel with an explicit latent space, and how the DAG restriction is lifted
- [[Energies and Entropies]] — The vault and the code use "energy" for at least seven different objects and "entropy" for five, and they are easy to confuse.
- [[Scalar and Multivariate Energy]] — the design decision that is ours, not the paper's: two energies, and the chain rule adapted to the multivariate one
- [[AutoBayes Examples as Factor Graphs]] — Appendix A, worked through
- [[AutoBayes to Lenticulum]] — the full naming dictionary paper → Julia
- [[Depth in Implicit Learning]] — is there a "no deep learning theorem"? Yes on the linear-Gaussian fragment and no elsewhere — it is the $d=1$ case of a degree bound the vault already derived; and depth-as-computation and depth-as-expressivity come apart
- [[Lux as a Parametric Lens]] — the concrete Lux.jl ↔ lens dictionary
- [[The Table Revisited]] — the README's table is an illustrating example (polynomials → varieties, the one pair where every slot becomes a theorem), read row by row. Holds up; the one real caveat is the Loss row collapses without a contrastive term; "symmetric" means *acausal*; the whole table is the $T\to0$ limit of the AutoBayes layer
- [[Three Senses of Implicit]] — the word means three independent things: implicit likelihood, implicit computation, implicit relation. Only the third buys you a polarity

## Path A: Gaussian factor graphs

Exact inference on linear-Gaussian graphs: beliefs, messages, schedules, the Bethe free energy,
and what changes on a loop. Code: `LenticulumCore`, `Mycelium`, `Lenticulum` (`gaussian.jl`, `nonlinear.jl`).

- [[Factor Graphs]] — bipartite structure; the two acyclicity notions (`istree` vs `isdag`)
- [[Everything is a Factor]] — data, priors, losses and optimisers as graph nodes, forced by Remark 24
- [[Beliefs]] — what travels on an edge: [[Trivial Belief]], [[Dirac Belief]], [[Gaussian Belief]], [[Sample Belief]], [[Categorical Belief]], [[Mixture Belief]], and how they pool (`combine`)
- [[Belief Algebra]] — every other operation (addition, mixture, logic, projection, tempering) as a factor whose messages are the operation; what exists, what is missing, what to build first
- [[Messages are Inversions]] — a factor → variable message is $c'_\pi$; both exclusion principles
- [[Polarity Resolution]] — target + available messages → a `Polarity`; the two legality checks
- [[Schedules]] — the schedule zoo; tree exactness; pruning by edge direction
- [[The Linear Gaussian Chain]] — the worked example: GTSAM's `OdometryExample`, exact marginals against the joint information matrix, and what this library is *not*
- [[Bethe Free Energy]] — Theorem 23 generalised to graphs; energies add, entropies get a counting correction
- [[Loopy Message Passing]] — what breaks off a tree, and the monodromy problem
- [[Composing Optimisers]] — damping, momentum, Anderson, SQUAREM and mirror maps as modifiers of a map: fixed points compose, convergence does not, and the spectrum of the linearised sweep says which modifier helps
- [[SLAM and Sensor Fusion]] — The canonical factor-graph application, and the one this project already has a worked example of ([[The Linear Gaussian Chain]]).
- [[Acausal Composition is a Hypergraph Category]] — how much structure it takes to wire an arbitrary graph rather than a DAG; a variable node is a Frobenius spider, and improper Gaussian beliefs are what make that work
- [[Dynamic Graphs]] — adding and deleting factors while the graph runs (SLAM); the Bethe free energy as the criterion; edges of a diffusion factor as noise levels
- [[Factor Graphs and Schemes]] — singular evidence, dual-number tangents and linear sheaf consistency
- [[Unusual Wirings]] — an output into a weight, an energy into an input, beliefs on weights and energies; risk measures as scalarisations

The belief types, one note each:

- [[Trivial Belief]] — `TrivialBelief()` is the belief that carries no information: the unit of [[Beliefs|`combine`]].
- [[Dirac Belief]] — `DiracBelief(x)` is a point mass $\delta_x$: certainty about a value.
- [[Gaussian Belief]] — `GaussianBelief(η, Λ)` is a Gaussian in canonical (information) form, $\mathcal N^{-1}(\eta, \Lambda) \propto \exp(-\tfrac12 x^\top\Lambda x + \eta^\top x)$, with precision $\Lambda = \Sigma^{-1}$ and
- [[Categorical Belief]] — `CategoricalBelief(p)` is a distribution over a finite set of states: a vector of probabilities, stored as log-probabilities.
- [[Mixture Belief]] — `MixtureBelief(components, weights)` is a weighted sum ``\sum_k w_k\,b_k`` of beliefs of one type.
- [[Sample Belief]] — `SampleBelief(samples, weights)` represents a distribution by weighted particles: the general-purpose fallback when no closed form is available, and the natural output of a sampler.

## Path B: relation learners

A diffusion model as a relation: the residual field, inference by root finding, the adjoint,
what the learner can represent, and what does not work yet. Code: `VariationalDiffusion`.

- [[The VP-SDE]] — Implemented in `lib/VariationalDiffusion.jl/src/schedule.jl`; see [[schedule]].
- [[The Diffusion Family]] — Entry point for the third of the three [[Implicit Learners]] families, worked out the way [[Algebraic Implicit Learners]] works out the first.
- [[Implicit Diffusion Learners]] — The derivation behind [[ImplicitREDDiff]], carried through.
- [[Inference Signatures]] — "Inference" for an implicit diffusion learner is not one map.
- [[Backpropagation through Implicit Inference]] — The Lagrangian of the original sketch, carried through: state equation, adjoint equation and gradient for an implicit diffusion learner.
- [[Deterministic Relaxation]] — The implicit diffusion learner with its randomness removed.
- [[Universal Approximation for Implicit Diffusion Learners]] — What an implicit diffusion learner can represent.
- [[The Diffusion Factor]] — Where [[ImplicitREDDiff]]'s selection matrices stop being notation.
- [[The Implicit Diffusion Factor as a Statistical Game]] — Is an implicit diffusion learner a [parameterized statistical game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)?
- [[Composing Diffusion Factors]] — Several learned relations on the same variables can be pooled: add their fields, and the answers of the sum are the configurations that satisfy all of them, e.g.
- [[Open Problems in Implicit Diffusion Learning]] — The honest ledger of the implicit diffusion learner, in two parts.
- [[MNIST in Both Directions]] — (later) one model of (image, label): classify, draw, and inpaint-and-classify by choosing ρ; the interfaces it needs first
- [[Energy-Based Factor Graphs]] — LeCun §6 is `Mycelium` without the beliefs. min-sum is the $T\to0$ limit of sum-product, so a `DiracBelief` message is a min-sum message and three packages' "missing entropy" is the Bethe form degenerating correctly
- [[Energy-Based Learning]] — the framework; why [[README]]'s table is this paper's table; loss functionals, the concept Lenticulum has no slot for; and the collapse problem, which `gaussian.jl` already guards against under the name `complexity`
- [[ImplicitREDDiff]] — The founding note of the project.
- [[Kernel Methods for Implicit Learning]] — Can kernel methods learn relations?
- [[Motivating Examples]] — the overview, and the six properties they share
- [[ProxDM and Proximal Alternatives]] — [[ImplicitREDDiff]] ends: *"Alternatively use [ProxDM](https://arxiv.org/pdf/2507.08956) to achieve a prox."* This note is what that alternative is, and how the whole family of diffusion-based inverse
- [[RED-Diff as a Statistical Game]] — Implemented in `lib/VariationalDiffusion.jl/src/reddiff.jl`; see [[reddiff]].
- [[Training Energy-Based Models]] — Song & Kingma, Du & Mordatch. Two of the three standard EBM training methods are already implemented here under other names: score matching is `VariationalDiffusion`, NCE is `Adversarial.RatioFactor` — and NCE with a known noise distribution is the `belief_logdensity` [[messages]] §1 has always wanted
- [[Why Learned Relations Do Not Collapse]] — how each learnable factor family avoids (or does not avoid) learning a relation that holds everywhere: normalisation in the data channels, the `output` option, log partitions, comparison; what is guarded and what is not (the DEQ)

## Path C: structure and regimes

Switching and regime factors, hierarchical models, expectation maximisation and the search over
structures. Code: `Lenticulum` (`constraint.jl`, `switching.jl`, `regime.jl`, `m_steps.jl`), `Mycelium`
(`cluster.jl`, `plates.jl`), `MyceliumLearning` (`em.jl`).

- [[Switching Constraint Factors]] — a linear constraint whose right-hand side a discrete outcome chooses: the smallest factor that leaves the Gaussian class, closed form in every direction; compositions are exactly the finite mixtures; mean-field messages
- [[Hierarchical Models]] — a hyperprior is one more factor on a shared variable; partial pooling is two polarities; eight schools, random slopes, crossed effects
- [[Finding a Factor Graph]] — explore, propose, fit and revise; Gaussian structure search, graphical lasso, and measured limits of Bethe + BIC
- [[The Regulariser and the Loss]] — the other two parts: the regulariser and the inferencer are inside one joint and one free energy; the optimiser and the loss are outside it, learnable only in a nested problem, with three conditions for the parts to compose
- [[The Inferencer and the Optimizer]] — the design note: meta-graphs over the base graph's messages and parameters. Half of it already exists as `OptimiserFactor` and `AmortisedInversion`; it is loop-free because it is coordinate descent on one objective; and it is the separation principle, exact for linear-Gaussian and failing into *exploration*
- [[The Two-Part Diagram]] — the shared shape is the wiring, which is a trace and already solved. What differs is the objective: one function (EM, VAE, active inference, LQG) fits the Bethe free energy; minimax (GANs) and general bilevel (actor–critic) do not

## Design

Design notes that cut across the paths.

- [[Backward Signatures]] — the backward pass at solver, factor and graph level: forward arguments + forward result + cotangent; `rrule`s on top (the solver's is built); unrolled vs adjoint message passing
- [[Composing Stochastic Factors]] — what the experiments of the branch `experiment-qm` showed: wiring, not depth, buys expressivity; the diffusion bound pins switching chains; learning needs exact blocks and mean-field iteration
- [[Extending Mycelium]] — the plan: a free energy per message rule, plates, rule-aware schedules, EP and mixture reduction, cluster nodes, structure edits
- [[Parallelism and Compilation]] — parallelism, GPUs and XLA/MLIR in a *dynamic* SLAM setting. There is no junction tree and no parallelism today; a sweep is quadratic in the number of factors (measured); and the compile-vs-dynamic tension dissolves if you compile the factor types, not the graph
- [[Probabilistic Types]] — "probabilistic type" names three different things. Beliefs are not the probability monad (the merge operation is not monadic), but $r_\theta \approx 0$ is a graded type judgment with the energy as its grade — and the grade lives in a semiring, which is the temperature
- [[Related Julia Projects]] — where this sits in the Julia ecosystem, and when to use something else. The nearest neighbour is `RxInfer.jl`; `IncrementalInference.jl` has already solved the `combine` gap by kernel-density BP; and Catlab's `oapply` is the subgraph-as-factor operation the vault records as missing
- [[RxInfer as a Backend]] — the nearest neighbour as an engine: gains for the probabilistic part, five things a replacement would lose, and a connector with learned relations as RxInfer nodes
- [[Sharing Types with RxInfer]] — (later) wrap ExponentialFamily types as beliefs instead of converting; reuse RxInfer's linearisation and unscented code for hand-written nonlinear factors (`NonlinearFactor` now does both itself)
- [[The Structural Gap to ModelingToolkit]] — the question in the other direction: what does MTK contain that Lenticulum does not? Five ranked items; two are not addable
- [[The Type Discipline of a Factor Graph]] — a factor graph is neither linear nor cartesian but Frobenius; putting polarity in the type domain already caught a real bug; and five separately-recorded gaps turn out to be one typing problem
- [[Time as a Base]] — the design note: what an MTK × Lenticulum extension with a time dimension would be. A base change, not a redesign — variables carry trajectories, and a belief over a trajectory *is* a chain factor graph
- [[Why Julia]] — why not C++, Rust, PyTorch, JAX or Mojo: in Julia ordinary code *is* the compiled, differentiable computation graph, with evidence from this repository and the costs stated

## Project records

- [[Refactor 2026-11]] — the layout after the refactor, what is verified, what is parked and how it comes back
- [[Problem Log]] — `LP-nnn`: what does not work yet and needs a decision, from GitHub issues to gaps in the theory, e.g. [[LP-002 The Jacobian side is not implemented]]
- [[Decisions]] — `L-ADR-nnnn`: what was chosen, by whom, and what follows. Among them [[L-ADR-0002 Energies are vectors, scalarised late]] and [[L-ADR-0003 Pointwise energy and free energy are separate levels]]
- [[Specifications]] — `L-SPEC-nnn`: contracts precise enough to test against, starting with [[L-SPEC-001 The Factor Interface]]
- [[Feature Backlog]] — every feature still to implement, with its source, its motivation and a proposed priority
- [[Research Questions]] — open mathematical questions (`RQ-nnn`)

## Implementation notes

One note beside each source file (`lib/*/src/*.md`); each package's overview note lists them:
[[LenticulumCore]], [[Mycelium]], [[MyceliumLearning]], [[Lenticulum]], [[VariationalDiffusion]].

## Parked

Notes on families and applications that are not on the three paths. They were not revised in the
refactor and may describe code that is parked too (`parked/`). See [[Refactor 2026-11]].

**Algebraic**

- [[Algebraic Implicit Learners]] — Entry point for the first of the three families in [[Implicit Learners]]: the approximator is an algebraic variety, inference is root finding, and the backward pass is the implicit function theorem (p
- [[Algebraic Statistics Bridge]] — The observation that makes this whole family more than a curiosity for Lenticulum: conditional independence is a polynomial constraint, so a Bayesian network *is* an algebraic variety, and marginalisa
- [[Algebraic versus Geometric Distance]] — Derivation.
- [[Backpropagation by the Implicit Function Theorem]] — Derivation.
- [[Branches and the Discriminant]] — The most satisfying concrete grounding in this note set: for an algebraic factor, the abstract latent space $\llbracket c \rrbracket$ of [Definition 1](https://mathstruct.org/CategoryTheory-ML-Wiki/Op
- [[Composition is Elimination]] — Composing two algebraic factors into one is a Gröbner basis computation.
- [[Differential Algebra and DAE Factors]] — Where differential algebra genuinely belongs — and it is not backpropagation.
- [[Fitting is a Nullspace Problem]] — Derivation and implementation.
- [[Grassmann Mixing and Plücker Coordinates]] — Zhang (2025) replaces self-attention by features of the 2-planes spanned by pairs of token states, written in Plücker coordinates.
- [[Inference as Root Finding]] — The forward pass of an algebraic factor.
- [[Open Problems in Algebraic Implicit Learning]] — The honest ledger.
- [[Symbolic Implicit Learning]] — Learn a relation $r(z) = 0$ as a formula: not a polynomial with fixed monomials, not a network, but a short symbolic expression found by search.
- [[The Algebraic Factor as a Statistical Game]] — Assembling everything into [AutoBayes Definition 20](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) and [Definition 27](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#
- [[The Parameter is a Grassmannian]] — Derivation.
- [[The Veronese Parametrisation]] — The single structural fact that makes the algebraic family tractable: the residual is linear in the parameter.
- [[Toric Models as Factors]] — A toric model is a discrete exponential family whose sufficient statistics are integer vectors: $p(x) \propto \theta^{a_x}$, a monomial in the parameters.
- [[Universal Approximation by Nash-Tognoli]] — [[README]]'s table claims the implicit analogue of Weierstraß is "compact smooth manifolds via Nash–Tognoli".
- [[Varieties Ideals and Real Nullstellensatz]] — The minimum algebra needed, and the three places where working over $\mathbb{R}$ — which we must — breaks the textbook complex picture.

**Equilibrium**

- [[DEQ as a Relation]] — A deep equilibrium model is defined by a relation and shipped as a function.
- [[NeuralODE as an Invertible Factor]] — A continuous ODE flow is invertible wherever its solution is unique in both directions.
- [[The Equilibrium Family]] — Entry point for the second of the three [[Implicit Learners]] families, implemented in `parked/ImplicitLayers.jl`.

**Adversarial**

- [[GANs as Two Factors]] — Can the GAN diagram be achieved with two factors?
- [[Implicit Generative Models]] — Entry point for the adversarial family, implemented in `parked/Adversarial.jl`.

**Applications**

- [[Climate and Dynamical Systems]] — Data assimilation is factor-graph inference with different vocabulary, an enormous state, and a sixty-year head start.
- [[Energy Markets and Power Grids]] — The domain where the physics and the economics are both networks of constraints, and where the standard method is already a factor graph under another name.
- [[Language Models]] — a product of diffusion experts, not a mixture; why the framing clarifies and the machinery does not
- [[Metabolomics and Proteomics]] — Where the exact relation is stoichiometry, the system is famously underdetermined, and a posterior therefore says something a point estimate structurally cannot.
- [[Molecular Dynamics]] — The domain where the energy-based reading is not an analogy — a force field already *is* a sum of local energy terms over a graph, which is [[Energy-Based Factor Graphs]] §1 written in chemistry.
- [[Trading and Financial Markets]] — The domain where the residual is the product.

**Comparisons**

- [[Geometric Deep Learning and Physical Laws]] — equivariance constrains the map, a law constrains the configuration; $\dot x=-x$ is equivariant and conserves nothing. The measured case: direct-force equivariant potentials are unstable in MD (Bigi–Langer–Ceriotti, ICML 2025). And the converse — this project has no equivariance
- [[ModelingToolkit as an Acausal Relation]] — the other bridge: MTK is the implicit column of [[README]]'s table minus the probability; Willems' behaviors; why MTK's incidence graph *is* a factor graph, and why MTK solves where Mycelium propagates
- [[Prolog and Logic Programming]] — "the Prolog of machine learning", scored honestly. Modes are polarities exactly; Prolog is the *boolean* semiring of the same framework; and the shortfall is one thing — schemas versus ground instances
