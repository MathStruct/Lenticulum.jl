#design #comparison

> Could RxInfer.jl serve as the message-passing backend instead of `Mycelium`? **For the
> probabilistic part, yes, and it would be a gain**: a mature reactive engine, a large library
> of message rules, streaming inference and a well-tested belief algebra. **As a replacement for
> everything, no**: five things this project is built around have no counterpart there. The
> recommendation is a connector in both directions, not a replacement.

> Sources: Bagaev, Podusenko & de Vries, *RxInfer: A Julia package for reactive real-time Bayesian inference*, JOSS 2023; Bagaev & de Vries, *Reactive Message Passing for Scalable Bayesian Inference*, Scientific Programming 2023; Şenöz, van de Laar, Bagaev & de Vries, *Variational Message Passing and Local Constraint Manipulation in Factor Graphs*, Entropy 2021; the RxInfer documentation (custom nodes, deterministic nodes, non-conjugate inference), read October 2026
>
> Bibliography: [[Bibliography#^bagaev2023rxinfer|Bagaev et al. 2023]] · [[Bibliography#^bagaev2023reactive|Bagaev & de Vries 2023]] · [[Bibliography#^senoz2021local|\cSenöz et al. 2021]]
>
> Theory (CT-ML wiki): [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## 1. What RxInfer is

RxInfer does Bayesian inference by **reactive message passing** on Forney-style factor graphs,
built on `ReactiveMP.jl` (the message engine) and `GraphPPL.jl` (model specification with an
`@model` macro). From its documentation and papers:

- **algorithms**: belief propagation and variational message passing, chosen per node, with
  local constraints (mean-field and others) on a constrained Bethe free energy (Şenöz et al.);
- **beyond conjugacy**: non-conjugate inference by optimisation-based message updates, and
  **deterministic nonlinear functions** ("delta nodes") with approximation methods;
- **custom nodes**: `@node` declares a node and its interfaces, `@rule` gives one message rule
  **per interface**, i.e. per direction, `@average_energy` its term in the free energy;
- **parameter learning**, **real-time streaming** inference, and Bethe free energy for model
  assessment;
- messages from a library of distributions (exponential families, point masses, sample lists,
  mixtures), with closed-form products where they exist.

Earlier versions of this vault said that RxInfer's factors "are probability distributions from
known families". That undersold it: with delta nodes, non-conjugate updates and custom rules it
reaches well beyond conjugate models.

## 2. What it would bring as a backend

| this project's component | RxInfer's counterpart | gain |
|---|---|---|
| `Mycelium` graphs, schedules, `propagate!` | the reactive engine, schedule-free | maturity, speed, streaming |
| `combine`, densities, `moment_match` ([[Belief Algebra]]) | BayesBase / ExponentialFamily products and projections | far more families, tested |
| Bethe free energy (`bethe_free_energy`) | constrained Bethe free energy | local constraints, VMP |
| Gaussian and linear factors | a library of nodes and rules | breadth |
| — | delta nodes, non-conjugate updates | factors from **hand-written** nonlinear functions, $y = f(x)$ with $f$ given, which `Lenticulum` lacks (its nonlinear factors are learned: diffusion, DEQ, neural ODE) |

"Nonlinear" needs care here. A diffusion factor is as nonlinear as a factor gets, but its
nonlinearity is learned and implicit (a relation, answered by root finding in any direction).
A delta node is the other kind: the user writes $y = f(x)$, and RxInfer turns the known $f$ into
Gaussian messages by linearisation, the unscented transform or CVI, which is how an extended or
unscented Kalman filter is written there. A generic factor of that kind, `NonlinearFactor(f)`,
is what `Lenticulum` is missing ([[Open Problems in Implicit Diffusion Learning]] I11).

Everything on the **probabilistic, specified-model** side of this project would run faster and
more completely there. The localisation tutorial, for example, is a model RxInfer handles
directly.

## 3. What a replacement would lose

1. **Learned relations queried in any direction.** RxInfer's rules are per interface, so
   messages already flow in every direction; that part is *not* a difference. The difference is
   the factor itself: a relation **learned from data** (a diffusion model of the joint space, a
   DEQ), whose answer in a given direction is a **root-finding solve** rather than a rule written
   in advance. A custom node can call such a solver inside its rules, but nothing in RxInfer
   learns the relation or differentiates through that solve.
2. **Training through inference.** Factors here are trained by gradients through their own
   inference (the implicit-function-theorem adjoint, [[Backpropagation through Implicit Inference]])
   and the graph treats data, losses and optimisers as factors ([[Everything is a Factor]]).
   RxInfer learns parameters of specified distributions by inference over them; gradient training
   of a network inside a node, through the node's solver, is not part of it.
3. **Polarity as a type.** Which channels are observed, unobserved or latent is part of a
   factor's type here, checked for legality and resolved over the graph
   ([[Channels and Polarity]], [[Polarity Resolution]]). In RxInfer, observation is a property of
   the model's data, not of the factor's interface.
4. **Vector energies and statistical games.** Factors here carry graded and multivariate
   energies and compose as statistical games ([[Scalar and Multivariate Energy]],
   [[Factors are Parameterized Statistical Games]]); RxInfer's average energy is a scalar per node.
5. **Non-probabilistic factors.** An acausal constraint or a DEQ residual is a first-class factor
   here; in RxInfer it has to be phrased as a (degenerate) probability node.

None of these is a criticism of RxInfer: they are what this project is *for*, and they are the
least finished parts of it.

## 4. Recommendation: connect, do not replace

Keep `Mycelium` as the small reference engine that implements the theory, and add an RxInfer
**package extension** with three pieces, in increasing effort:

1. **Belief conversion** between this project's beliefs (Gaussian, categorical, mixture, Dirac,
   sample) and RxInfer's distribution types; also the connector [[Belief Algebra]] §6 asks for.
   **Done** (October 2026): `as_distribution` and `as_belief`, an extension of `LenticulumCore`
   that loads with ExponentialFamily.jl ([[interop]]). Gaussians convert exactly, canonical form
   to canonical form; pooling agrees with RxInfer's `prod`; and a one-step conjugate model
   solved by RxInfer and by `Lenticulum`'s Gaussian factors gives the same posterior.
2. **A shared benchmark**: the localisation tutorial's graph run in both, with identical
   posteriors and free energy, which makes the overlap and the difference concrete.
3. **A learned relation as an RxInfer node.** The mapping is unusually direct: an incoming
   Gaussian message on a channel is the implicit learner's **soft clamp** (anchor ``z_0`` and
   precision ``\rho``, [[Inference Signatures]] #4), and the outgoing message is the
   **Laplace Gaussian** that `implicit_laplace` already computes. RxInfer users would get learned,
   direction-free relations as nodes; this project would get RxInfer's engine around them.

Item 3 is the real backend question answered in the useful direction: RxInfer as the engine
*around* learned relations, with the relations staying in this project.

Whether `Mycelium` could store RxInfer's own types instead of converting, and how a
hand-written nonlinear factor could reuse RxInfer's approximations: [[Sharing Types with RxInfer]]
(noted for later).

Related: [[Sharing Types with RxInfer]], [[Related Julia Projects]], [[Belief Algebra]], [[Inference Signatures]],
[[Messages are Inversions]], [[Why Julia]], [[The Two-Part Diagram]]
