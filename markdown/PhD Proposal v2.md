# PhD Proposal (second version) — Calibrated Uncertainty for Acausal Network Models

**One graph, mixing known physics with fitted components, answering any direction of question
— and reporting how much to trust the answer.**

## Summary

Network models in engineering are written *acausally*: as equations relating quantities, with
no distinguished input or output. Tools that solve them return point estimates. Tools that
return posteriors require the model to be written causally, as a directed generative process,
and cannot express a physical law that holds only approximately.

This project builds the object that does both — a factor graph whose nodes are *relations*
rather than functions, on which one inference procedure produces posteriors and a
model-comparison score. A verified prototype exists. This proposal sets out four capabilities
it makes possible, each with the specific mathematical problem standing in the way.

## 1. What already works

A Julia implementation of six packages, ~740 tests, each family checked against a closed form
rather than against itself. Three results are worth stating because they are exact, not
approximate:

- **Posteriors and evidence on a linear-Gaussian network.** On a chain of noisy measurements
  the graph reproduces the exact joint posterior — every mean and every variance, to machine
  precision, against a direct sparse solve. It also returns the exact negative log marginal
  likelihood, which it is never given: it sums local energies and entropies with a
  combinatorial correction, and the evidence is what remains.
- **The same equation solved in every direction.** An acausal constraint over ``n`` quantities
  admits ``n`` solve directions, none privileged, all exact. A resistive divider and a
  three-way conservation law are solved as written, with no rearrangement into input/output
  form.
- **Learned components as drop-in nodes.** A diffusion model, a deep equilibrium network and a
  neural ODE each wrap into the same interface as a physical equation, and the graph does not
  distinguish them.

## 2. What this makes possible, and what stands in the way

### C1. Model comparison for physical networks

Because the objective is a variational free energy that coincides with ``-\log p(\text{data})``
on the tractable fragment, a network model can be *scored*, not merely fitted — and two
competing models of the same system compared on the same footing, with model complexity
accounted for automatically rather than by a hand-chosen penalty.

> **Open problem — what does a mixed graph minimise?**
> Components with tractable densities return distributions; components wrapping a solver or an
> optimiser return points. Point-valued message passing is the zero-temperature limit of the
> distributional kind, so a graph containing both adds two quantities that are not
> commensurable. A second gap sits underneath: the objective is evaluated at the inferred
> configuration and contains no term penalising the rest of the configuration space, so it is
> degenerate for any model free to flatten its own energy surface — currently prevented only
> because noise models are fixed rather than learned.
>
> *Required:* a temperature or semiring discipline making mixed graphs well posed, and an
> objective with provable non-degeneracy under learned noise.

### C2. Grey-box networks with calibrated uncertainty

Part of a network is known physics; part is a fitted surrogate where the physics is unavailable
or too expensive. Today those live in separate tools and the uncertainty does not cross the
boundary. Here they are nodes in one graph, and the posterior spans both.

> **Open problem — composing beliefs that have no density.**
> Pooling two beliefs about the same quantity is exact addition in the Gaussian case and
> undefined otherwise, which blocks every non-Gaussian or sample-based component. Density-ratio
> estimation offers a route — a classifier separating two sample sets estimates their
> log-density ratio, and against a known reference recovers the density itself — but the
> resulting operation is not idempotent, hence not associative, while message passing assumes
> associativity; and its accuracy degrades silently as the effective sample size collapses.
>
> *Required:* an associative pooling operation for sample-based beliefs, with a computable
> error estimate and convergence conditions for message passing over it.

### C3. One model, every question

Estimation, prediction, control and design are the same model with different quantities held
fixed. Moving the clamps turns a state estimator into a simulator into a design problem,
without rewriting anything — which is what an acausal formulation is *for*, extended to the
probabilistic case.

> **Open problem — cycles.**
> Physical networks are loopy: conservation laws, mechanisms and reaction networks all close
> loops. On such graphs the method is imperfect in a specific and dangerous way — Gaussian
> loopy belief propagation returns **exact means and incorrect variances**. We reproduce this
> on a resistive divider, where every variance is inflated by a stable common factor. The point
> estimates are right and the error bars are wrong, with nothing reporting it, which for an
> engineering user is the worst available failure mode. Separately, a loop of purely acausal
> constraints with no priors cannot start: every message needs every other channel to be
> informative, so nothing propagates.
>
> *Required:* variance correction on loopy constraint graphs, a criterion for when to propagate
> versus assemble and solve directly, and consistent initialisation.

### C4. Models that carry proofs

A model used for engineering decisions should carry checkable guarantees about *when its
answers mean anything* — not a footnote in a paper. Several of the relevant conditions are
structural and therefore mechanisable.

> **Open problem — well-posedness that is decided rather than declared.**
> Whether a component may be solved in a given direction is currently a predicate the model
> author asserts. For differential-algebraic systems this is **structural identifiability**,
> and it is *decidable* by differential elimination. Related conditions are similarly
> structural: whether a residual is square, whether an assignment of solve-directions across a
> graph is globally consistent (a matching problem), whether a differential index exceeds one
> so that hidden constraints are being silently violated.
>
> *Required:* classify which conditions are decidable, and produce formally verified
> certificates for (i) legality of a direction assignment, (ii) exactness of a schedule on a
> given graph, (iii) non-degeneracy of a training objective. Proof assistants are the natural
> tool, and the underlying compositional structure is of a kind already being formalised.

### C5. Irregularly sampled dynamics

Sensors do not tick together. Variables currently carry values rather than trajectories, so
dynamics must be discretised by hand before entering a graph. The extension carries
trajectories with Gauss–Markov beliefs, whose joint precision is exactly block-tridiagonal — so
a belief over a trajectory *is itself* a chain graph, and querying it at an unvisited time is a
local interpolation rather than a re-solve.

> **Open problem.** Index reduction and consistent initialisation must happen *before* any
> belief is propagated, and both are symbolic operations the framework cannot perform.
> Non-Gaussian trajectory beliefs lose the sparsity the construction depends on.

## 3. Programme

In dependency order rather than strict sequence: **C1's problem first**, since a graph without
a well-defined objective cannot have its convergence analysed; then **C3's** as the numerical
core and the part with the most immediate engineering value; then **C2's** as the algebraic
core and the enabler for genuinely learned components. **C4** runs throughout — certificates
are cheapest to design alongside the properties they certify — and **C5** is the applied
extension, validated on a network model with real asynchronous data.

Validation is against closed forms wherever one exists, as the current test suites already are,
and against direct sparse solves elsewhere.

## 4. Fit

Applied analysis with a substantial computational component, aimed at network-structured
application areas — energy systems, flow and transport, reaction networks — where models are
already written acausally and where uncertainty quantification is wanted but hard to obtain.
It is Bayesian inverse problems with the causal restriction lifted.

It also connects directly to optimisation and control: a controller and an observer
interconnected with a plant is, in this formulation, three nodes sharing variables, and the
classical separation principle becomes the statement that the resulting problem decouples
exactly in the linear-quadratic-Gaussian case and only approximately otherwise — with the
approximation error being precisely the exploration term of dual control.

Implementation is in Julia, where the acausal modelling and scientific-computing ecosystem
already lives. The certification strand is proof-assistant work. Both are present in the
prototype's design rather than bolted on.

## References

- LeCun, Chopra, Hadsell, Ranzato, Huang. *A Tutorial on Energy-Based Learning*, 2006.
- Willems. *The Behavioral Approach to Open and Interconnected Systems*, IEEE Control Systems
  Magazine, 2007.
- Weiss & Freeman. *Correctness of belief propagation in Gaussian graphical models of arbitrary
  topology*, Neural Computation, 2001.
- Fong & Spivak. *Hypergraph Categories*, 2019.
- Stein & Samuelson. *A Category for Unifying Gaussian Probability and Nondeterminism*, CALCO
  2023.
- Anderson, Barfoot, Tong, Särkkä. *Batch nonlinear continuous-time trajectory estimation as
  exactly sparse Gaussian process regression*, Autonomous Robots, 2015.
- Song & Kingma. *How to Train Your Energy-Based Models*, 2021.
- Pantelides. *The consistent initialization of differential-algebraic systems*, SIAM J. Sci.
  Stat. Comput., 1988.
- Feldbaum. *Dual control theory* I–IV, Automation and Remote Control, 1960–61.
