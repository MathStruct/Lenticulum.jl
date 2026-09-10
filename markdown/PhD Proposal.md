# PhD Proposal — Inference on Factor Graphs of Implicit Relations

**Uncertainty quantification for acausal network models whose components may be learned.**

## Summary

Physical network models — power grids, flow networks, reaction systems — are written
*acausally*: as equations relating quantities, with no distinguished input or output.
Probabilistic inference, by contrast, is almost always formulated *causally*, over directed
generative models. The two do not currently meet: acausal modelling tools solve hard equations
and return points; probabilistic tools return distributions but cannot express a physical law
that holds only approximately, or a component that was fitted rather than derived.

This project builds the object in between: a **factor graph whose factors are relations**,
each of which may be an exact physical equation, a soft constraint with a noise model, or a
learned surrogate — and on which a single inference procedure produces posteriors, not just
estimates. The prototype exists and is verified against closed forms. The proposal is to
resolve the open problems that prevent it from being trustworthy, and to make its
well-posedness conditions machine-checkable rather than asserted.

## 1. Setting

A **factor** is a relation among named channels, ``r_\theta(x_1,\ldots,x_n) \approx 0``, with
no fixed direction: which channels are inputs is decided per call. Factors are wired into a
bipartite graph and inference is message passing. The objective is a variational free energy
which, on the linear-Gaussian fragment, equals the exact negative log marginal likelihood —
so the framework can *score* a model, not merely fit one.

This subsumes three things usually kept apart:

| | is a factor with |
|---|---|
| an acausal physical equation | a fixed residual and a noise covariance |
| a Bayesian prior or measurement | a density |
| a learned component (surrogate, neural operator, generative prior) | parameters |

The point is that inference does not care which is which, so a graph may mix known physics
with fitted components and report calibrated uncertainty over the whole thing.

## 2. Starting point

A working Julia implementation exists: six packages, ~740 tests, each factor family checked
against a closed form rather than against itself. It includes linear-Gaussian and acausal
constraint factors, deep equilibrium and neural-ODE factors, a diffusion-model factor, and
generative/density-ratio factors, together with graph construction, scheduling and free-energy
accounting.

Equally important, the failures are documented rather than hidden: the repository carries an
extensive written record of what does not work and why. The open problems below are therefore
*mapped*, not speculative — each is a recorded, reproduced failure with a known boundary.

## 3. Major open problems

### P1. Composing beliefs that have no density

Pooling two beliefs about the same quantity is exact addition in the Gaussian case and
undefined otherwise. Everything downstream — particle messages, non-Gaussian priors, learned
components that only sample — is blocked on this single operation.

Density-*ratio* estimation offers a route: a classifier separating two sample sets estimates
their log-density ratio, and against a known reference distribution recovers the normalised
density itself. But the resulting pooling is **not idempotent**, hence not associative, while
message passing assumes it is; and its accuracy degrades silently as the effective sample size
collapses.

*Open:* an associative pooling operation for sample-based beliefs, with a computable error
estimate and conditions under which message passing over it converges.

### P2. Inference on graphs with cycles

Physical networks are loopy — Kirchhoff's laws, mechanisms, reaction networks all close loops.
On such graphs the method is provably imperfect in a specific way: Gaussian loopy belief
propagation returns **exact means and incorrect variances**. We reproduce this on a resistive
divider, where every variance is inflated by a stable common factor. For an engineering user
this is the worst failure mode available: the point estimates are right and the error bars are
wrong, with nothing reporting it.

There is a second, sharper failure: a loop of purely acausal constraints with no priors cannot
*start* — every message requires every other channel to be informative, so nothing propagates
and the result is a converged-looking fixed point containing no information.

*Open:* variance correction on loopy graphs of physical constraints; a principled criterion for
when to propagate and when to assemble and solve directly; and consistent initialisation.

### P3. What does a mixed graph actually minimise?

Two distinct regimes are currently summed as if commensurable. Factors with tractable
densities return distributions and behave like ordinary variational inference. Factors wrapping
a solver, an optimiser or a generator return **points** — which is not a defect but a different
semiring: point-valued message passing is min-sum, the zero-temperature limit of sum-product.
Mixing the two adds a zero-temperature energy to a unit-temperature free energy and calls the
result a free energy.

Underneath is a second gap. The objective is evaluated at the inferred configuration and
contains no term raising the energy elsewhere — in energy-based-learning terms it is the
*energy loss*, which collapses for any architecture that can flatten its own energy surface.
It is safe at present only because every noise model is a fixed hyperparameter, and it stops
being safe the moment those are learned.

*Open:* a temperature or semiring discipline that makes mixed graphs well defined, and a loss
functional for factor graphs with provable non-degeneracy.

### P4. Well-posedness that is decided rather than declared

Whether a factor may be run in a given direction is currently a predicate the model author
asserts. For differential-algebraic systems this is **structural identifiability**, and it is
*decidable* — by differential elimination. Related conditions are similarly structural: whether
a residual is square, whether an assignment of solve-directions across a graph is globally
consistent (a matching problem), whether a differential index exceeds one and hidden
constraints are being violated.

This is where the project should stop asserting and start proving. A model that will be used
for engineering decisions ought to carry machine-checked certificates for the properties that
make its answers meaningful.

*Open:* classify which well-posedness conditions are decidable; produce formally verified
certificates for (i) legality of a direction assignment, (ii) exactness of a given schedule on
a given graph, (iii) non-degeneracy of a training objective. Proof assistants are the natural
tool; the underlying categorical structure (relations composing as a hypergraph category) is
already of a kind being formalised.

### P5. Continuous time

Variables currently carry values, not trajectories, so dynamics must be discretised by hand
before entering a graph. The natural extension carries trajectories with Gauss–Markov beliefs,
whose joint precision is exactly block-tridiagonal — so a belief over a trajectory *is itself*
a chain factor graph, and querying it at an unvisited time is local interpolation. This is what
irregularly-sampled sensor data needs.

*Open:* index reduction and consistent initialisation must happen before any belief is
propagated, and both are symbolic operations the framework cannot currently perform;
non-Gaussian trajectory beliefs lose the sparsity that makes the whole construction viable.

## 4. Programme

Roughly, and in dependency order rather than strict sequence:

1. **P3 first**, because it decides what the other results mean. A graph without a
   well-defined objective cannot have its convergence analysed.
2. **P2** — the numerical core, and the part with the most immediate engineering value.
3. **P1** — the algebraic core, and the enabler for genuinely learned components.
4. **P4** in parallel throughout, since certificates are cheapest to design alongside the
   properties they certify.
5. **P5** as the applied extension, validated on a network model with real, asynchronous data.

Validation is by closed form wherever one exists — the existing test suites are built this way
— and against direct sparse solves elsewhere.

## 5. Fit

The work is applied analysis with a substantial computational component. It sits naturally
against network-structured application areas — energy systems, flow and transport, reaction
networks — where models are already written acausally and where uncertainty quantification is
wanted but hard to obtain. It is Bayesian inverse problems with the causal restriction
removed, and it connects directly to optimisation and control: a controller and an observer
interconnected with a plant is, in this formulation, three factors sharing variables, and the
classical separation principle is the statement that the resulting problem decouples exactly
in the linear-Gaussian case and only approximately otherwise.

Implementation is in Julia, which is where the acausal modelling and scientific-computing
ecosystem already lives; the certification strand (P4) is proof-assistant work. Both halves are
already present in the prototype's design, and neither is bolted on.

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
