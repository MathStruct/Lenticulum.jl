# The Inferencer and the Optimizer

> The proposal: attach **factor graphs to the factor graph**. One whose job is to aid
> *inference* on the base graph, one whose job is to *optimise* it.
>
> Two findings. First, **half of it is already implemented** — as two unrelated features that
> nobody had connected. Second, the reason it comes out loop-free is not a happy accident: it
> is the $f = g$ row of [[The Two-Part Diagram]] §3, and it is the **separation principle** of
> control theory, with the same conditions and the same failure mode.
>
> Nothing here is implemented as described. This is a design note.

## 1. The proposal, stated precisely

A base factor graph has three kinds of state:

| state | lives on | who computes it now |
|---|---|---|
| beliefs / messages | **edges** (two per edge, one each way) | each factor's local rule |
| parameters `ps` | **factors** | an external training loop |
| the graph itself | — | you, by hand |

The proposal is to make the first two the *variables of further factor graphs*:

$$
\underbrace{\mathcal{I}}_{\text{Inferencer}} \;\longrightarrow\; \text{the base graph's \textbf{messages}}
\qquad\qquad
\underbrace{\mathcal{O}}_{\text{Optimizer}} \;\longrightarrow\; \text{the base graph's \textbf{parameters}}
$$

That is a sharp combinatorial statement rather than a metaphor:

- **The Inferencer is a factor graph whose variables are the base graph's directed edges.** A
  message lives on a directed edge, so a graph over messages is a graph over $2|E|$ variables.
  Its factors say how a message should be updated given its neighbours — which is what the BP
  update rule already is, written by hand.
- **The Optimizer is a factor graph whose variables are the base graph's factors** — one
  variable per parameter block. Its factors say how a parameter should move given the local
  energies.

Both are *meta*-graphs: their variables are the base graph's state, not the base graph's
variables.

## 2. Half of it exists, in two places that never met

### The Optimizer already exists

[[Everything is a Factor]] took this position from the start — *"optimizers (like ADAM, …) and
losses become nonparametric factors in our computation graph"* — and `Mycelium` implements it:
`OptimiserFactor`, `AbstractUpdateRule`, `GradientDescent`, `Momentum`, `Nesterov`,
`rule_get` / `rule_put` / `optimiser_step`.

The theory is Cruttwell et al.'s Definition 3.14: a stateful update is a lens
$U : (S\times P,\ S\times P) \to (P, P')$, which [[Learning Components as Parametric Lenses]]
covers. So an optimiser is already a factor, already in the graph, already carrying its own
state.

What it is *not* is a **graph**. It is a per-factor node with a fixed update rule, not a
network with structure across factors and certainly not a learned one.

### The Inferencer exists, per factor

`LenticulumCore.AmortisedInversion(net)` is documented as:

> *$c'$ realised by a learned network — a VAE encoder. `net` is an `AbstractLuxLayer`, so the
> inversion carries its own parameters, independent of the forward kernel's. That a factor has
> **two independently parametrised halves** is the structural reason a factor cannot be a Lux
> layer.*

That is an Inferencer, scoped to one factor: a learned thing whose job is to do inference
better than the analytic rule would. `Adversarial.RatioFactor` is the first factor where it is
literally true — the discriminator is trained separately from what it scores.

What it is *not* is **graph-wide**. Each factor amortises its own inversion in isolation;
nothing learns to schedule, to damp, or to route information across the graph.

> [!important] The finding
> The Optimizer-as-factor and the amortised Inferencer were designed independently, live in
> different packages, and are **the same construction at two different scopes**. Both attach a
> learned or stateful process to part of the base graph and let it act on that part's state.
>
> Seeing them as one thing is what makes "attach a factor graph to the factor graph" the
> natural generalisation rather than a new idea.

## 3. Why the Inferencer is worth having

Belief propagation's update rule is *fixed*: pool the other messages, invert the factor, send.
That rule is exact on a tree and has no guarantees off one — and this project keeps running
into exactly that:

- loopy Gaussian BP gets exact means and wrong variances, by a stable factor of five in the
  divider test ([[ModelingToolkit as an Acausal Relation]] §6);
- a loopy acausal graph cannot even bootstrap without weak priors (`constraint.md` §4.2);
- schedules are hand-picked, and `report.converged` means the messages stopped moving, not
  that the answer is right.

Every one of those is a place where a *learned* update rule could do better than the analytic
one, because the analytic one is optimal only in a regime the graph is not in. That is
precisely what the "learning to infer" literature does — amortised inference (Gershman &
Goodman), recurrent inference machines (Putzky & Welling), neural-enhanced belief propagation
(Satorras & Welling), which learns corrections to BP messages on loopy graphs.

So the Inferencer is not speculative machinery; it is the standard remedy for the failure mode
this project has documented three times.

## 4. Why it is not a loop

The intuition in the proposal — *a control-like diagram without looping behaviour* — is
correct, and [[The Two-Part Diagram]] §3 says exactly why.

Both meta-graphs minimise **the same** functional as the base graph: its free energy. The
Inferencer minimises it over *beliefs*; the Optimizer minimises it over *parameters*. That is
$f = g$: coordinate descent on one objective, not a game.

$$\underbrace{\min_{q}\ F(q,\theta)}_{\text{Inferencer}}
\qquad\qquad
\underbrace{\min_{\theta}\ F(q,\theta)}_{\text{Optimizer}}$$

Which is **EM**, and variational inference's coordinate ascent, and the E-step/M-step split,
all of which are the same alternation. There is no equilibrium to seek and no best response to
compute — each half strictly decreases one number.

> [!note] The loop moves from the wiring to the levels
> Alternation is still a loop *in time*. What changes is that it is no longer a cycle *in the
> diagram*: the meta-graphs read and write the base graph's state, and within any one level
> nothing is circular.
>
> That is the same move as unrolling a recurrent network into a DAG — and it is the *other*
> option from the one [[Time as a Base]] describes. Feedback can be turned acyclic by
> **unrolling in time** (a chain of variables) or by **lifting a level** (a graph over the
> state). This proposal is the second, and the two are alternatives rather than competitors.

And on *"but implicit relations"*: the meta-graphs being factor graphs rather than feedforward
networks is not decoration. An Inferencer that is itself a relation can be run in more than one
direction — asked "what message would explain this marginal?" as readily as "what marginal
follows from these messages". A learned optimiser that is a relation can be asked what
parameters would have produced an observed update. Whether those inverse queries are *useful*
is open; that they are even askable is the difference between this and stacking two networks.

## 5. This is the separation principle, with its known failure

Control theory has done this split, has a theorem about when it is valid, and has a name for
what goes wrong.

**The separation principle** (LQG): the optimal controller decomposes into an optimal
*estimator* (a Kalman filter) and an optimal *state-feedback controller*, and the two may be
designed **independently**. Estimate, then act on the estimate, and the composite is optimal.

That is the Inferencer/Optimizer split exactly — inference and optimisation designed apart and
composed. And its scope is exactly the scope of this project's exactness results:

| | separation holds | this project |
|---|---|---|
| linear dynamics, quadratic cost, Gaussian noise | **exactly** | `GaussianFactor`, [[The Linear Gaussian Chain]] — exact |
| anything else | **no** | everything else — approximate |

> [!warning] Where it breaks, and what the breakage is called
> Separation fails when the optimiser's choices change what the inferencer gets to see. Then
> the optimal action must trade off *doing well now* against *learning something useful* — and
> that trade-off has a name, **dual control** (Feldbaum, 1960), and a familiar face in machine
> learning: **exploration**.
>
> So the honest statement of the proposal's validity is: **the Inferencer/Optimizer split is
> exact in the linear-Gaussian case and approximate otherwise, and the approximation error is
> exploration.** That is a strong result, not a weak one — it is the same boundary every other
> exactness claim in this vault sits on.

## 6. The categorical home

There is one, and it is a good fit. David Jaz Myers' categorical systems theory builds a
double category of open dynamical systems with two kinds of morphism:

- **covariant** ones — trajectories, steady states, periodic orbits;
- **contravariant** ones, which *"allow for plugging variables of some systems into parameters
  of other systems"*.

That second clause is this proposal, verbatim. The Inferencer plugs its variables into the base
graph's messages; the Optimizer plugs its variables into the base graph's parameters. Both are
contravariant morphisms of open systems, and the reason they do not compose like ordinary
factors is that they are **morphisms in the other direction of a double category**, not
morphisms in the same category.

That also explains, after the fact, why [[Acausal Composition is a Hypergraph Category]] §6's
decorated-cospan story does not cover this. Cospans compose subgraphs into bigger subgraphs —
peers, side by side. A meta-graph is not a peer of the base graph; it is *over* it. Different
direction, different structure.

## 7. What would have to be built

In rough order of how much each buys, and none of it is done:

1. **Parameter sharing between factor nodes.** `ps` is a flat `NamedTuple` keyed by factor
   name, so two nodes cannot name the same entry — which blocks weight tying
   ([[GANs as Two Factors]] §2) *and* blocks any meta-graph that wants to apply one learned
   rule at many sites. It is the smallest missing piece and the most blocking.
2. **A message store that is addressable as variables.** The Inferencer's variables *are*
   `MessageStore`'s entries. Today the store is a `Vector{Any}` indexed by edge, private to
   the passing loop.
3. **A graph-level `AmortisedInversion`.** Per-factor amortisation exists; nothing amortises
   the *schedule*.
4. **A free energy that both levels can read.** `scalar_free_energy` exists and is computed
   from the store, so this is closest to done — but it is Gaussian-exact only, and for
   Dirac-valued factors it degenerates to a plain energy sum
   ([[Energy-Based Factor Graphs]] §3.2). A meta-graph descending it would be descending
   different things in different regions of the graph, and nothing would say so.
5. **Something to stop the Optimizer collapsing.** [[Energy-Based Learning]] §3: the free
   energy is LeCun's *energy loss*, the one with no contrastive term. It is safe today only
   because every metric is a fixed hyperparameter. An Optimizer with real freedom over
   parameters is exactly the situation where that stops being true.

Point 5 is the one to worry about. The others are engineering; that one is a theory gap, and
it says an Optimizer built on today's free energy would minimise it happily and learn nothing.

Related: [[The Two-Part Diagram]], [[Everything is a Factor]],
[[Learning Components as Parametric Lenses]], [[Bayesian Lens]],
[[Acausal Composition is a Hypergraph Category]], [[Time as a Base]],
[[Energy-Based Learning]], [[Energy-Based Factor Graphs]], [[Loopy Message Passing]],
[[The Linear Gaussian Chain]], [[Schedules]]
