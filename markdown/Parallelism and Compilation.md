# Parallelism, GPUs and Compilation

> How compatible is this library with parallel execution, GPUs, and ahead-of-time compilation
> (XLA/MLIR via Reactant.jl) — particularly in a **dynamic SLAM setting** where variables and
> factors are added and deleted as you go?
>
> Two assumptions in the question do not hold, one **measured complexity bug** turned up while
> checking, and the compile-versus-dynamic tension has a clean resolution that nobody has to
> choose between. Benchmarks in `bench/scaling.jl`.

## 1. Two things that are not there

**There is no junction tree.** `istree(g)` *checks* whether the factor graph happens to be a
tree; nothing *builds* one from a loopy graph. There is no triangulation, no clique
construction, no elimination ordering — `schedules.jl` offers `SequentialSchedule`,
`FloodingSchedule`, `TreeSchedule`, `ForwardBackwardSchedule` and `ResidualSchedule`, all of
which are orders over the *existing* edges.

**There is no parallelism.** `sweep!` is `for t in tasks(sched)`. `FloodingSchedule`'s own
docstring calls it *"the classical parallel BP update"* — and it is, semantically: all messages
at iteration ``k`` depend only on iteration ``k-1``, double-buffered. It is simply executed
sequentially. **The parallel structure is already declared and unexploited.**

## 2. What the measurements say

### The untyped store costs two orders of magnitude

`MessageStore` holds `Vector{Any}` for each direction, which `messages.md` §5 already concedes
means "message passing will not be fast". The size of that concession, on the same access
pattern with and without concrete types:

| store | 20 passes over 200k messages | allocated |
|---|---|---|
| `Vector{Any}` | 327 ms | 19.2 MB |
| concretely typed | 2.5 ms | **0 bytes** |

**130× slower, and allocation-free becomes 19 MB.** Every message access is a dynamic dispatch
and a heap-boxed load.

### And a sweep is quadratic in the number of factors

This was not previously recorded anywhere. Chain of ``n`` poses, `tree_schedule`:

| n | edges | messages | time (ms) | alloc/msg (B) |
|---|---|---|---|---|
| 10 | 19 | 38 | 0.11 | 2 058 |
| 50 | 99 | 198 | 0.52 | 5 046 |
| 100 | 199 | 398 | 1.22 | 8 958 |
| 200 | 399 | 798 | 4.72 | 16 021 |
| 400 | 799 | 1 598 | 8.68 | 30 420 |

Per-message cost *grows with graph size*, which it must not. The mechanism is confirmed:

> `ps` is a `NamedTuple` keyed by factor name, and the scheduler resolves a factor's parameters
> with a **runtime `Symbol`**. `getfield(nt, ::Symbol)` on a runtime symbol is **linear in the
> number of fields** — measured at 27 ns for 10 fields, 161 ns for 100, 1198 ns for 800, dead
> linear.

So each message pays ``O(\text{factors})`` to find its own parameters, and a sweep is
``O(n^2)``. In a SLAM setting — where the graph grows without bound — this is precisely the
wrong asymptotic, and it is invisible at test scale.

> [!important] This is the cheapest fix in the project
> Resolve factor names to integer indices once at `build`, and index a `Vector` instead of a
> `NamedTuple`. `FactorGraph` already carries `factor_index`; the parameters just are not
> travelling that way. It is a small change, it commits to no design, and it removes an
> asymptotic.

## 3. Where the parallelism actually is

Four independent levels, in increasing order of what they would cost to obtain:

| level | granularity | available today? |
|---|---|---|
| **within a flooding sweep** | every message at iteration ``k`` is independent | semantics yes, execution no |
| **across sibling subtrees** | in `tree_schedule`, independent branches | no — the schedule is a flat list |
| **across junction-tree cliques** | independent cliques of a triangulated graph | no junction tree exists |
| **across a batch of graphs** | same topology, different data | the GPU-friendly one; no batching |

The first is free: `FloodingSchedule` is already double-buffered, so `sweep!`'s loop over
independent tasks is a `Threads.@threads` away once §2's store is typed. Nothing conceptual is
in the way.

### But a SLAM trajectory is a chain, and a chain is the worst case

Worth stating clearly because it is not obvious. `tree_schedule`'s critical path is twice the
tree depth. For a chain of ``N`` poses the depth is ``N``, so **there is no parallelism along a
trajectory at all** — the exact structure a SLAM problem has.

That is not a dead end; it is a solved problem elsewhere. Särkkä and García-Fernández showed
that Bayesian filtering and smoothing recursions can be written in terms of **associative
operators**, so the whole sweep becomes an all-prefix-sums problem and a parallel scan gives
``O(\log N)`` span instead of ``O(N)``.

> [!important] The chain has a known logarithmic-depth parallel form
> [[The Linear Gaussian Chain]] is exactly the linear/Gaussian case those authors specialise
> to, and `forward_backward_schedule` is exactly the RTS smoother they parallelise. So the
> single most important graph shape in this project has an ``O(\log N)`` GPU formulation in the
> literature, and the current schedule is the ``O(N)`` one.
>
> This is the highest-value parallelism result available, and it needs no junction tree.

## 4. GPUs: the obstruction is upstream of any kernel

Three structural problems, none of which is about kernel design:

1. **`Vector{Any}`** — type instability forecloses fusion, vectorisation and tracing alike (§2).
2. **Heterogeneous beliefs per edge** — a `GaussianBelief` here, a `DiracBelief` there, so
   there is no array of like things to batch.
3. **Tiny dense matrices** — a 1×1 or 3×3 solve per message. On a GPU that is pure overhead
   unless thousands are batched into one call.

What GPU message passing needs is the standard answer from graph neural networks, and it is
worth saying that a **factor graph is a GNN with hand-written message functions**:

> Store messages **struct-of-arrays, grouped by (factor type, channel dimension)**, and
> represent topology as **index arrays** consumed by gather/scatter. Then one message update
> for all `GaussianFactor`s of dimension ``d`` is a single batched kernel, and the graph is
> *data* rather than *code*.

That single representational change unlocks levels 1 and 4 of §3, GPU execution, and §5's
compilation story simultaneously. It is the load-bearing refactor.

## 5. The compile-versus-dynamic tension, and why it is not real

The concern is right in general and dissolves under one design decision.

**Reactant/XLA wants** static shapes, static control flow, compile once and run many times.
**Dynamic SLAM wants** to add and delete variables and factors continuously. Compile the graph
and you recompile every step, which would indeed be ruinous.

So do not compile the graph:

> [!important] Compile the factor *types*, not the graph
> The compiled artifact is one batched kernel per **(factor type, dimension)** pair — of which
> there are a handful, fixed at model-design time. The graph enters as **index arrays** telling
> that kernel which edges to gather from and scatter to.
>
> Adding a pose and an odometry factor then **appends to an index array**. No new code, no new
> shapes, no recompilation. Deleting is a mask.

Two supporting techniques, both standard:

- **Capacity and masking.** Preallocate to a capacity, mask the unused tail, grow by doubling.
  Shapes change ``O(\log n)`` times over a run rather than every step, and each growth costs one
  recompile — which is the bucketing strategy used for dynamic batch sizes in ML serving.
- **Recompilation only on novelty.** A new *kind* of factor triggers a compile. In SLAM, after
  the first few frames there are no new kinds.

The residual honest cost: the first frame of each new factor type pays a compile, and the
masked capacity is wasted work proportional to slack.

## 6. Autodiff: the implicit structure is an asset here, not a liability

Today there is **no AD dependency anywhere** — deliberately. `VariationalDiffusion` avoids it
via RED-Diff's stop-gradient; `ImplicitLayers` needs it for the implicit function theorem and
substitutes finite differences, honestly labelled a test-scale tool.

The path to keeping AD under XLA/MLIR is Reactant tracing plus Enzyme, and it imposes the same
requirement §4 already imposes: no type instability, no `Vector{Any}`, no dynamic dispatch in
the traced region. One refactor serves both.

One rule matters more than the rest:

> **Never trace through a solver.** A Broyden loop or a Picard iteration has a data-dependent
> trip count, which is exactly what a static-shape compiler cannot express. Put the solve behind
> a **custom rule** derived from the implicit function theorem —
> ``\partial z^\ast/\partial x = (I - \partial_z g)^{-1}\partial_x g``, one linear solve — and
> the traced program contains a `solve`, not a loop.

Which flips the usual intuition. An unrolled deep network has a fixed but long trace; an
implicit layer has an *unbounded* loop that the IFT collapses to a single well-shaped linear
solve. **Implicit layers are more compile-friendly than explicit ones, provided the solver is
behind a rule** — see [[DEQ as a Relation]] §4 and [[solve]] §3, which derive the IFT already.

RED-Diff is the easy case: forward passes only, fixed step count, no branching. It is traceable
as written.

## 7. What a junction tree would buy — three things at once

The structure the question assumed exists is worth building, because it answers three separately
recorded problems with one construction:

1. **Exactness on loopy graphs.** [[Loopy Message Passing]] and the resistive-divider result —
   exact means, variances inflated by a stable factor — are properties of loopy *belief
   propagation*, not of the model. Junction-tree message passing is exact.
2. **A parallel schedule.** Independent cliques process concurrently; the critical path becomes
   the tree height rather than the edge count.
3. **Incremental update.** This is the SLAM requirement directly: iSAM-style **clique
   recycling** re-solves only the cliques affected by a new measurement rather than the whole
   graph.

`IncrementalInference.jl` does all three, on the Bayes (junction) tree, with clique recycling —
see [[Related Julia Projects]] §6. Anything built here should be measured against it.

## 8. An ordering

Roughly by value per unit of work, and each step is useful on its own:

1. **Integer-index the parameter lookup.** Removes an ``O(n^2)`` (§2). No design commitment.
2. **Type the message store**, grouped by factor type. Unlocks everything below, and is worth
   ~130× on message access by itself (§2).
3. **Thread the flooding sweep.** Nearly free once (2) is done; the semantics are already right.
4. **Associative-scan the chain.** ``O(\log N)`` on the shape that matters most (§3).
5. **Index-array topology and batched kernels.** GPU, and the precondition for (6).
6. **Reactant + Enzyme**, with solvers behind IFT rules (§6).
7. **Junction tree**, for exactness, clique parallelism and incremental updates (§7).

Nothing above (1)–(3) requires a GPU to be worth doing, and (1) and (2) are the two that make
the current numbers embarrassing rather than merely modest.

## 9. What was not measured, and why

No CUDA benchmark was run, despite a GPU being available. Measuring kernel throughput on
type-unstable scalar code would say nothing useful: the obstruction is at the representation
level (§4), several layers above where a kernel would sit. A GPU number would be a
well-measured answer to the wrong question.

## Sources

- Särkkä & García-Fernández, *Temporal Parallelization of Bayesian Smoothers*, IEEE TAC
  **66**(1):299–306, 2021 — [arXiv:1905.13002](https://arxiv.org/abs/1905.13002). Filtering and
  smoothing as associative operators; ``O(\log T)`` span by parallel scan.
- Kaess et al., *iSAM2: Incremental smoothing and mapping using the Bayes tree*, IJRR 2012 —
  clique recycling for incremental factor graphs.
- [IncrementalInference.jl](https://github.com/JuliaRobotics/IncrementalInference.jl) — Bayes
  tree, clique recycling, non-parametric beliefs, in Julia.
- [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) — tracing Julia to MLIR/XLA, with
  Enzyme for differentiation.

Related: [[Related Julia Projects]], [[The Linear Gaussian Chain]], [[Schedules]],
[[Loopy Message Passing]], [[messages]], [[DEQ as a Relation]], [[solve]],
[[Energy-Based Factor Graphs]], [[The Inferencer and the Optimizer]]
