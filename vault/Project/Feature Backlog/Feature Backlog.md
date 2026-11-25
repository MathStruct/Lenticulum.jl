#overview

> The features Lenticulum.jl should get, in one list: what each is for, where it was asked for,
> what measurement motivates it, and how far it is. A feature is added here when a
> specification, a problem, a review or a measurement asks for it, and moved to "done" with the
> commit that implemented it. The rows are in one note per section, listed below; this note holds
> the conventions, the dependency graph and the proposed order. Priorities are proposals until Daniel sets them.

> Sources: original to this repository; the open lists of [[Specifications]] and [[Problem Log]], [[LP-023 Findings of the external reviews of 2026-11-04]].

Status: *planned* (specified or asked for, not started), *proposed* (an idea with a reason, no
specification), *partial*, *done*. Priority: **high** (it limits something measured now),
**medium**, **low**.

**Ids and dependencies.** Every open row has an id: P (the three paths), I (inference), O
(out-of-core), S (structure), L (learning), U (utility factors), G (decisions and games), F (families), T (tooling), Q (quality and engineering). The
column *needs* names the rows that must be done first, because the row builds on them; a row that
only goes better after another says so in its text, not there. Ids are stable: a row that is done
moves to the done note with its id, and a new row takes the next free number of its section. A row with
`—` in *needs* can start now.

## Sections

| id | section | what |
|---|---|---|
| P | [[Backlog P The Three Paths]] | what the refactor review of 2026-11-10 asked for on the three paths; it comes before every other section |
| I | [[Backlog I Inference and Graph Execution]] | compilation, execution plans, schedules, batching, and inference as dynamics on the free energy |
| O | [[Backlog O Out-of-Core Graphs]] | a graph kept in a store and loaded where a query reaches; planned as its own package |
| S | [[Backlog S Structure Search and Editing]] | edits, moves, gates, scores and a search driver |
| L | [[Backlog L Learning]] | M-steps, gradient semantics, `train!`, hyperparameters, and losses for energy factors |
| U | [[Backlog U Utility Factors and Data]] | data loading and the small exact factors written by hand today (asked for 2026-11-05) |
| G | [[Backlog G Decisions, Control and Games]] | control as inference, owned scalarisations in games, open games, beliefs over hypotheses (asked for 2026-11-16) |
| F | [[Backlog F Families]] | implicit diffusion: calibration and posterior samplers; likelihoods, polynomial and algebraic factors |
| T | [[Backlog T Tooling]] | benchmarks for optimiser design and regression tracking |
| Q | [[Backlog Q Quality and Engineering]] | the SciML/JuliaHub standard across the repository: types for dispatch, any AD backend, any array type, parallel execution, benchmarks and profiles as gates, and the factor graph compiled ahead of time with Sophia (asked for 2026-11-24) |

Done rows: [[Backlog Done 2026-11]] (newest first, one note per month).

**Keeping it current.** A new row goes into its section's note with the next free number of that
section. A row that is done is removed from its section's note and written, with its id, at the top
of the current month's done note under the day's heading. If the row is in the dependency graph
below, mark it done there or remove it.

## What depends on what

The open rows that need another open row (every row not drawn here needs nothing open):

```mermaid
graph LR
  F2[F2 calibration<br/>benchmark] --> F3[F3 calibration<br/>remedies]
  F2 --> F4[F4 posterior samplers]
  F2 --> L15[L15 time-independent<br/>energy]
  I15[I15 belief-space<br/>interface] --> I12[I12 Langevin on F]
  I15 --> I13[I13 mirror descent on F]
  T1[T1 optimiser<br/>benchmark] --> I13
  O2[O2 nodes as<br/>description + arrays] --> O3[O3 store interface]
  O2 --> O1[O1 lazy nodes]
  O3 --> O1
  O1 --> O5[O5 parameters on demand]
  I5[I5 integer-slot<br/>store, done] --> O5
  S2[S2 moves and<br/>warm starts] --> S7[S7 structure-search<br/>driver]
  S6[S6 scores<br/>beyond BIC] --> S7
  L7[L7 train!] --> L12[L12 learned optimisers]
  L9[L9 hypergradients] --> L12
  G1[G1 reward and<br/>policy factors] --> G2[G2 posterior that<br/>keeps the dynamics]
  L3[L3 stop gradient] --> G4[G4 Bellman residual<br/>as a loss]
  P16[P16 owned<br/>scalarisations] --> G5[G5 payoff gradient<br/>through beliefs]
  P16 --> G6[G6 joint update<br/>rules]
  P16 --> G7[G7 graphs with owners:<br/>what F's readers do]
  P16 --> G8[G8 entropy-regularised<br/>games]
  G1 --> G10[G10 Markov games]
  G9[G9 open games] --> G10
  T3[T3 JET on<br/>kernels] --> Q3[Q3 type stability<br/>as a gate]
  Q5[Q5 AD backend<br/>matrix, done] --> Q6[Q6 native rules for<br/>implicit adjoints]
  Q4[Q4 element type<br/>from the input, done] --> Q7[Q7 device-generic<br/>factors, done]
  Q4 --> Q19[Q19 element types on<br/>every message path]
  Q7 --> Q20[Q20 devices on<br/>every message path]
  Q19 --> Q20
  Q7 --> Q8[Q8 hardware<br/>GPU job]
  Q9[Q9 thread safety] --> Q10[Q10 parallel<br/>execution]
  Q1[Q1 the standard] --> Q11[Q11 init/solve!<br/>caches]
  Q11 --> Q12[Q12 allocation<br/>budgets]
  I2[I2 typed execution<br/>plans] --> Q16[Q16 graph compiled<br/>with Sophia]
  Q7 --> Q16
```

F2 is *partial*; its present problems (circle, lemniscate, rings) suffice for F3, F4 and L15.

**Proposed order (2026-11-11, until Daniel sets one):** of that list one thing is open, the energy
threshold of F3 (off-relation detection), which needs an energy network. The rest is done (I5, P1,
P17, P18, P19, I18, P11, I6, I19, I20, I21, and the round trip and ridge residual of F3:
[[Backlog Done 2026-11]]; the later measurements against RxInfer are in [[Benchmark against RxInfer]]).
No order is proposed for the other open rows.
