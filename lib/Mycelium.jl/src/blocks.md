#implementation

> Message blocks: the store holds the messages of one template edge of a plate, for all replicas,
> as one value, and a plate sweep reads and writes whole blocks; any other reader indexes an edge
> as before and gets its column. With the factor-analysis plate on blocks a sweep is about 9 times
> the hand-written E-step instead of about 240 times, and the free energy and its gradient over a
> plate are one call per template factor (`bench/plates.jl`).

> Sources: original to this repository (decision: [[L-ADR-0010 Plates get batched message blocks]]). code: `blocks.jl`, `messages.jl` (`MessageStore`), `plates.jl` (`_step_blocks!`, `_step_cavity!`), `passing.jl` (`factor_messages` on blocks), `free_energy.jl` (`local_free_energies`, `factor_beliefs`, `belief_free_energies`), `factors.jl` (`DataFactor`), `../../../src/constraint.jl`, `../../../src/gaussian.jl`; tests: `../../../test/plates.jl`; benchmark: `../../../bench/plates.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

Implements: [[L-SPEC-006 Plates]] §5–§6.

## Blocks

A `BeliefBlock` is $n$ beliefs of one kind:

| block | holds | column $i$ |
|---|---|---|
| `TrivialBlock(n)` | nothing | `TrivialBelief()` |
| `PointBlock(X)` | $X \in \mathbb R^{d \times n}$ | `DiracBelief(X[:, i])` |
| `GaussianBlock(η, Λ, improper)` | $\eta \in \mathbb R^{d \times n}$; $\Lambda$ shared ($d \times d$) or a vector of $n$ | `GaussianBelief(η[:, i], Λᵢ, improper[i])` |
| `VectorBlock(beliefs)` | anything | `beliefs[i]` |

`belief_block(beliefs)` picks the most specific block for a vector of beliefs, keeping a precision
shared when every belief holds the same precision object. That sharing is the point. In a plate
of tied factors with data as points, every replica's message has the same precision (the prior's,
or $K A_t$ of a tied linear factor), so the block is one $d \times n$ matrix and one $d \times d$ one.

The operations a sweep needs work column for column, just as the per-belief rules do:

- `combine_blocks`: Gaussian blocks add canonical parameters (shared plus shared stays shared),
  a point dominates a Gaussian, the trivial block is the unit, and the improper flag follows
  `combine` on each column. Any other pair is combined column by column with `combine`, with its
  errors.
- `block_distance`: the largest `belief_distance` over the columns ($\max(\|\Delta\eta\|_\infty, \|\Delta\Lambda\|_\infty)$
  for Gaussians, the norm of the difference for points), without temporaries.

## The store

`MessageStore`'s two directions are `EdgeMessages`, an `AbstractVector{Any}` that indexes like the
`Vector{Any}` it replaced. `MessageStore(g)` gives each template edge of each plate a *block row*
(the edges `pl.edges[k, :]`, replica $i$ in column $i$). A row either holds one `Message` whose
belief is a block, or its edges hold their own messages:

- `s.to_variable[e]` for an edge of a row held as a block returns `Message(replica(block, i),
  iteration)`, a view built on demand. Marginals, the free energy, both gradient drivers, the
  sampler and the tests read edges this way and did not change.
- `s.to_variable[e] = m` on such an edge first writes every column back as an edge message
  (`_unblock!`), so a per-edge write never mixes with a stale block.
- `fill!` (used by `reset!`) clears both. `copy` returns a plain `Vector{Any}` of the views, and a
  store built from vectors (`MessageStore(tv, tf, iteration)`, as the coupled drivers do) has no
  rows, so every block path is skipped for it.

## Sweeps on blocks

`plate_schedule` records, per lifted group, the block row of each channel's edges
(`_GroupPlan.rows`), and for a group of variable → factor messages a `_CavityPlan`: the rows of
the other edges at each member's variable, in the same order for every member.

- **factor → variable** (`_step_blocks!`): the inputs are the rows' blocks (a row filled edge by
  edge is gathered into a block). The call is `factor_messages(factors, target, polarity, inputs::NamedTuple, priors, ps, st)`,
  which returns one block, and the block is written to the target row. The default method takes
  each column and calls the per-replica `factor_messages`, so a factor without block arithmetic
  still works. With damping or a detached target edge the result is committed edge by edge.
- **variable → factor** (`_step_cavity!`): the excluded marginal of every member is
  `combine_blocks` over the other rows, one block operation per row.

When a variable is pinned, an input row is partly filled, or a row is uninformative in some
columns only, the group takes the per-edge path of before. The tests check that a grouped sweep
still gives the sequential schedule's messages to $10^{-12}$, and that after a sweep of the
factor-analysis plate all six live rows are blocks: four Gaussian blocks with a shared
precision and two point blocks.

Batched methods: `LinearConstraintFactor` (inputs as points, trivial, or proper Gaussians with one
shared precision: one factorisation, then $H = Kc\,\mathbf 1^\top - \sum_n (K A_n) M_n$), `GaussianPrior`
(one conversion when the replicas share mean and covariance), and `DataFactor` (the data as one
point block).

## Energies on blocks

A plate factor row whose messages are blocks (belief-propagation nodes on one key, edges in
rows, none detached) is evaluated by three hooks. Each defaults to a loop over replicas:

- `local_free_energies(factors, msgs, ps, st)`: every replica's local free energy, used by
  `factor_free_energies` and `scalar_free_energy`;
- `factor_beliefs(factors, msgs, ps, st)`, outside AD, and `belief_free_energies(factors,
  beliefs, ps, st)`, differentiable in `ps`: the parameter gradient's frozen beliefs and its
  energies (LP-020). `free_energy_gradient` differentiates one item per row instead of $n$.

`LinearConstraintFactor` and `GaussianPrior` implement them through `FrozenGaussianBeliefs` (in
Lenticulum's `gaussian.jl`): with every free input a Gaussian block of shared precision, the
factor beliefs share one covariance and entropy, and only the means differ, one column each. The
variable terms of the Bethe free energy go by rows as well: the marginal block of a variable row
and its entropies, one entropy evaluation for a shared precision.

`test/plates.jl` compares a store with blocks against the same messages held edge by edge (a
store built from the copied vectors): the scalar free energy agrees to $10^{-13}$ (relative), each
factor's graded terms to $10^{-12}$, and the gradient of the tied entry to $10^{-10}$ with ForwardDiff
and Zygote ($10^{-6}$ with reference differences, whose step is $10^{-6}$).

## Speed

`bench/plates.jl`, factor analysis with $d = 3$, $k = 2$ (minimum of five runs; a fresh store and one
sweep):

| $N$ | plate, grouped, before | plate, grouped, on blocks | by hand |
|---|---|---|---|
| 100 | 0.24 ms | 0.02 ms | 0.0015 ms |
| 300 | 0.71 ms | 0.04 ms | 0.0037 ms |
| 1000 | 2.39 ms | 0.09 ms | 0.0102 ms |

"Before" is the table of `plates.md` (commit 600a1c28). On blocks the grouped plate is about 9 to 13
times the closed form. The L-ADR-0010 expectation was 2 to 10; at $N = 1000$ it lands at its upper end.

The scalar free energy and its gradient (reference differences, the default backend) after one
sweep, with the messages as blocks against the same messages per edge:

| $N$ | $F$, blocks | $F$, per edge | $\nabla F$, blocks | $\nabla F$, per edge |
|---|---|---|---|---|
| 100 | 0.12 ms | 1.09 ms | 0.30 ms | 5.23 ms |
| 1000 | 1.14 ms | 10.67 ms | 2.06 ms | 57.40 ms |
| 3000 | 3.49 ms | 32.88 ms | 5.51 ms | 166.47 ms |

## What it does not do

- Mixtures (switching factors) and other non-Gaussian messages are held as `VectorBlock`s and run
  per replica; a batched mixture block (components × replicas) is not implemented.
- Per-replica precisions (`Λ` as a vector) are combined and compared column by column; no factor
  method produces or consumes them in batch, so they take the per-replica fallback inside
  `LinearConstraintFactor`.
- Improper Gaussian inputs with flat directions other than all of $\mathbb R^d$ take the per-replica path.
- The AD paths through inference (`adjoint = :implicit`, non-diagonal couplings) replay on stores
  without rows, so they are not batched.
- The validity checks of the free energy (edge directions, tied state) still visit every node;
  at $N = 1000$ they are most of the remaining 1.1 ms.

## Difficulties

1. **A constructor that called itself.** `MessageStore` takes vectors or `EdgeMessages`, and an
   outer method on `AbstractVector` matched `EdgeMessages` too, so it recursed. The struct has an
   inner constructor on `EdgeMessages`, which is more specific.
2. **What cost the time after the first version.** The first block sweep took 0.24 ms at
   $N = 1000$; per group, the costs were small per-replica loops over abstractly typed fields: a
   pin check per member (an empty-pin shortcut), a detach check per member (moved into the plan),
   `repeat` of a short vector (a broadcast into `similar`), and the residual's temporaries. The
   free energy at first sped up 4 times only: a row check called `findfirst` per member
   (quadratic), the plate records are a `Vector{Any}` so the row loops were dynamically typed
   (function barriers), and `scalarisation(f.factor)` was dispatched per node (a typed row sum).
3. **Zygote differentiates through a store built from vectors.** The coupled drivers build
   `MessageStore(tv, tf, iteration)` inside AD (`_functional_infer`), which used to hold the
   vectors directly and now wraps them in `EdgeMessages`. Zygote had no adjoint for the wrapper
   (the LP-011 tests in `gradient_couplings.jl` caught it), and its row detection called
   `sum(...; init)`, which has no rule. The ChainRulesCore extension gives `EdgeMessages(v)` an
   `rrule` that passes the wrapped vector's tangent through, and marks `_energy_rows` and
   `_has_rows` non-differentiable (they are structural, like the other graph queries there).
