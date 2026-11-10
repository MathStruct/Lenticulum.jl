#design

> **Status: accepted (2026-11-03, Daniel: "I accept option 1!"; proposed the same day by Claude; implemented the same day, both steps).** Related: [[L-SPEC-006 Plates]], [[Extending Mycelium]] §2, [[L-ADR-0009 Factor nodes carry a message rule]], [[L-SPEC-001 The Factor Interface]]. Code it would change: `lib/Mycelium.jl/src/messages.jl` (`MessageStore`), `plates.jl`, `passing.jl` (`factor_messages`), `free_energy.jl`; `lib/Lenticulum.jl/src/constraint.jl`, `lib/Lenticulum.jl/src/gaussian.jl`.

> Sources: original to this repository; measurements: `bench/plates.jl`, `lib/Mycelium.jl/src/plates.md` ("Speed").
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Context

Plates ([[L-SPEC-006 Plates]]) are unrolled: every replica's edge holds its own `Message` in the
store's `Vector{Any}`, and every factor and variable is a separate node. After the review fixes
(`plates.md`, "What changed after the review"), the factor-analysis E-step at $N = 1000$ takes
2.39 ms in a grouped plate against 0.0099 ms written by hand, a factor of about 240. The profile
has no single cause left. The time is spread over:

- one `Message` and one belief object per replica edge, in untyped storage, so every read is a
  dynamic dispatch;
- variable → factor messages computed one `excluded_marginal` at a time;
- each replica's input `NamedTuple`, built at run time;
- commits and residuals per replica.

The factor methods themselves are already batched (`factor_messages`: one factorisation and one
matrix product per group). What is not batched is the data structure between them. The design
note asked for a small factor of hand-written code, and learning through a graph is the point of
the library, so this decides whether plates get a batched representation, and of what shape.

## Options

1. **Message blocks in the store.** For each template edge of a plate the store holds one
   *block* per direction instead of $N$ messages: a `BatchedBelief` with the replicas as the last
   dimension, such as Gaussian natural parameters $\eta \in \mathbb R^{d \times N}$ with a precision that
   is shared ($d \times d$) or per replica ($d \times d \times N$), points $X \in \mathbb R^{d \times N}$, or
   categorical log-probabilities $k \times N$. Then:
   - a plate group is one call on blocks: `factor_messages` receives batched inputs and returns a
     block;
   - a variable → factor group is one elementwise `combine` of the variable's other blocks (sums
     of $\eta$ and of shared or stacked $\Lambda$);
   - `s.to_variable[e]` for a single replica edge returns a view (`BatchedBelief` column $i$), so
     everything outside plates (marginals, the free energy, the replay and implicit adjoints,
     sampling) keeps working unchanged, though at per-replica speed.

   Shared precisions are kept shared through `combine` when both operands share theirs, which is
   the common case: data points, a shared prior, and tied linear factors give every replica the
   same precision.
2. **A plate executor beside the store.** `PlateSchedule` keeps its own arrays for the plate's
   internal edges during a sweep and writes the per-replica messages back into the ordinary store
   at the end of each sweep (or on demand). Nothing in the store changes, but every sweep pays
   the write-back unless the reader is plate-aware, and two copies of the same messages must be
   kept consistent.
3. **Concretely typed storage everywhere.** Replace `Vector{Any}` by storage typed per edge class,
   and the abstract node and edge vectors by typed ones, so that per-message overhead falls for
   every graph. This does not batch anything. Per message it would remove the dynamic dispatch,
   but not the per-replica allocations or the per-replica loops.

## Recommendation

Option 1, in two steps:

1. Blocks and views in the store; batched `factor_messages` for `DataFactor`, `GaussianPrior` and
   `LinearConstraintFactor` (the factor-analysis plate); batched variable → factor messages;
   every other factor reached through per-replica views, as today. The test is the existing one:
   a plate equals the loop of its graphs and the sequential schedule. `bench/plates.jl` measures
   the gain.
2. Batched energies: a `belief_free_energies` hook so the free energy and its gradient over a
   plate are one call per tied key, not one per replica.

*Expectation, not measured:* with the factor-analysis plate in blocks, a sweep is six
block operations of the size of the hand-written E-step, which suggests a factor of 2 to 10
over the closed form rather than 240.

Option 3 is worth doing for its own sake (it helps every graph), but it does not reach the target
for plates, and it touches more of the code.

## Decision

Option 1, accepted by Daniel on 2026-11-03, with both steps in one piece of work:

- message blocks in the store for plate edges, with per-replica views kept ("Keep them for now"),
  so every path outside plates keeps working;
- batched energies (step 2) are included ("include it").

## Consequences

If option 1 is accepted:

- `MessageStore` gains block storage for plate edges, and `Message` reads through views. Code that
  writes `s.to_variable[e]` directly for a plate edge (tests, the sampler's pins) needs an accessor.
- Factors opt in to batching by implementing `factor_messages` on blocks. A factor that does not is
  run per replica through views, so no existing factor breaks.
- Mixtures (switching factors) stay per replica at first: a batched mixture belief (components ×
  replicas) is a further step.
- The AD paths keep differentiating per-replica views; batched replay is not part of this.

**Update (2026-11-03).** Implemented as recommended, in `lib/Mycelium.jl/src/blocks.jl` (note:
`blocks.md`), with batched methods for `DataFactor`, `GaussianPrior` and `LinearConstraintFactor`
and the energy hooks `local_free_energies`, `factor_beliefs` and `belief_free_energies`
([[L-SPEC-006 Plates]] §5–§6). Measured by `bench/plates.jl` at $N = 1000$: a grouped sweep takes 0.09 ms
against 0.0102 ms by hand (2.39 ms before), about 9 times, the upper end of the expectation above;
the free energy takes 1.1 ms and its gradient 2.1 ms, against 10.7 and 57 ms with the same messages
per edge. Writes to single edges go through `setindex!` on the new `EdgeMessages`, which turns a
row back into per-edge messages; no accessor was needed, so no code outside plates changed.
