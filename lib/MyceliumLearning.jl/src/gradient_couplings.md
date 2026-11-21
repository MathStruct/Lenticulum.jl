#implementation

> Explicit finite schedule derivatives, per-edge cuts, parameter subset freezing,
> terminal score estimators and a separate straight-through factor. The diagonal default
> holds each factor's belief fixed (since LP-020, 2026-11-01; it held the cavities fixed before).
> Supported domains and limits are explicit below.

> Sources: original implementation; AutoBayes Definition 29 and Remark 30; code:
> `gradient_couplings.jl`, `free_energy.jl`, `graph.jl`; tests:
> `lib/MyceliumLearning.jl/test/gradient_couplings.jl`, `test/gradient_couplings.jl`.
>
> Bibliography: [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Automatic Differentiation of Expected Values](https://mathstruct.org/CategoryTheory-ML-Wiki/Automatic-Differentiation-of-Expected-Values)

## API and domain (2026-10-27)

`connect!(...; coupling=DiagonalCoupling(), detach=false)` labels one factor-port
attachment, in both message directions. `factor!(...; freeze=())` accepts `:all`,
one subtree path `(:encoder,)`, or several paths `((:encoder,), (:decoder, :bias))`.
Tuple components may use integer indices. Missing paths throw when `ps` is supplied.

`free_energy_gradient(store,g,ps,st; ad, schedule, nsweeps=1)` retains the old
fixed-store driver when every edge is diagonal. Otherwise it differentiates a fresh,
empty-store replay of the specified schedule. The supplied store is not a record of
inference history and is not replayed. Sequential, tree, forward/backward, flooding and residual schedules are supported,
including `damping` (the batch C contract below). A finite unroll computes the derivative of
that iteration, including local free energies and Bethe variable corrections. It is
not a fixed-point derivative or a guarantee of exact inference in a loopy graph.

Reference differences, DI `AutoForwardDiff()` and `AutoZygote()` are supported for
the tested scalar/array Dirac and Gaussian cases. Other schedule backends and nonpositive sweep counts raise `ArgumentError`. The existing
diagonal driver retains its LP-002 factor backend support.

## Mixed Jacobians

Write each scheduled message as a function of earlier messages and parameters. A
Jacobian path is the product of its local Jacobians; different paths add. At each
crossed boundary:

1. `detach=true` makes the entire boundary derivative zero, regardless of coupling.
2. Diagonal holds that message at its reference value. Local explicit parameter
   derivatives survive; derivatives entering through that message are dropped.
3. Exact retains the full derivative of the computed belief representation.
4. Pathwise retains Gaussian/Dirac representation derivatives. Gaussian moments
   propagate analytically: this is the Rao–Blackwellised Gaussian reparametrisation
   rule, without differentiating a random-number generator. Other informative
   families throw; uninformative identity messages are allowed.
5. Score replaces a terminal incoming Gaussian/categorical distribution by a fixed
   sampled Dirac and adds its density score weighted by the terminal factor cost.

Thus a path crossing both exact and pathwise boundaries survives; one crossing a
diagonal or detached boundary is cut. This is a schedule-level realisation of the
blocks omitted in AutoBayes Definition 29, rather than a single graph-wide switch.
Each factor-port attachment is a boundary: a factor-variable-factor connection has
two attachments, both of which must permit the desired derivative.

## Score estimates

`ScoreFunctionCoupling(; baseline=nothing)` uses a fixed finite scalar baseline
(zero when omitted). Supply an explicit `rng` and `nsamples >= 2`. Samples are drawn
outside AD; each differentiated replicate contains the direct loss derivative and
`(reference_terminal_loss - baseline) * (logq - reference_logq)`. The score accounts
for parameter-dependent probabilities, including categorical messages. The baseline
is independent of both parameters and the current draw, so its expected score is zero.

This implementation supports one sweep and terminal factors whose ports are all
`Absorbing`. Categorical probabilities must be strictly positive; a changing-support boundary
has no supplied derivative. Duplicate ports for the same variable in a sampled terminal
factor require a joint sampling rule and throw. Bidirectional sampling would replace BP densities and entropies with
particles and change the objective; it is rejected. Unsupported samplers/densities
throw. The estimated objective is the expectation of terminal **point** costs under
those sampled messages, plus the other analytic local terms and Bethe variable
corrections. Replacing a terminal distribution by a Dirac does not estimate its entropy. Multiple
terminal score edges can coexist with exact/pathwise/diagonal boundaries upstream.

`GradientEstimate.value` and `.standard_error` have `ps`'s shape, with componentwise
sample standard errors and `.nsamples`. A score `parameter_step` returns a fourth
entry containing this estimate; deterministic steps retain their three entries.
No confidence guarantee or exactness is asserted for a sampled value. A coupled
parameter step commits inference plus one local-energy state advance from its first
replicate; other replicates start from the same initial state and do not accumulate
state updates. The diagonal step retains its previous fixed-store state advance.

## Freezing and straight-through values

`islearnable=false` identifies an intrinsically nontrainable factor; `isfrozen=true`
pins all parameters for this run without changing `learnable_factors`. Both preserve
input derivatives. A graph-local `freeze` selects a subset. Frozen coordinates have
zero graph parameter gradient and no entries in the optimiser vectors; fully pinned
factors have no optimiser key. Look-ahead and update operate on the remaining coordinates.
Input cotangents still pass through the factor, including when every parameter is pinned.

A factor with some coordinates frozen is rebuilt by `_replace_active` from its reference values and
the active coordinates. Under ForwardDiff those are duals and the frozen ones `Float64`, and the
rebuilt vector takes their promoted type: before I29 (2026-11-21) it was a `Vector{Real}`, and
`svdvals` in `GaussianFactor`'s rank check refused it, so ForwardDiff through a replay with
`freeze = (:b,)` threw while Zygote and reference differences worked. Tests:
`lib/MyceliumLearning.jl/test/gradient_couplings.jl` ("I29") and the frozen-key seed of
`lib/Lenticulum.jl/test/coupled_keys.jl`, where ForwardDiff and Zygote now agree to 1e-16.

`StraightThroughFactor(forward,surrogate; channels=(:in,:out))` is a parameterless
directed factor with zero local game cost. Its message is the forward value, with the
surrogate input Jacobian. The callables are fixed functions of one scalar/array input.
Core's `straight_through` has a ChainRules reverse rule using the surrogate pullback
and a ForwardDiff extension preserving surrogate dual partials. Other graph derivative
backends, including reference differences, throw: differences of `round` cannot
recover an identity surrogate. Arbitrary outer AD calls must choose a supported backend.

## Verification and difficulties

The tests independently derive the linear chain derivative and the Gaussian marginal
likelihood derivative. They check cuts, mixed boundaries, subset/all freezing,
learnability, optimiser state, categorical/Gaussian score means and variance reduction,
and round/identity surrogate derivatives against surrogate differences.

The legacy diagonal gradient holds **cavities**, not full factor posteriors, fixed.
Gaussian `local_free_energy` recomputes its posterior from these cavities and its own
parameters. It therefore differs from true EM even at a BP solution. The Gaussian test
derives this local derivative separately and checks that true fixed-posterior EM equals
the marginal-likelihood derivative (the envelope identity). This distinction is retained
for compatibility rather than calling the legacy result an EM gradient.

Reverse AD through the mutable inference store caused aliased cotangent accumulation;
replay uses an immutable store with copied message vectors. Graph structure and freeze indices are constants.
The ordinary `infer!` reverse rule replays its realised sweep count and task choices, ignoring
cotangents of state/reports; it does not differentiate convergence decisions. Graph descriptions (including data
stored inside them) are fixed in the reverse rule; expose a differentiable source
through its parameter tree. No graph/data cotangent interface is claimed. Gaussian
schedule messages use functional matrix inverses, and `belief_cov` has the inverse
matrix pullback, avoiding inverse-of-Cholesky mutation. Ordinary inference arithmetic
is retained. Gaussian joint precision assembly now preserves promoted element types.

Replay keeps one message per task (a version, I17, below) and copies nothing per task; under
Zygote its reverse pass is O(tasks × degree + edges + keys). Before I17 it copied an edge-sized
vector per task, O(K E) for K tasks and E edges. Forward mode also scales with parameter dimension. The opt-in implicit graph adjoint is documented in [[graph_adjoint]]. General
device support, higher-order score rules and stochastic solver-inversion derivatives
are not claimed.


### LP-002 batch A endpoint state policy (2026-10-27)

NeuralODE factors can opt into `state_policy=:fixed`: constant incoming Lux state
per solve and one final endpoint update. The existing diagonal graph parameter
gradient and `parameter_step` driver use the factor's policy and fixed state
subtree. `lib/ImplicitLayers.jl/test/lp002_batch_a.jl` checks these gradients
against a state-dependent affine closed form with ForwardDiff, Zygote, Mooncake
and runtime-activity Enzyme reverse. Default evaluation-state threading is
unchanged; state derivatives, schedule replay backends and graph adjoint scope
are unchanged. [[flow]] records the solver/backend restrictions.


## LP-002 batch B device replay (2026-10-27)

`lib/ImplicitLayers.jl/test/device_arrays.jl` extends finite static replay to
JLArrays Float32/Float64, with scalar indexing disabled, using ForwardDiff/Zygote.
Dirac and diagonal Gaussian messages cover exact/pathwise/diagonal couplings,
attachment detach, subset/all freezing, raw static `infer!` AD and parameter steps.
A quadratic terminal factor has a closed Gaussian expectation: its variance term
is included, as is the graph's Bethe variable-entropy correction. Comparing only
the terminal expectation would omit a precision derivative in this exposed graph.

Active-coordinate gathers upload host integer metadata once and use device array
indexing. Frozen replacement gathers from concatenated old/active coordinates;
masking and pinning broadcast. Integer metadata has no tangent. Scalar host
comprehensions of parameter values previously broke both replay and diagonal steps.

Terminal diagonal-Gaussian score tests use an explicit host RNG, bulk-copy noise
to the device and compare gradient estimates and componentwise standard errors
with CPU using common draws. This tests the finite estimator, not its statistical
convergence or a native GPU RNG. Device Gaussian density/entropy use elementwise
algebra. Dense Gaussian message factorisations, categorical score devices,
nonterminal score rules, other schedules/backends and hardware GPUs remain outside
this coverage. Enzyme/Mooncake replay continues to throw a backend error.

Raw inference also exposed a tuple/vector cotangent mismatch when final replay
message vectors were converted by splatting into `Any[...]`. A normal vector copy
preserves the vector tangent representation. Device Gaussian convergence distances
strip duals (reports are metadata) and use typed reduction initializers; Float64
initializers could not hold a Float32 dual maximum. Pooled diagonal-Gaussian
messages are checked against the analytic pooled mean/variance/entropy derivative.


### Implicit messages select their derivative backend

Replay now carries `ad` as a constant context into `_gradient_message`. The
ImplicitLayers DI extension routes DEQ messages through `solve_state/solve_input`
with that backend, so the nonzero downstream graph loss uses the IFT primitive
rather than root-iteration AD. Ordinary inference and diagonal (fixed-belief)
training keep their existing primal/default behavior. The raw ChainRules inference
replayer uses the same hook with a reverse-mode sentinel selecting AutoZygote.

Device tests check both DEQ and Euler-flow message directions, graph gradients and
parameter steps against dense closed forms; raw Zygote static inference is checked
too. Raw ForwardDiff through implicit `infer!` is not an automatic IFT interface:
use `free_energy_gradient(...; ad=AutoForwardDiff(), schedule=...)` for explicit
backend selection. Reverse-input DEQ replay with nonempty Lux state is rejected:
its ordinary inversion threads evaluation state while the explicit IFT equation
fixes it. Forward-state DEQ messages retain fixed-state solves. No general graph
fixed-point adjoint or new native replay backend is introduced.

Raw ForwardDiff DEQ messages now reject dual-valued unannotated solves with an
ArgumentError pointing to the explicit graph gradient driver, preventing accidental
root-iteration differentiation. Ordinary primal messages are unchanged.


The backend-context migration also includes the top-level GaussianFactor,
GaussianPrior and GaussianComposite specialisations. Their existing matrix-inverse
formulas remain the replay path; omitting the new argument bypassed those rules
and exposed inverse-of-Cholesky mutation in the first full CPU regression run.
The existing top-level closed marginal-likelihood oracle detects this failure.


## Batch C: realised schedules (2026-10-28)

`free_energy_gradient(...; schedule, nsweeps, damping=0.0)` starts from empty
messages and differentiates exactly that finite computation, with reference
differences, ForwardDiff or Zygote. The all-diagonal default still uses its
fixed-store partial derivative. `adjoint=:implicit` explicitly selects the
converged-message equation in [[graph_adjoint]], including for all-diagonal graphs.

Sequential/tree/forward-backward tasks read all preceding updates. Flooding
captures a store at each sweep boundary: every candidate reads that snapshot,
while functional Lux state threads in task order as in ordinary `sweep!`.
Residual scheduling runs the actual priority executor outside AD, recording
every committed task in order (including tie-breaking and early exit). Replay
then executes **that fixed sequence**, with the same initial state. Probe-only
state updates are discarded, as in ordinary residual scheduling. Factors must
not mutate incoming state or hide randomness/side effects: replay assumes a
deterministic functional factor evaluation. The raw ChainRules `infer!` rule
records residual choices in its own reference pass. Its existing rule detaches
state after every task and returns no state/report cotangents. The parameter
driver includes functional state dependencies in its finite computation, as
before; these two state contracts therefore differ for stateful factors.

This is the derivative of the **realised schedule**, a piecewise-defined
computation. Task choices, convergence counts, message families and polarity
branches are fixed during differentiation. At a residual tie or stopping
threshold it is a selected-branch derivative, not a derivative of argmax or a
guarantee that perturbed inference picks the same tasks. Reference differences
for residual tests freeze the recorded sequence too. Reconverged differences
are a separate oracle for the implicit path.

For each `:to_variable` update, compute the candidate, then call exactly
`damp(candidate, old, 1-damping)` if an old message exists and damping is positive.
Variable-to-factor pools are undamped; unsupported belief damping remains the
ordinary inspectable no-op. Gaussian canonical parameters and numeric Diracs
are tested. Device damping weights are cast to payload precision to avoid
Float32 widening; CPU primal arithmetic is unchanged. Score estimators retain
their one-undamped-static-sweep restriction.

`test/graph_adjoint.jl` checks loopy and tree Gaussian replay against ordinary
inference and fixed-sequence differences, for exact/pathwise couplings, damping
and raw Zygote inference. `test/graph_adjoint_devices.jl` checks Float32/Float64
diagonal Gaussian tree/loop replay with JLArrays scalar indexing disabled against
CPU derivatives of the same recorded operations. This is emulation, not hardware
GPU validation. The implicit coordinate adapter is CPU-only even with Krylov.

Enzyme/Mooncake replay remains explicitly rejected. The standalone reproducer
`lib/MyceliumLearning.jl/examples/lp002_replay_backends.jl` bypasses that guard to probe
the actual implementation with runtime-activity Enzyme modes and Mooncake.
Mooncake succeeds on the minimal Dirac probe but fails on Gaussian Bethe
entropy at `_improper_check`'s `try/catch`. Runtime-activity Enzyme forward
fails in `messages_into`; reverse fails in Base's `setindex_widen_up_to`.
The dated LP-002 section records versions and commands. The public backend
guard is retained; no general native backend support follows from a Dirac probe
or from factor-local endpoint rules.

## Fully cut factors at fixed beliefs (LP-021, 2026-11-02)

The objective of the coupled drivers (`_graph_scalar`, used by the unrolled replay and by the
implicit adjoint's loss) evaluated every factor's local free energy at its messages. A factor
whose edges are all diagonal or detached has constant messages, so its parameter derivative was
the fixed-message one, which LP-020 showed is not the free-energy gradient: with default couplings
`adjoint = :implicit` was 0.58 off on a three-variable tree, and one `ExactCoupling` edge anywhere
switched every fully cut factor from the default driver's fixed-belief gradient to this one.
`_frozen_beliefs` now evaluates `factor_belief` for those factors at the reference messages,
outside AD, and `_graph_scalar` differentiates `belief_free_energy` at that belief. Factors with a
derivative path, and factors without `factor_belief`, are unchanged. `test/lp021.jl` checks the
all-diagonal, one-exact-edge and all-exact cases against central differences.

## Mixed cuts: couplings only in a replay (LP-021 option 3, 2026-11-04)

At a fixed point the free-energy gradient is the envelope one, whatever the couplings; a coupling
can only change the derivative of a computation that is not at its fixed point, or of another
loss. So `free_energy_gradient` without `schedule` uses the envelope driver for graphs with
diagonal, exact or pathwise couplings (`_deterministic_couplings`; before, non-diagonal edges
required a replay), and `adjoint = :implicit` returns the envelope gradient at its converged fixed
point when the graph is **mixed** (`_mixed_cuts`: a cut and a differentiated edge, both carrying
messages that depend on the parameters; a data factor's edges and an observed variable's do not
count). On a graph that is not mixed the implicit adjoint is kept; it equals the envelope gradient
there (`test/lp021.jl`, `test/graph_adjoint.jl`). A replay with `schedule` still differentiates the
replayed sweeps, and warns on a mixed graph: holding a parameter-dependent message fixed next to
differentiated ones gave errors of 0.18–0.73 against the free-energy gradient, at convergence
(the LP-021 note has the table). Score-function couplings and `StraightThroughFactor` keep the
replay semantics.

## Reverse mode by message versions (I17, 2026-11-21)

**The cost.** Three paths differentiate through message passing: the unrolled replay (`_replay`,
`free_energy_gradient` with `schedule`), Zygote through `infer!` (`_functional_infer`, which the `rrule` of
`infer!` differentiates) and the implicit adjoint ([[graph_adjoint]]). Each was a fold over the tasks that
read `ps` and the message store as a whole per task and copied the store per task (`_with_message`). Under
Zygote a read gives a cotangent as wide as the container it reads (I16, `passing.md` "The backward pass by
key"), so the cotangents added up as O(tasks × keys + tasks × edges). `_functional_infer` also rebuilt the
pinned parameters of every key inside the fold, O(keys) reads by name per task.

**What is done.**
- *A task reads a local store.* `_task_reads(g, t)` names what a task reads: a to-factor task the to-variable
  messages of its variable; a to-variable task the to-factor messages of its factor and the to-variable
  messages of its variable (the target edge included, which damping reads). `_candidate` runs unchanged on
  a `_LocalStore` of those messages, whose `_EdgeSlice` throws on an edge outside it: the check that the
  reads are complete. It takes the factor's parameter and state subtrees, not the trees, and returns the
  new state as `Some(state)`, or `nothing` for a to-factor task, which writes none.
- *The replay is one function with one rule.* `_replay_forward` is a plain loop: task $i$ writes message
  version $i$, a table holds each slot's live version (0 for none), and a buffered (flooding) task reads the
  version a slot had at the start of its sweep. State has versions per parameter key (negative numbers for
  the initial state), so tied factors share theirs. `_replay_run` runs it; its `rrule`
  (`MyceliumLearningChainRulesCoreExt`) takes a pullback per task (`rrule_via_ad` of `_task_step`, whose
  modes hold what the fold bodies did: `stop_gradient` of detached messages and of the state in `infer!`,
  the tape's belief on cut edges and the score term in the replay), then walks the tasks backwards: a
  task's message gets the accumulated cotangent of its version, its pullback's input cotangents are added
  onto the versions and the subtree it read. O(tasks × degree + edges + keys). Without AD, and under
  ForwardDiff and reference differences, the loop is the primal. `_reference_tape` uses the same loop.
- *`_graph_scalar` reads once*, as the free energy does since I16 (`_subtrees`, `_states`, `_local_stores`);
  its two `sum`s are kept, so its value is unchanged bit for bit (`_sum_onto` would regroup: LP-073).
- *`_pinned_parameters` by position*, the key → node lookup taken once from `_key_groups`
  (`_parameter_nodes`); in `_functional_infer` it is computed once, outside the loop.

**Found by measuring after the first version.** The first version (Codex, from the design above) made the
unrolled gradient linear, but the Krylov product still grew 2.5–3.1× per doubling; the backward pass was
linear, the forward pass of Zygote's pullback was not. Three causes, each found by an allocation profile
(`Profile.Allocs`) at 61 and 241 factors:
- the I16 rule of `_subtrees` read `getfield(ps, i)` in code specialised on `ps`, a `NamedTuple` of one
  field per factor; with a run-time `i` that read cost O(keys), so the rule's forward pass was
  O(nodes × keys): 10.7 MB at 241 keys and 722 nodes, against 0.1 MB for the primal. A `@nospecialize`
  helper alone, or a read through `_IndexedTree`, was inlined back into the specialised caller; a
  `@noinline @nospecialize` reader (`_read_slots`) gives 0.13 MB. This rule serves I16's free energy too;
- `_same_message_layout`, a check, was traced by Zygote and generated a pullback per task at run time
  (45.6 of 139.8 MB at 241 factors); it and `_check_pathwise` are `@non_differentiable` now;
- the `map` over the tasks in `_graph_update` captured `ctx`, whose type holds the type of `ps`; each task's
  dynamically dispatched call cost O(keys) with it. The per-task body is now a function of small arguments
  (`_update_part`), and the closure captures `g`, the damping and the backend. The unrolled replay passes
  small arguments per task already.

**Measured.** `bench/coupled_scaling.jl` (Zygote on the chain of `state_scaling.jl`, every edge
`ExactCoupling()`, the tree schedule, one sweep replayed; best of 3; "first" is the call that compiles the
pullback for a new type of `ps`). Before: the code at `3ff93526`, in its own worktree; after: this change;
the same script on the same machine, one after the other (logs: `~/.cache/lenticulum/i17/`).

Unrolled replay, `free_energy_gradient(…; ad = AutoZygote(), schedule)` (`clean-{old,new}-unrolled.log`):

| factors | gradient before (ms) | gradient (ms) | allocated before (MB) | allocated (MB) | first call before (s) | first call (s) |
|---|---|---|---|---|---|---|
| 31 | 64.9 | 29.8 | 46.8 | 26.6 | 126.8 | 66.9 |
| 61 | 190.1 | 57.0 | 247.9 | 57.2 | 156.2 | 13.9 |
| 121 | 478.7 | 114.8 | 418.8 | 120.8 | 766.8 | 20.9 |

Implicit adjoint, one Krylov product (Zygote of `_graph_map_dot` in the messages and `ps`) and Zygote of the
loss (`clean-{old,new}-implicit.log`):

| factors | product before (ms) | product (ms) | allocated before (MB) | allocated (MB) | loss gradient before (ms) | loss gradient (ms) | first product before (s) | first product (s) |
|---|---|---|---|---|---|---|---|---|
| 31 | 53.4 | 19.7 | 42.9 | 22.1 | 37.0 | 15.3 | 79.8 | 31.4 |
| 61 | 178.7 | 38.9 | 168.1 | 44.5 | 100.8 | 28.7 | 106.3 | 8.0 |
| 121 | 747.2 | 76.4 | 699.3 | 89.6 | 361.6 | 56.8 | 588.1 | 16.7 |

Every column after the change grows 1.9–2.1× per doubling; before, the gradients grew 2.5–4.2× and the
first calls up to 5.5×. At 121 factors the unrolled gradient is 4.2 times faster and the Krylov product
9.8 times. The first row's first call includes loading Zygote. At 241 factors (`bench-final-*.log`, the
same script while a review ran beside it) the new code took 273 ms and 250 MB for the unrolled gradient and
169 ms and 181 MB for the product; the code before, measured with other runs beside it
(`unrolled.log`), took 3.3 s and 1.31 GB for the unrolled gradient. Reference differences, one replay per
coordinate, stay quadratic: 156 → 671 → 2 965 ms before, 98 → 432 → 1 632 ms after.

**Tests.**
- `lib/Lenticulum.jl/test/coupled_keys.jl` (Codex, the implementer's checks): the old functions copied as
  `_old_*`, compared with the new on random trees and loops, flooding, residual and tree schedules,
  `nsweeps` 1–3, damping, detached and diagonal edges, tied and frozen keys, shuffled, missing and extra
  keys; values `==`, Zygote against the old Zygote and against ForwardDiff; state versions and a seeded
  score replay; the coordinate pullback's layouts (dense and diagonal precisions, Dirac, Float32); and the
  allocations of one unrolled gradient and of one Krylov product at 10 and 40 steps, less than 5× (4.5× and
  4.1×; 8.9× for the unrolled gradient before, 5.9× for the product in the first version).
- `lib/Lenticulum.jl/test/coupled_spec.jl` (Mistral, written from the specification against the code before
  the change, without the implementation): Zygote = ForwardDiff = reference differences on the three paths,
  tied keys as the sum of untied ones, key order and extra keys, detached and mixed cuts, the implicit adjoint
  dense = Krylov = the converged replay, gradients of three fixed instances and a score estimate recorded as
  literals on the old code. Four of its mistakes were fixed by the session (marked in the file).

**What it does not do.**
- ForwardDiff and reference differences still cost a replay per parameter coordinate (quadratic by nature),
  and the dense implicit solve one reverse pass per coordinate; the change makes each pass linear.
- The compilation of the pullback still grows with the number of keys (I1); it is 20–45 s at 241 factors.
- Plates and block rows are not on these paths; device arrays were tested as before (the device tests of
  Mycelium and MyceliumLearning, run by Codex).
