#implementation

> Explicit finite schedule derivatives, per-edge cuts, parameter subset freezing,
> terminal score estimators and a separate straight-through factor. The default
> fixed-cavity gradient is retained. Supported domains and limits are explicit below.

> Sources: original implementation; AutoBayes Definition 29 and Remark 30; code:
> `gradient_couplings.jl`, `free_energy.jl`, `graph.jl`; tests:
> `lib/Mycelium.jl/test/gradient_couplings.jl`, `test/gradient_couplings.jl`.
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

Replay copies an edge-sized vector for each message task. With K tasks and E edges,
storage/copying can grow as O(K E), in addition to backend tapes. Forward mode also
scales with parameter dimension. The opt-in implicit graph adjoint is documented in [[graph_adjoint]]. General
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
rather than root-iteration AD. Ordinary inference and diagonal/fixed-cavity
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
`lib/Mycelium.jl/examples/lp002_replay_backends.jl` bypasses that guard to probe
the actual implementation with runtime-activity Enzyme modes and Mooncake.
Mooncake succeeds on the minimal Dirac probe but fails on Gaussian Bethe
entropy at `_improper_check`'s `try/catch`. Runtime-activity Enzyme forward
fails in `messages_into`; reverse fails in Base's `setindex_widen_up_to`.
The dated LP-002 section records versions and commands. The public backend
guard is retained; no general native backend support follows from a Dirac probe
or from factor-local endpoint rules.
