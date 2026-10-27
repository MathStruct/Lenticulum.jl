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
inference history and is not replayed. Static sequential, tree and forward/backward
schedules are supported, with no damping. A finite unroll computes the derivative of
that iteration, including local free energies and Bethe variable corrections. It is
not a fixed-point derivative or a guarantee of exact inference in a loopy graph.

Reference differences, DI `AutoForwardDiff()` and `AutoZygote()` are supported for
the tested scalar/array Dirac and Gaussian cases. Other schedule backends, flooding,
residual scheduling and nonpositive sweep counts raise `ArgumentError`. The existing
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
The ordinary `infer!` reverse rule replays its realised static sweep count, ignoring
cotangents of state/reports; it does not differentiate convergence decisions. Graph descriptions (including data
stored inside them) are fixed in the reverse rule; expose a differentiable source
through its parameter tree. No graph/data cotangent interface is claimed. Gaussian
schedule messages use functional matrix inverses, and `belief_cov` has the inverse
matrix pullback, avoiding inverse-of-Cholesky mutation. Ordinary inference arithmetic
is retained. Gaussian joint precision assembly now preserves promoted element types.

Replay copies an edge-sized vector for each message task. With K tasks and E edges,
storage/copying can grow as O(K E), in addition to backend tapes. Forward mode also
scales with parameter dimension. No implicit graph adjoint, GPU support, higher-order
score rule or stochastic solver-inversion derivative is claimed.
