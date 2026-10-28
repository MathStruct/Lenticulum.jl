# [Mycelium](@id mycelium)

```@meta
CurrentModule = Mycelium
```

Factor **graphs** and **message passing** — how factors are wired and in what order they
talk. This is the layer with no counterpart in Lux, because in Lux the wiring is a DAG and the
order is implied by it.

## Building a graph

```julia
b = GraphBuilder()
variable!(b, :x, 1)
variable!(b, :y, 1)
factor!(b, :f, some_factor)
connect!(b, :f, :in, :x; direction = Bidirectional())
connect!(b, :f, :out, :y; direction = Bidirectional())
g = validate(build(b))
```

`validate` checks the structure: every edge names a channel its factor actually declares,
nothing is isolated, and dangling outputs are reported. It is a separate, optional step.

`connect!(...; coupling, detach=false)` accepts all four built-in couplings.
`factor!(...; freeze=(:encoder,))` pins a subtree, several tuple paths pin several,
and `freeze=:all` pins the full tree. Frozen coordinates have zero graph gradient
and no optimiser state; input derivatives are preserved. `isfrozen=true` pins all
parameters for this run; `islearnable=false` excludes an intrinsically nontrainable
factor from `learnable_factors`. Frozen learnable factors remain in that candidate list.

Edges are **directed**, and the direction restricts which messages are legal:

| direction | the factor can |
|---|---|
| `Bidirectional()` | both send and receive |
| `Emitting()` | only send (a data clamp) |
| `Absorbing()` | only receive (a loss) |

## Two acyclicity notions

Do not confuse them:

| predicate | about | decides |
|---|---|---|
| `istree(g)` | the **undirected** graph | whether message passing is **exact** |
| `isdag(g)` | the **directed** multigraph | whether the graph is Lux-compatible |

They are independent.

## Running inference

```julia
marg, report, st, store = infer!(g, tree_schedule(g), ps, st)
```

`marg` is a `NamedTuple` of marginal beliefs keyed by variable name; `report` says whether
messages converged; `store` holds the messages, and is what the free-energy functions read.

Schedules: `tree_schedule` (exact on trees), `flooding_schedule` (general, iterative),
`forward_backward_schedule` (chains — the Kalman/RTS smoother), `SequentialSchedule`.

## Structural factors

Data, priors, losses and optimisers are all factors here rather than special machinery:

`DataFactor`, `PriorFactor`, `LossFactor`, `RelayFactor`, `OptimiserFactor`, and the update
rules `GradientDescent`, `Momentum`, `Nesterov`.

## Free energy

`scalar_free_energy`, `bethe_free_energy`, `factor_free_energies` and
`variable_corrections` compute the graph's objective — a sum of per-factor energies plus a
per-variable entropy correction. On a converged linear-Gaussian tree this equals the exact
negative log marginal likelihood; loopy graphs use the Bethe approximation; see the [worked example](@ref getting-started).

`uses_cavity(factor)` defaults to `false`; implement it as `true` whenever
`local_free_energy` reads incoming variable-to-factor messages. All graph free-energy entry
points reject `Emitting()` edges on those factors with `ArgumentError`, identifying the
factor, channel and variable and recommending `Bidirectional()` (LP-014). Missing messages
on `Absorbing()` or `Bidirectional()` edges keep their existing behaviour. `PriorFactor` and
`LossFactor` return `true`; the constant-zero local free energies of `DataFactor`,
`RelayFactor` and `OptimiserFactor` retain `false`.

`combine_with_overlap(c,d)` returns the pooled belief and log overlap for
categorical and Gaussian pairs. A Gaussian first argument must be proper;
an improper second argument is allowed up to a constant depending on it alone.
Disjoint categorical support returns `(c,-Inf)`. Mixture weights and the
AlgebraicFactors homotopy extension use this public rule.

## Parameter steps

After inference has populated incoming messages:

```julia
using DifferentiationInterface, ADTypes, ForwardDiff
ad = AutoForwardDiff()
grad = LenticulumCore.free_energy_gradient(store, g, ps, st; ad)
ps, st, optstate = parameter_step(store, g, ps, st; ad, rule=GradientDescent(0.01))
# On subsequent steps pass optimiser_state=optstate (needed for Momentum/Nesterov).
```

The objective is the sum of scalarised local free energies, with beliefs fixed.
Bethe variable corrections have zero parameter derivative under that convention.
Only learnable factors are updated. Inference schedules and `OptimiserFactor` remain
separate; no gradient is propagated through the mutating inference schedule.
LuxFactor requires `residual=true` to learn from endpoint scoring in this mode.
The ImplicitLayers AD tests fit a known ODE coefficient and check Nesterov look-ahead
updates with ForwardDiff and Zygote. The diagonal parameter driver also has
closed-form state-dependent SciML endpoint tests with ForwardDiff, Zygote,
Mooncake and runtime-activity Enzyme reverse. NeuralODE factors opt into
`state_policy=:fixed` to hold incoming Lux state constant per solve and obtain
returned state from one final endpoint evaluation. The factor validates its
solver/backend combination before differentiation; see [ImplicitLayers](@ref implicitlayers) for tested solvers and the guarded Kvaerno3/InterpolatingAdjoint
limit. This extends endpoint coverage without changing schedule derivatives.

## Graph schedule adjoints (LP-002 batch C)

For Exact/Pathwise edges, finite replay now supports sequential, tree,
forward/backward, flooding and residual schedules, including `damping`:

```julia
grad = LenticulumCore.free_energy_gradient(store, g, ps, st;
    ad=AutoForwardDiff(), schedule=flooding_schedule(g), nsweeps=20, damping=0.3)
```

Replay starts from empty messages. Flooding reads one snapshot per sweep.
Residual replay records the chosen task sequence in a reference pass, then
differentiates that fixed sequence. This is a derivative of the realised
piecewise schedule; argmax, ties and convergence decisions have no derivative.
The raw Zygote `infer!` rule also replays its actual count/choices and damping.
Score estimates retain the one-undamped-static-sweep restriction.

At a converged BP root, opt into a message-space implicit adjoint:

```julia
using Zygote
grad = LenticulumCore.free_energy_gradient(store, g, ps, st;
    ad=AutoZygote(), schedule=flooding_schedule(g), adjoint=:implicit,
    maxsweeps=1000, tol=1e-10, damping=0.0,
    linear_solver=:dense, linear_tol=1e-10, linear_maxiter=1000)
# using Krylov; linear_solver=:krylov selects matrix-free GMRES.
# parameter_step forwards the same keywords.
```

The reference schedule finds a root from empty messages; `nsweeps` controls only
the unrolled driver. The implicit map is synchronous BP with outgoing damping.
It solves `(I-DT)'λ = ∂L/∂m` and returns `∂L/∂ps + (Dps T)'λ`, including Bethe
corrections in `L`. This requires a differentiable local root and invertible
`I-DT`; a contraction suffices. Convergence is checked against every BP row and
every linear solve checks its original residual. These checks do not prove
uniqueness or conditioning.

Diagonal/detach edges hold their legal directed update rows constant: their message
variations are zero in the linear system. Frozen parameter coordinates are
pinned while message derivatives through the factor survive. With mixed cuts,
this agrees with the limit of cut replay when its derivatives converge.
Exact/Pathwise only, with CPU real numeric
Dirac or Gaussian messages (dense or diagonal precision) and recursively empty
message state. Reference differences, ForwardDiff and Zygote are tested.
Native Enzyme/Mooncake, implicit score sampling, device coordinates, evolving
message state and higher-order graph adjoints are unsupported.

Dense solves store a quadratic message Jacobian and have cubic factorisation
cost. GMRES avoids that matrix, retains at most 30 Krylov vectors and uses a
backend gradient per transposed product. With ForwardDiff that gradient still
scales with message dimension. Both methods also pay reference inference and
functional store-copy costs; neither differentiates the reference iterations.
Finite replay retains its sweep-dependent tape/copy cost.

`lib/Mycelium.jl/test/graph_adjoint.jl` supplies dense Gaussian tree oracles,
reconverged loopy Bethe differences and unrolled convergence measurements.
`graph_adjoint_devices.jl` tests the new replay with diagonal Gaussian JLArrays
in Float32/Float64 and scalar indexing disabled; implicit device calls reject
both linear solvers because the coordinate adapter is CPU-only. No hardware GPU
result is claimed. LP-002 and the implementation notes record measurements and
the native-backend and metric/divergence reproducers.

## Known gaps

- `combine` is partial: particles can be reweighted by a density or an
  `AbstractPotentialBelief`. `LogPotentialBelief` wraps a callable log-density; two
  potentials combine by adding log-densities. Two particle clouds still cannot be pooled
  directly. Gaussian, categorical, and mixture products with an arbitrary potential throw
  rather than resample; structured belief pairs retain their explicit rules.
- Potential distances compare callable identity (otherwise `Inf`); damping potentials is
  an inspectable no-op through `can_damp`.
- No parameter sharing between factor nodes, so weight tying is not expressible.
- Loopy message passing has no convergence guarantees, and `report.converged` means the
  *messages* stopped moving, not that the answer is right.

## API

```@index
Modules = [Mycelium]
```

### Graphs

```@autodocs
Modules = [Mycelium]
Pages = ["graph.jl"]
```

### Polarity resolution

```@autodocs
Modules = [Mycelium]
Pages = ["polarity_resolution.jl"]
```

### Messages and beliefs

```@autodocs
Modules = [Mycelium]
Pages = ["messages.jl", "gaussian.jl"]
```

### Schedules

```@autodocs
Modules = [Mycelium]
Pages = ["schedules.jl"]
```

### Running inference

```@autodocs
Modules = [Mycelium]
Pages = ["passing.jl"]
```

### Free energy

```@autodocs
Modules = [Mycelium]
Pages = ["free_energy.jl"]
```

### Structural factors and optimisers

```@autodocs
Modules = [Mycelium]
Pages = ["factors.jl"]
```

### Everything else

```@autodocs
Modules = [Mycelium]
Pages = ["Mycelium.jl"]
```

## Game quantities and conformance

Data and Relay have zero game free energy and entropy. Prior supports exact Dirac and
categorical games; arbitrary Gaussian `nlogp` expectations still throw. Loss is a terminal
game: pass every channel as a point or Dirac in the named observation `y`; its entropy is
zero with the loss's shape. Optimiser is an update rule and both game methods throw
`ArgumentError`. Structural factors use the shared checker in `test/conformance.jl`:
direct messages for emitting factors (both Relay directions), explicit pointwise/game
opt-outs for Optimiser and sink/message/pointwise opt-outs for Loss. Their supported game
methods are compared to `local_free_energy` using option 1.

## Finite schedule gradients and straight-through values

```julia
connect!(builder, :model, :out, :y; coupling=ExactCoupling(), detach=false)
factor!(builder, :encoder, encoder; freeze=(:pretrained,))
grad = free_energy_gradient(store, graph, ps, st;
    ad=AutoForwardDiff(), schedule=tree_schedule(graph), nsweeps=1)
```

The diagonal default preserves the existing derivative at fixed incoming cavity
messages. Gaussian factors still recompute their local posterior; this is not generally
a fixed-posterior EM derivative. Other choices replay from empty messages, with explicit
finite sweep count. A derivative path survives only if every crossed attachment permits
it: exact/pathwise keep their local Jacobian, diagonal cuts it, and detach overrides all.
Pathwise supports Dirac and analytic Gaussian moment propagation; other informative
families throw. Exact differentiates the iteration, including Bethe corrections, without
promising converged/exact inference in loopy graphs.

Static sequential/tree/forward-backward schedules and DI ForwardDiff/Zygote or reference
differences are supported. Reverse `infer!` replays its realised static sweep count with
zero damping. State/reports have no cotangents. Flooding/residual replay, other schedule
backends and GPU are unsupported. Replay storage/copying grows with tasks times edges.

`ScoreFunctionCoupling(; baseline=nothing)` supports sampled terminal Gaussian/categorical
messages on all-absorbing factors for one sweep. Supply an explicit `rng` and
`nsamples >= 2`; `GradientEstimate` contains `.value`, componentwise `.standard_error`,
and `.nsamples`. Its objective is expected terminal point cost plus the other analytic local/Bethe
terms; sampling a terminal Dirac does not estimate distribution entropy.
Categorical probabilities must be strictly positive. The baseline is a fixed finite scalar. Score `parameter_step` returns four entries,
including the estimate; deterministic steps retain `(ps_new, st_new, optimiser_state)`.
Coupled steps commit the first replicate's inferred state plus one local-energy advance;
other replicates do not accumulate state changes. Bidirectional score sampling throws because replacing BP densities by samples changes
the free-energy objective. Mixed terminal score and upstream analytic/cut rules compose.

`StraightThroughFactor(round, identity)` has `(:in,:out)` channels, forward round values,
identity input Jacobian and zero local cost. Its fixed callables take scalar/array inputs.
ForwardDiff and ChainRules reverse mode are supported; the graph derivative entry points
reject other backends, including differences of the forward function.

```@autodocs
Modules = [Mycelium]
Pages = ["gradient_couplings.jl"]
```


## Device graph gradients

JLArrays tests cover Float32/Float64 static replay with ForwardDiff/Zygote for
Dirac and diagonal Gaussian messages: exact/pathwise/diagonal couplings, detach,
subset/all freeze, raw static inference AD and `parameter_step`. Active-coordinate
helpers use device gathers and broadcast masks. GradientDescent/Momentum/Nesterov
cast configuration coefficients to device parameter precision; CPU rules stay as
before. Terminal diagonal-Gaussian score estimates and standard errors match CPU
with common host RNG draws bulk-copied to the device. Dense Gaussian devices,
categorical score devices, other schedule backends and hardware GPUs remain untested.


Implicit DEQ messages in explicit-backend replay use IFT solves, including reverse
input messages with empty state. Both DEQ and Euler message directions have
nonzero downstream-loss tests; raw Zygote replay also selects the implicit rule.
For ForwardDiff implicit graphs, use the explicit `free_energy_gradient` driver
rather than relying on raw `infer!` to select IFT. Reverse-input DEQ replay with
nonempty Lux state is rejected because primal threading and fixed-state IFT differ.

Raw ForwardDiff DEQ messages now reject dual-valued unannotated solves with an
ArgumentError pointing to the explicit graph gradient driver, preventing accidental
root-iteration differentiation. Ordinary primal messages are unchanged.


The existing GaussianFactor, GaussianPrior and GaussianComposite replay adapters
accept that backend context too, preserving their functional matrix-inverse
formulas and closed-form CPU marginal-likelihood gradient checks.
