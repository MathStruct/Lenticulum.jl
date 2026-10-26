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

Per-edge gradient couplings are not implemented (LP-011, L-SPEC-002). `connect!` and `Edge`
accept only `DiagonalCoupling()`; other couplings throw `ArgumentError` at construction,
including direct and explicitly parameterized `Edge` construction. The default preserves
existing behavior and does not itself insert AD stops. Both `build` and `validate` throw
`ArgumentError` naming any factor declaring `isfrozen == true`, even when non-learnable:
`islearnable == false` only filters the optimiser list, not a frozen backward pass.
Message passing is unchanged.

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
per-variable entropy correction. For a linear-Gaussian graph this equals the exact negative
log marginal likelihood; see the [worked example](@ref getting-started).

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
