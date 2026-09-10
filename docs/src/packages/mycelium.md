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
nothing is isolated, and dangling outputs are reported.

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

## Known gaps

- `combine` works for Gaussians and Diracs and throws for anything else — pooling general
  beliefs needs densities.
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
Pages = ["messages.jl"]
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
