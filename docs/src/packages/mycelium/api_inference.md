# [Mycelium API: schedules and inference](@id mycelium-api-inference)

```@meta
CurrentModule = Mycelium
```

Schedules, the execution of a schedule (`step!`, `sweep!`, `propagate!`, `infer!`) and the hooks of
the message rules (expectation propagation, mixture reduction, mean field). The rule types
themselves are with the graph on the [Mycelium API](@ref mycelium-api) page.

## Schedules

```@autodocs
Modules = [Mycelium]
Pages = ["schedules.jl"]
```

## Running inference

```@autodocs
Modules = [Mycelium]
Pages = ["passing.jl"]
```
