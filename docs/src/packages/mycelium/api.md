# [Mycelium API](@id mycelium-api)

```@meta
CurrentModule = Mycelium
```

Every exported type and function of Mycelium, by source file. The guide is on the
[Mycelium](@ref mycelium) page; plates and message blocks are on
[Mycelium API: plates and message blocks](@ref mycelium-api-plates).


```@index
Modules = [Mycelium]
```

## Graphs

```@autodocs
Modules = [Mycelium]
Pages = ["graph.jl"]
```

## Polarity resolution

```@autodocs
Modules = [Mycelium]
Pages = ["polarity_resolution.jl"]
```

## Messages and beliefs

```@autodocs
Modules = [Mycelium]
Pages = ["messages.jl", "gaussian.jl"]
```

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

## Free energy

```@autodocs
Modules = [Mycelium]
Pages = ["free_energy.jl"]
```

## Sampling

```@autodocs
Modules = [Mycelium]
Pages = ["sampling.jl"]
```

## Structural factors and optimisers

```@autodocs
Modules = [Mycelium]
Pages = ["factors.jl"]
```

## Exposed parameters

A parameter block of a factor can become a channel, so that a variable holds the weight, a
prior on it is an ordinary factor, and connecting two factors' exposed channels to one variable
ties the weight (L-SPEC-003 §1). With the channel observed as a point, any factor works; a
belief on it needs the factor's `parameter_message` and `parameter_free_energy`.

```@autodocs
Modules = [Mycelium]
Pages = ["exposure.jl"]
```

## [Gradient couplings](@id mycelium-api-couplings)

```@autodocs
Modules = [Mycelium]
Pages = ["gradient_couplings.jl"]
```

## Everything else

```@autodocs
Modules = [Mycelium]
Pages = ["Mycelium.jl"]
```
