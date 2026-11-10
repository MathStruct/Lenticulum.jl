# [MyceliumLearning](@id myceliumlearning)

Learning on a `Mycelium` graph: what changes the parameters. Inference (graphs, messages,
schedules, the free energy) is in [Mycelium](@ref mycelium-api), which does not depend on this
package.

| what | functions |
|---|---|
| the gradient of the graph free energy at fixed factor beliefs (the envelope gradient) | `free_energy_gradient(store, g, ps, st)` |
| one optimiser step | [`parameter_step`](@ref) |
| differentiating through a finite schedule, score estimates | the edge couplings; [`GradientEstimate`](@ref), [`StraightThroughFactor`](@ref) |
| the implicit adjoint of a converged message fixed point | `free_energy_gradient(...; adjoint = :implicit)` |
| expectation maximisation with closed-form M-steps | [`em_step`](@ref), [`fit_em`](@ref), [`m_step`](@ref) |
| the gradient averaged over samples (Monte Carlo EM) | [`sample_gradient`](@ref) |

At a fixed point of message passing on a tree whose messages are exact, the envelope gradient is
the gradient of ``-\log p`` of the data; the condition is in the vault's *Bethe Free Energy*.

## The gradient and the optimiser step

```@autodocs
Modules = [MyceliumLearning]
Pages = ["gradient.jl"]
```

## [Gradient couplings](@id mycelium-api-couplings)

```@autodocs
Modules = [MyceliumLearning]
Pages = ["gradient_couplings.jl", "graph_adjoint.jl"]
```

## Expectation maximisation

```@autodocs
Modules = [MyceliumLearning]
Pages = ["em.jl"]
```

## Sample gradients

```@autodocs
Modules = [MyceliumLearning]
Pages = ["sample_gradient.jl", "MyceliumLearning.jl"]
```
