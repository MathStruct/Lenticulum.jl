# [Factors](@id vd-factors)

```@meta
CurrentModule = VariationalDiffusion
```

`DiffusionFactor` carves one vector space into named channel blocks; the `Polarity` decides
which blocks are clamped and how hard, which is the selection-matrix formulation
``P = \rho_{in}P_{in} + \rho_{out}P_{out} + \rho_{latent}P_{latent}``.

Its inversion is chosen by `prox`:

| `prox =` | inversion | message |
|---|---|---|
| `REDDiff(λ, steps)` | RED-Diff's stochastic proximal solve | a `DiracBelief` |
| `ImplicitProx(nodes; …)` | the deterministic implicit solver, with convergence and stability reported (`implicit_solution`) and a backward pass (`implicit_factor_pullback`) | a `DiracBelief`, or with `message = :gaussian` the Laplace Gaussian with the target's anchor divided out |
| `ProxDM(prox; t, λ, …)` | proximal inference (`prox_infer`) with a learned or exact prox; the same report and backward pass (through `prox_pullback`) | a `DiracBelief` |

With `ProxDM` the factor has **two parameter trees**, `(predictor = …, prox = …)`: the
`NoisePredictor` for the free energy's score term, and the prox for the inversion
([ProxDM](@ref vd-proxdm)).

`ImplicitProx(nodes; batched = true, linsolve = KrylovSolve())` passes the cost options of
[Implicit inference](@ref vd-implicit) on to the factor.

## API

### RED-Diff

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["reddiff.jl"]
```

### The factor

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["src/factor.jl"]
```

### The implicit solver as the inversion

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["implicit_factor.jl"]
```
