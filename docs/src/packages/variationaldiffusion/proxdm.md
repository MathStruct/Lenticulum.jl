# [ProxDM](@id vd-proxdm)

```@meta
CurrentModule = VariationalDiffusion
```

## Proximal diffusion models

ProxDM (Fang et al. 2025) queries the prior through ``\operatorname{prox}_{-\lambda\log p_t}``
instead of its score. `proximal` is the interface, `ProxNetwork` a learned prox
(``v - \sqrt\lambda\,\varepsilon_\theta(v; t, \lambda)``), `MixtureProx` the exact one for a Gaussian
mixture. `proxdm_sample` is the paper's sampler (PDA and PDA-hybrid), `prox_infer` the implicit
learner's query by half-quadratic splitting, `proximal_matching_loss` the training loss.

## Differentiating through proximal inference

`prox_pullback(p, sol, z₀, ρ, t, z̄, ps, st; λ, linsolve)` is the adjoint of `prox_infer`: one
solve with ``(I - \operatorname{diag}(a)D_{FF})^\top``, where ``D`` is the prox's input Jacobian
and ``a_i = 1/(1+\lambda\rho_i^2)``, gives the cotangents of the clamps, targets, precisions and
the prox's parameters. It is also the `ChainRulesCore.rrule` of `prox_infer`, so Zygote
differentiates through it. `MixtureProx` has exact derivatives; a `ProxNetwork` takes them from
its `ad` backend. With `linsolve = KrylovSolve()` only prox VJPs are used.

## In a factor graph

`DiffusionFactor(blocks, pred; prox = ProxDM(prox; t, λ))` inverts by `prox_infer`. The
factor then has two parameter trees, `(predictor = …, prox = …)`: the inversion reads the
prox's, the free energy's score term the predictor's. `implicit_solution` and
`implicit_factor_pullback` work as for `ImplicitProx`, the latter through `prox_pullback`.

`proxdm_sample(…; scheme = :trapezoidal)` is a second-order sampler at the same cost as
PDA: one prox per step, the explicit score read off the previous prox.

Known gap: the factor's message is a point only (no Laplace Gaussian).

## API

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["proxdm.jl"]
```
