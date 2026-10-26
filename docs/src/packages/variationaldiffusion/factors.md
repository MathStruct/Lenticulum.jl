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

## Conformance and game-level limits

`test/conformance.jl` runs the shared default lens checker on all five accepted binary
assignments and the unary point polarity. Fixed zero-noise quadrature and a unit Gaussian
predictor give independent linear-field and energy fixtures, including the two-coordinate
posterior from an `N(0,I)` anchor and a site of precision `c I`.

`supported_polarities` lists all `3^n - 2^n` assignments through four channels (65 entries),
with one-target/all-other-observed entries first. Above that cap it throws; use
`supports_polarity` or the lazy `polarities` iterator. Structural sink/direction checks do
not enumerate. Joint inversion and likelihood messages concatenate the unobserved blocks
in channel order. Gaussian/mixture messages restrict the full-state construction to those
blocks; likelihood messages divide out their diagonal anchors. Improper mixture components
throw because their normalized weights are undefined. Graph `factor_message` stays per target.

The LP-017 testsets check complete lists, lazy traversal, structural answers above the cap,
joint point/Gaussian/mixture messages, correlated covariance and anchor removal. Its four
broken assertions now pass. Game checks retain an explicit opt-out: the graph's score
coordinate is a denoising surrogate, not posterior entropy, and its inferred polarity/trivial
prior cannot preserve an explicit game's split and precisions (LP-009). These checks do not
certify arbitrary priors, joint latent posteriors, stochastic REDDiff or learned ProxDM.
