# [Getting started](@id getting-started)

Lenticulum learns **relations instead of functions**: one model of a joint space, queried in
whichever direction you need. This page installs it, runs a first query, and points you to the
tutorial that fits your background.

## Install

Lenticulum is not registered yet. From a clone of the
[repository](https://github.com/MathStruct/Lenticulum.jl):

```julia
using Pkg
Pkg.develop(path = "/path/to/Lenticulum.jl")
Pkg.develop(path = "/path/to/Lenticulum.jl/lib/VariationalDiffusion.jl")   # and the others under lib/
```

## A first query

A diffusion model of points on the unit circle, in closed form so that nothing needs training,
asked for ``y`` given ``x = 0.6``. The relation has two answers, and `implicit_roots` returns
both:

```@example first
using VariationalDiffusion, LuxCore, Random
sched = VPSDE()
θ = range(0, 2π; length = 49)[1:48]
circle = NoisePredictor(GaussianMixtureEps(sched, vcat(cos.(θ)', sin.(θ)'); s = 0.05), sched)
ps, st = LuxCore.setup(Xoshiro(0), circle)
m = ImplicitDiffusion(circle, field_nodes(Xoshiro(1), 2; samples = 8))

answers, _ = implicit_roots(m, [0.6, 0.0], [Inf, 0.0], ps, st)   # x clamped (Inf), y free (0)
[round(a.z[2]; digits = 3) for a in answers]
```

Swap the precision vector to `[0.0, Inf]` and the same model answers "``x`` given ``y``".

## Where to go next

| you come from | start with |
|---|---|
| **machine learning** | [A relation without training](@ref tutorial-circle), then [Train a small diffusion model](@ref tutorial-train), [Robot arm: one model, every direction](@ref tutorial-arm), [the kernel baseline](@ref tutorial-kernel) and [Intersections](@ref tutorial-intersections) |
| **statistics or robotics** (Kalman filters, GTSAM) | [Localisation as a factor graph](@ref tutorial-localization): odometry and GPS, smoothing versus filtering, and the noise level estimated from the marginal likelihood |
| **physics** | [Symmetry is not a law](@ref tutorial-forces) and [Discovering a force law](@ref tutorial-particles) |
| **category theory** | the [theory vault](https://mathstruct.github.io/Lenticulum.jl/dev/vault/), entry point *Factors are Parameterized Statistical Games* |
| **looking for an API** | the package pages, starting with [LenticulumCore](@ref lenticulumcore), and the [Vocabulary](@ref vocabulary) |

Every tutorial is also available as a Jupyter notebook, and every work cited is on the
[References](references.md) page.
