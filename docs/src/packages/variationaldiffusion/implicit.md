# [Implicit inference](@id vd-implicit)

```@meta
CurrentModule = VariationalDiffusion
```

The implicit learner turns a noise predictor into a **relation**: a fixed set of field nodes
makes RED-Diff's regulariser gradient a deterministic field ``g_\theta``, and a query clamps
some coordinates (`ρ = Inf`), anchors others softly (finite `ρ`) and solves
``g_\theta(z) + \rho^2 \odot (z - z_0) = 0`` on the rest.

```julia
nodes = field_nodes(rng, n)                       # fixed (t, ε, w)
m     = ImplicitDiffusion(pred, nodes; batched = true)
sol, st = implicit_infer(m, z₀, ρ, ps, st)        # a root, with converged / stable flags
b = implicit_pullback(m, sol, z₀, ρ, z̄, ps, st)   # gradients by the adjoint
```

## All answers, and how sure

`implicit_roots` solves from several starts and keeps the distinct stable answers;
`implicit_laplace` and `laplace_belief` give the Laplace covariance at an answer;
`implicit_mixture` combines all answers into a `MixtureBelief` whose weights are Laplace mass
estimates when the model has an energy. Build the model with `λ = density_lambda(schedule,
nodes)` for covariances in data units.

## Many free coordinates

Every solver step, the Laplace covariance and the adjoint need only the Jacobian block on the
**free** coordinates, and that is all they compute. With `ImplicitDiffusion(pred, nodes;
batched = true)` all nodes of one noise level go through the network in one call. When most of
the state is free (generating an image), pass `linsolve = KrylovSolve()` to `implicit_infer`,
`implicit_pullback`, `laplace_belief`, `implicit_mixture` or `ImplicitProx`: Newton steps and
the adjoint are then solved by GMRES on Jacobian-vector products (`prior_jvp`) and
vector-Jacobian products (`prior_vjp`), stability is tested by Lanczos, and the Laplace
covariance is computed only on the coordinates asked for. On a 794-dimensional MLP with 784
free coordinates, `implicit_infer` takes about 1 s; one dense 784 × 784 block takes 33 s.

## Many queries, and classification

`implicit_infer_many(m, Z₀, ρ, ps, st)` solves the columns of `Z₀` as separate queries with
one polarity, and evaluates all of them in each network call: one call per noise level for
the field, one more for the free-coordinate blocks. The answers are those of `implicit_infer`
query by query. `implicit_classify(m, z₀, ρ, ps, st; label, classes)` frees the label
coordinates, starts from every class code and returns a `CategoricalBelief`. The mass of
roots far from every code is reported as `unassigned`, and the Gaussian mixture the belief was
read from is kept. Given a matrix of queries, it solves them all at once. On three classes with
a closed-form model the class probabilities are within 0.04 of the exact posterior.

With a predictor with noise blocks, `ImplicitDiffusion(pred, nodes; levels =
conditional_levels(pred, ρ))` holds the clamped blocks clean at every node, so the field on the
free coordinates is the smoothed conditional score. On the unit circle this moves the answers
from radius 0.978 to 0.993 (field levels up to 0.05). When a query has a single root,
`categorical_answer(…; temperature, cov_scale)` reads class probabilities from the root's position
and covariance, with both fitted on held-out answers by `calibrate_temperature` (on MNIST a
calibration error of 0.009). The solvers' default
tolerance follows the arithmetic precision (`default_tolerance`). For relations with an energy,
`implicit_infer` uses a trust-region method on the query's energy (`method = :auto`;
[nocedal2006numopt](@cite), [steihaug1983cg](@cite)): on the
circle every one of 160 starts converges (against 122 for Newton-or-descent), and on MNIST 19 of
20 modes with 784 free pixels, in about 6 s each, where Newton-or-descent found none.

## Off the data

Inference answers every query, also one the data never covered. `coverage_reference(m, data,
ps, st; calibration)` scores held-out points; `coverage(ref, sol)` then ranks an answer by its
distance to the data, the denoiser's reconstruction error and, with an energy, $-\log p_t$,
each as a conformal p-value ([bates2023conformal](@cite)). On the unit circle about 5% of
fresh data points get `p < 0.05`, and the answer to the clamp $x = 1.3$, which converges, gets
0.006.

## Samples, not only modes

The root finders return modes. `implicit_sample(m, z₀, ρ, ps, st; n, steps, resample)` draws
`n` samples of the free coordinates given the hard-clamped ones (`ρ = Inf`) with the reverse
diffusion, replacing the clamped coordinates at every step; `resample > 0` adds RePaint's
harmonisation, which a strongly coupled clamp needs. With noise blocks, `clamp_level = tmin`
holds the clamped blocks clean instead, and `guidance = w` adds classifier-free guidance
([ho2022cfg](@cite)). On MNIST, holding the label clean is what makes label → image samples
depend on the label (85.5% recognised, against 10.5% with replacement).

Two options make the sampler work on a product of relations, whose summed scores are not the
scores of the noised product ([du2023reduce](@cite)). `corrector = :langevin` or `:mala`
adds Markov chain steps at every noise level ([roberts1996mala](@cite)); they fix the shape of
each mode. `smc = true` also weights and resamples the chains, a sequential Monte Carlo sampler
over the noise levels ([delmoral2006smc](@cite), [skreta2025fkc](@cite)); it fixes how the mass
is split between modes, which local moves cannot. Both `:mala` and `smc` need log-densities
(mixtures, kernel estimates, energy networks).

## Differentiating through inference

With ChainRulesCore loaded (Zygote loads it), `implicit_infer` has an `rrule` whose pullback is
`implicit_pullback`: `Zygote.gradient(ps -> loss(first(implicit_infer(m, z₀, ρ, ps, st)).z), ps)`
works without unrolling the solver, and gives gradients for `z₀` and `ρ` too.

## Products of relations

`ProductRelation(m₁, m₂; weights)` pools relations on the same variables: its field is the
weighted sum, its stable roots are where all relations hold, e.g. the intersections of a
learned circle and a learned line. Every query function and the adjoint accept it.

Relations on different variables compose with pairs `relation => coordinates`, and a
**separator** is a relation with weight ``-1`` on the shared coordinates:
`ProductRelation(mA => [1, 2], mB => [2, 3], mS => [2]; weights = [1, 1, -1])` is
``p_A(x, y)\,p_B(y, z)/p_S(y)``. Without the separator the shared marginal counts twice: on a
chain whose two modes carry 0.8 and 0.2, the Laplace mixture weighs them 0.80 / 0.20 with it and
0.94 / 0.06 without.

## API

### The relation, the solver and the adjoint

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["implicit.jl"]
```

### Answers and uncertainty

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["answers.jl"]
```

### Matrix-free linear algebra

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["krylov.jl"]
```

### Many queries at once

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["many.jl"]
```

### Coverage

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["coverage.jl"]
```

### Sampling

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["sampler.jl"]
```

### Products

```@autodocs
Modules = [VariationalDiffusion]
Pages = ["product.jl"]
```
