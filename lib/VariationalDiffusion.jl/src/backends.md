#implementation

> How a learned network's derivatives are computed: through **whatever AD backend the user
> picks**, on whatever device, while the package itself keeps no AD dependency. One field on
> `NoisePredictor`, three hooks, two package extensions. And the models this is meant for:
> **small** diffusion models on a low-dimensional joint space, not image U-Nets.

> Sources: code: `predictor.jl` (`ad`, `_apply`), `analytic.jl` (`epsilon_jacobian`, `epsilon_vjp_params`), `ext/VariationalDiffusionDifferentiationInterfaceExt.jl`, `ext/VariationalDiffusionReactantExt.jl`, `examples/circle_mlp.jl`; the ADTypes.jl, DifferentiationInterface.jl and Reactant.jl documentation; Moses & Churavy, *Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients*, NeurIPS 2020 (Enzyme)
>
> Theory: [[Backpropagation through Implicit Inference]] §8 says which derivatives are needed and why.

## 1. What needs a derivative, and what does not

| quantity | used by | how it is computed |
|---|---|---|
| $\varepsilon_\theta(x, t)$ | everything | `epsilon` → `_apply(ad, …)`: plain Lux, or compiled |
| $\partial_x\varepsilon_\theta$, an $n\times n$ Jacobian | `implicit_infer` (Newton), `implicit_pullback` (adjoint solve) | `epsilon_jacobian`: the backend, or finite differences without one |
| $(\partial_\theta\varepsilon_\theta)^\top w$, a parameter VJP | `implicit_pullback` only | `epsilon_vjp_params`: the backend; an error without one |

Sampling, RED-Diff and inference with a closed-form predictor need none of this. A learned
network needs the Jacobian for fast inference and the VJP to be trained through its own
inference. Nothing is ever differentiated through a *solver*: the solver loop stays in plain
Julia, and the implicit function theorem turns the backward pass into one linear solve plus one
VJP per field node ([[Parallelism and Compilation]] §6, "never trace through a solver").

## 2. The design: an `ad` field and three hooks

```julia
pred = NoisePredictor(model, VPSDE(); input, ad = AutoZygote())
```

`ad` is any ADTypes object, or `nothing` (the default). The core defines three internal
functions, and the extensions add methods for them:

| hook | default (no extension) | `DifferentiationInterface` loaded | `Reactant` loaded, `ad = AutoReactant()` |
|---|---|---|---|
| `_apply(ad, pred, x, t, ps, st)` | `LuxCore.apply` | unchanged | compiled forward, cached per shape |
| `_ad_jacobian(ad, …)` | not called: finite differences | `DI.jacobian` | compiled Enzyme input-VJP, one call per output row |
| `_ad_vjp_params(ad, …, w)` | `ArgumentError` naming the fix | `DI.pullback` | compiled Enzyme parameter-VJP |

So the package's own dependency list is unchanged (`LuxCore`, `Random`, `LinearAlgebra`, …).
ADTypes, DifferentiationInterface and Reactant are **weak dependencies**: a user who loads them
gets the extension, and a user who does not pays nothing.

## 3. Backends

| `ad =` | status | notes |
|---|---|---|
| `AutoZygote()` | tested | the usual Lux choice on CPU |
| `AutoForwardDiff()` | tested | forward mode; cheap for the $n\times n$ Jacobian, expensive for the parameter VJP of a large net |
| `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))` | tested | plain `AutoEnzyme()` fails on Lux layers on CPU with an `EnzymeRuntimeActivityError`; runtime activity is required |
| `AutoMooncake()` | expected to work | through DI like the others; not in the test suite |
| `AutoReactant()` | tested (CPU), separate suite | XLA-compiled forward and Enzyme VJPs; the same code runs on a GPU |

The backend package itself (Zygote, Enzyme, …) is loaded by the user, as with any DI call.

## 4. The DifferentiationInterface extension

Three details make it work across backends:

- **Flattened parameters.** A Lux parameter tree is a nested `NamedTuple`, which not every
  backend accepts as the differentiation variable. It is flattened to one vector and rebuilt
  inside the differentiated function, with the vector's element type, so ForwardDiff's duals
  pass through. The rebuild is non-mutating (Zygote) and the offsets are computed outside the
  differentiated function and passed in.
- **Constants, not closures.** Model, input, time and state go in as `DI.Constant`s, which is
  what Enzyme wants.
- **Flattened outputs.** A model may return an $n\times1$ column for a vector input (batch-style
  `input` functions do). The output and the cotangent are both `vec`'d, so the VJP never sees
  mismatched shapes. Without this Zygote fails inside LuxLib's matmul rule; with Reactant the
  same mismatch would *broadcast* silently to an $n\times n$ product, which is worse. Both are
  regression-tested.

## 5. The Reactant extension

DifferentiationInterface has no Reactant backend, so `AutoReactant()` has its own extension,
following the pattern of `EITDenoiser.jl`: `Reactant.@compile` once per argument shape, keep
the compiled thunk in a cache, call it with device arrays.

- **Forward, parameter VJP, input VJP** are three compiled programs. The VJPs are
  `Enzyme.gradient` inside the traced function, so XLA sees one fused forward+backward graph.
- **Time is a traced number** (`ConcreteRNumber`), so the dozens of noise levels in a
  `FieldNodes` set reuse one compiled program. Only a new *shape* recompiles.
- **Host arrays in, host arrays out.** The solvers are ordinary Julia loops that call the
  network a few times per iteration. Each call moves its arguments to the device and the result
  back. For the small models here this costs little and keeps every solver unchanged; for a
  large model, the right next step is to batch all field nodes into one call (one compiled
  program evaluating $\varepsilon_\theta$ at every node), which the field's structure allows.
- **The Jacobian** is $m$ calls of the compiled input-VJP, one per output coordinate. With
  $n = \dim Z$ small that is cheap; a compiled `Enzyme.jacobian` would make it one call.

The test (`test/reactant/runtests.jl`) has its own environment because Reactant downloads XLA.
It checks the compiled forward pass against plain Lux at several $t$, both derivatives against
finite differences, and a full `implicit_infer` + `implicit_pullback` through the compiled
network against the plain one.

## 6. Small models

A diffusion model here is a prior over the **joint space of one factor**: a handful of
coordinates (a pose, a few sensor readings, a low-dimensional latent), not an image. That
changes what a good architecture is:

- **Capacity is cheap, calls are not.** One inference iteration evaluates the network at every
  field node, plus Jacobians; one backward pass adds a VJP per node. A model that is evaluated
  thousands of times per query should be small.
- **Architectures that fit.** An MLP with a sinusoidal time embedding is the baseline and is
  what is tested. Residual MLPs, time conditioning by FiLM, small transformers over the channels
  of a factor, and parameter-free closed forms (`GaussianMixtureEps`) all fit the same
  interface. `input` adapts $(x, t)$ to whatever the model expects; the package never looks
  inside.
- **Train on the noise levels inference uses.** The implicit learner reads the field at the
  levels of its `FieldNodes` (small $t$ by default). A denoiser trained on $t \sim U(10^{-3}, 0.2)$
  serves it better than one spread over $[0,1]$.

`examples/circle_mlp.jl` is the end-to-end check. It trains a 3-layer MLP with **5,122
parameters** on points of the unit circle for 6,000 Adam steps (about 20 s on a laptop CPU),
then uses it as a relation:

| query | answer | radius |
|---|---|---|
| $x = 0.6$, start $y = +0.5$ | $y = +0.780$ | 0.984 |
| $x = 0.6$, start $y = -0.5$ | $y = -0.774$ | 0.979 |
| $x = 0$, start $y = +0.7$ | $y = +0.991$ | 0.991 |

Both branches are found and every solve reports `converged` and `stable`. The radius is about
2% inside 1, which is the smoothing bias of the field at the node levels
([[Implicit Diffusion Learners]] §5). The adjoint derivative of $(y-0.7)^2/2$ with respect to
the clamped $x$ and to a hidden-layer weight agrees with finite differences to 8 digits. The
learned field is not a gradient (Jacobian asymmetry about 5% for the 18k-parameter variant),
which is why the adjoint uses $J^\top$ and not $J$.

## 7. Implementation difficulties

- **Enzyme on Lux needs runtime activity on CPU**, even for a direct `NamedTuple` gradient;
  documented above rather than worked around.
- **Zygote rejects mutation**, so the parameter rebuild and the time embedding are written
  with broadcasting and slicing (a comprehension produced a `Matrix{Any}` that Zygote could not
  handle).
- **`ad = AutoReactant()` without Reactant loaded** silently uses plain Lux for the forward
  pass and DI (which then fails) for the derivatives. The result is correct but not compiled.
- **`ProxNetwork` has no `ad` field yet.** Nothing differentiates through `prox_infer`
  (see [[proxdm]] §5), so nothing needs one.

Related: [[predictor]], [[analytic]], [[implicit]], [[proxdm]],
[[Backpropagation through Implicit Inference]], [[Parallelism and Compilation]],
[[Lux as a Parametric Lens]]
