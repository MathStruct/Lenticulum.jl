# NeuralODE as an Invertible Factor

> The cleanest bidirectional factor in the project, and the reason is a theorem about ODEs
> rather than anything about neural networks: **a flow is a diffeomorphism, so integrating
> backwards inverts it.**
>
> Implemented as `NeuralODEFactor` in `lib/ImplicitLayers.jl/src/neuralode.jl`; see
> [[neuralode]] and [[flow]].

## 1. The relation

$$\frac{dz}{dt} = f_\theta(z,t),
\qquad
r(z_0, z_1) = z_1 - \Phi_{t_0\to t_1}(z_0)$$

`DiffEqFlux.NeuralODE(dynamics, tspan, Tsit5())` is a Lux layer: `(n)(x, ps, st)` builds an
`ODEProblem`, calls `solve`, and returns `(ODESolution, st)`. Direction fixed at construction,
output a solution object. Wrapping *that* gives [[luxfactor|`LuxFactor`]] and one polarity.

Keeping the dynamics instead gives two, and the second costs nothing:

```julia
flow_forward(f, z₀, ps, st)   # integrate t₀ → t₁
flow_reverse(f, z₁, ps, st)   # integrate t₁ → t₀   — the same call, endpoints swapped
```

## 2. Why this is different from every other bidirectional factor

Ranking the project's factors by what their *second* direction costs:

| factor | second direction | conditions |
|---|---|---|
| **`NeuralODEFactor`** | **identical to the first** | **none** |
| `LinearConstraintFactor` | identical — acausal by construction | none |
| `GaussianFactor` | different arithmetic; both closed-form | none |
| `DEQFactor` | a root-find that may not converge | needs $\dim x = \dim z$ |
| `LuxFactor` | does not exist | — |

`supported_polarities` returns two elements **unconditionally**: no dimension test, no solver,
no convergence question, no invertible-architecture constraint. Normalising flows spend their
entire architectural budget on invertibility — coupling layers, RealNVP, autoregressive masks,
all so the inverse is computable. A NeuralODE gets it for free by being a flow, and the
network inside can be anything.

> [!important] This is the strongest case in the project for factors over layers
> The forward and reverse passes of a NeuralODE are the *same code*. A framework that insists
> on a fixed direction is throwing away a symmetry the model already has, for nothing.

## 3. `ExactInversion`, and defending the label

`assemble` produces a lens with `LenticulumCore.ExactInversion`, not `SolverInversion`. The
claim is precise:

- $\Phi_{t_1\to t_0}$ **is** the inverse of $\Phi_{t_0\to t_1}$, exactly, as a fact about
  flows;
- what is inexact is the *discretisation*, which is a property of the integrator, not of the
  inversion. Refine the step and the error vanishes like $h^4$ with no bias left over.

Contrast `DEQFactor`, whose `SolverInversion` may converge to a **different root** or to none
at all. Those are qualitatively different failures, and [[Bayesian Lens]] cares about the
difference — *"nothing constrains an inversion to be exact… the quality of the choice is what
the free energy measures."* Collapsing both under `SolverInversion` would lose it.

## 4. Densities, not just points

A flow transports distributions, and the correction is the instantaneous change of variables
(Chen et al.; FFJORD):

$$\frac{d}{dt}\log p(z(t)) = -\operatorname{tr}\frac{\partial f_\theta}{\partial z}
\qquad\Longrightarrow\qquad
\log p(z_1) = \log p(z_0) - \int_{t_0}^{t_1}\!\operatorname{tr}\,\partial_z f_\theta\,dt$$

`flow_logdet` computes it, and the test suite checks it against a linear field where
$\operatorname{tr}\partial_z f = \operatorname{tr}A$ is constant.

> [!warning] And it is dead code
> The inversion still returns a `DiracBelief`, because the belief type that could hold a
> transported density is not reachable from a `lib/` package. `flow_logdet` is the exact
> quantity a density transport needs, it is correct, it is tested, and nothing can use it.
>
> That is the sharpest single illustration of why `GaussianBelief` is in the wrong package —
> see [[The Equilibrium Family]] §5.

## 5. What it cannot do

### 5.1 Change dimension

$z(t)$ lives in one space for all $t$, so `NeuralODEFactor` takes **one** `dim` and both
channels share it. A NeuralODE cannot be an encoder that compresses. (It is also why §4 works
at all — a change of variables needs a bijection between spaces of equal dimension.)

`DEQFactor`'s two channels are independent spaces; this one's are the same space at two times.

### 5.2 Survive stiffness, or strong dissipation in reverse

`lib/ImplicitLayers.jl` uses fixed-step explicit RK4, deliberately: an *adaptive* solver picks
different steps forwards and backwards, which destroys the round-trip property this whole note
is about. The cost is that explicit RK4 on a stiff system diverges quietly, and nothing
detects it.

Worse, and more specific: **what contracts forwards expands backwards.** For strongly
dissipative dynamics the reverse integration is unstable — the well-known reverse-mode
NeuralODE problem, and why real implementations checkpoint rather than re-integrate. The test
suite uses mildly negative eigenvalues, which is the easy case.

So §2's "the second direction costs nothing" is true of the *mathematics* and true of the
*implementation* on non-stiff problems, and false in general. The honest form of the claim:

> A flow is invertible exactly. Its numerical inverse is as good as your integrator, and the
> integrator is doing something harder in reverse than it was doing forwards.

### 5.3 Tell you it was inaccurate

`local_free_energy` is $\tfrac12\|z_1 - \Phi(z_0)\|^2$, which is **identically zero on the
relation** — if $z_1$ came from $\Phi(z_0)$ then the residual vanishes by construction. Good
invariant, useless diagnostic: it cannot distinguish a well-integrated flow from a badly
integrated one, because both satisfy their own residual.

The informative quantity is the discretisation error against a finer solve. Nothing computes
it, and `integrate` — unlike `solve_root` — returns no report at all.

## Sources

- Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS
  2018 — the flow layer, the adjoint, the instantaneous change of variables.
- Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud, *FFJORD*, ICLR 2019 — the Hutchinson
  estimator for the trace.
- [DiffEqFlux.jl NeuralDELayers](https://docs.sciml.ai/DiffEqFlux/stable/layers/NeuralDELayers/)
  — the layer being wrapped.

Related: [[The Equilibrium Family]], [[DEQ as a Relation]], [[Implicit Learners]],
[[Bayesian Lens]], [[Copiers Cups and Caps]], [[neuralode]], [[flow]]
