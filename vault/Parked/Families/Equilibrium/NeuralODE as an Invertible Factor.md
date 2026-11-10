#model

> A continuous ODE flow is invertible wherever its solution is unique in both directions.
> `NeuralODEFactor` exposes both endpoint integrations, but explicit numerical time reversal
> does not invert the discrete forward map. Exactness depends on the integrator and grid.
>
> Implemented in `lib/ImplicitLayers.jl/src/neuralode.jl`; see [[neuralode]] and [[flow]].

> Sources: Hairer, Lubich, Wanner, *Geometric Numerical Integration*, 2nd ed., II.3 (adjoint and symmetric methods); Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018; Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud, *FFJORD*, ICLR 2019; [DiffEqFlux.jl NeuralDELayers](https://docs.sciml.ai/DiffEqFlux/stable/layers/NeuralDELayers/)
>
> Bibliography: [[Bibliography#^hairer2006geometric|Hairer et al. 2006]] · [[Bibliography#^chen2018neuralode|Chen et al. 2018]] · [[Bibliography#^grathwohl2019ffjord|Grathwohl et al. 2019]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## 1. The relation

$$
\frac{dz}{dt} = f_\theta(z,t),
\qquad
r(z_0, z_1) = z_1 - \Phi_{t_0\to t_1}(z_0)
$$

`DiffEqFlux.NeuralODE(dynamics, tspan, Tsit5())` is a Lux layer: `(n)(x, ps, st)` builds an
`ODEProblem`, calls `solve`, and returns `(ODESolution, st)`. Direction fixed at construction,
output a solution object. Wrapping *that* gives [[luxfactor|`LuxFactor`]] and one polarity.

Keeping the dynamics instead gives two endpoint integrations:

```julia
flow_forward(f, z₀, ps, st)   # integrate t₀ → t₁
flow_reverse(f, z₁, ps, st)   # integrate t₁ → t₀   — the same call, endpoints swapped
```

## 2. Two supported endpoint directions

Both polarities integrate the same dynamics, with endpoints swapped. Built-in Euler/RK4 use
the same fixed number of stages either way; adaptive or implicit methods can require different
work in reverse. Supporting a polarity does not guarantee accuracy or convergence.

| factor | second direction | conditions |
|---|---|---|
| `NeuralODEFactor` | endpoint integration | numerical accuracy and inverse conditioning |
| `LinearConstraintFactor` | identical acausal solve | solvable selected block |
| `GaussianFactor` | closed-form likelihood | the declared Gaussian model |
| `DEQFactor` | root-find | square residual; convergence not guaranteed |
| `LuxFactor` | unavailable | one direction only |

## 3. Inversion of the discrete residual map

The factor scores $r=z_1-\Phi_h(z_0)$ using its numerical forward map. Reverse Euler/RK4
only approximate its inverse, even in exact arithmetic; `assemble` now labels them
`SolverInversion(integrator)`. Tsit5 is also nonexact, with tolerance-controlled integration
error. No shared inversion types changed.

The optional `SciMLIntegrator` can use `ImplicitMidpoint` or `Trapezoid`. Their fixed-step
maps are symmetric (Hairer–Lubich–Wanner II.3): $\Phi_{-h}=\Phi_h^{-1}$, up to nonlinear
solve tolerance and roundoff. The trait permits `ExactInversion` only for these methods with
`adaptive=false`, matching grids (`dt` divides the interval), default step limiters and a
conservative whitelist of solver options. `assemble` also requires empty Lux state.
This is discrete inverse accuracy, not exactness against the continuous flow. See [[flow]].

Euler's round-trip residual is $O(h)$, with leading term $a^2Th$ for $z'=az$.
RK4 has $O(h^4)$ forward error and a corresponding round-trip bound, but this scalar linear
round trip is actually $O(h^5)$: its step polynomial satisfies
$R(x)R(-x)=1+x^6/72+x^8/576$. The measured doubling ratios are therefore 2 and 32; 16 belongs
to RK4's forward error. [[neuralode]] §4.3 gives the derivation, tested convergence tolerances
and the table from `lib/ImplicitLayers.jl/examples/flow_roundtrip.jl`.

At $a=-5,T=1,N=50$, the script measures Euler's residual as 0.394994 (start point 20.9%
off), RK4's as $6.953127\times10^{-7}$; RK4 at $N=1$ gives 895.18. These are truncation
errors, not merely floating-point effects.

## 4. Densities, not just points

A flow transports distributions, and the correction is the instantaneous change of variables
(Chen et al.; FFJORD):

$$
\frac{d}{dt}\log p(z(t)) = -\operatorname{tr}\frac{\partial f_\theta}{\partial z}
\qquad\Longrightarrow\qquad
\log p(z_1) = \log p(z_0) - \int_{t_0}^{t_1}\!\operatorname{tr}\,\partial_z f_\theta\,dt
$$

`flow_logdet` computes it, and the test suite checks it against a linear field where
$\operatorname{tr}\partial_z f = \operatorname{tr}A$ is constant.

> [!warning] Density transport is not implemented
> `flow_logdet` computes the continuous divergence correction; it is not generally the exact
> determinant of the discrete integrator map. `GaussianBelief` is now in `LenticulumCore`,
> but a general non-Gaussian pushforward belief is still missing. Inversions return points.

## 5. What it cannot do

### 5.1 Change dimension

$z(t)$ lives in one space for all $t$, so `NeuralODEFactor` takes **one** `dim` and both
channels share it. A NeuralODE cannot be an encoder that compresses. (It is also why §4 works
at all — a change of variables needs a bijection between spaces of equal dimension.)

`DEQFactor`'s two channels are independent spaces; this one's are the same space at two times.

### 5.2 Remove inverse conditioning

Built-in explicit methods have no error estimator and can fail on coarse/stiff problems.
Optional OrdinaryDiffEq solvers add adaptive and implicit choices and check completion, but
no integrator fixes the conditioning of reversing a contracting flow. For $z'=az$, $a<0$,
the inverse derivative amplifies absolute endpoint perturbations by $e^{|a|T}$ (about 148.4
for $a=-5,T=1$). This is absolute amplification, not that scalar map's relative condition
number. Even a symmetric method may invert its own map accurately while giving the wrong
continuous initial point.

SciMLSensitivity gradients remain LP-002. Its
[`BacksolveAdjoint` docs](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#SciMLSensitivity.BacksolveAdjoint)
describe backward re-integration, the instability of reverse solutions and checkpointing.
The extension here does not implement adjoints.

### 5.3 Let you touch the trajectory in between

`NeuralODEFactor` exposes $(z_0, z_1)$ and nothing else. **You cannot attach a factor to
$z(0.5)$** — the integration happens inside and only the endpoints reach the graph.

Which is a little damning, because it is the same move [[DEQ as a Relation]] criticises
`DeepEquilibriumNetwork` for: sealing the solve inside and handing back the result. This factor
un-seals the *direction* and keeps the *time* sealed. Time is a latent of the factor, not a
shared base the graph can address.

[[The Structural Gap to ModelingToolkit]] §2 is about the general form of this, and
[[Time as a Base]] is what fixing it would look like — a variable that carries the whole
trajectory, queryable at any $t$ by interpolation.

### 5.4 Tell you it was inaccurate

`local_free_energy` is $\tfrac12\|z_1 - \Phi(z_0)\|^2$, which is **identically zero on the
relation** — if $z_1$ came from $\Phi(z_0)$ then the residual vanishes by construction. Good
invariant, useless diagnostic: it cannot distinguish a well-integrated flow from a badly
integrated one, because both satisfy their own residual.

Reverse-solved endpoints can expose round-trip residuals, which the lens game retains.
Accuracy against the continuous flow needs a closed form or refinement comparison; the
example script performs the former for linear dynamics. `integrate` returns only the final
state, not an error report. The SciML extension checks solve completion internally.

## Sources

- Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS
  2018 — the flow layer, the adjoint, the instantaneous change of variables.
- Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud, *FFJORD*, ICLR 2019 — the Hutchinson
  estimator for the trace.
- [DiffEqFlux.jl NeuralDELayers](https://docs.sciml.ai/DiffEqFlux/stable/layers/NeuralDELayers/)
  — the layer being wrapped.

Related: [[The Equilibrium Family]], [[DEQ as a Relation]], [[Implicit Learners]],
[[Time as a Base]], [[The Structural Gap to ModelingToolkit]],
[[Inversions and Bayesian Lenses]], [[Open Models and Latent Channels]], [[neuralode]], [[flow]]
