# The Equilibrium Family

> Entry point for the second of the three [[Implicit Learners]] families, implemented in
> `lib/ImplicitLayers.jl`.
>
> The claim: **SciML already builds these models out of relations, and then hands you a
> function.** A `DeepEquilibriumNetwork` is $z = g_\theta(z,x)$ with one direction chosen and
> the solve sealed inside; a `NeuralODE` is a flow with the time direction fixed. Un-sealing
> them is what this package does, and the number of supported polarities is the measure of
> how much came back.

## 1. What SciML gives you

Both packages produce an `AbstractLuxLayer`:

```julia
DeepEquilibriumNetwork(cell, solver)          # (x, ps, st) ↦ (z★, st) — solve info in st
NeuralODE(dynamics, tspan, Tsit5())           # (x, ps, st) ↦ (ODESolution, st)
```

That is excellent engineering and it is a *function*. [[Lux as a Parametric Lens]] explains
precisely what that costs: a Lux layer **is** a lens, so its direction is fixed at
construction, and a Lenticulum factor only *becomes* a lens once a polarity is chosen. A
wrapped SciML layer therefore has exactly one polarity.

The relation is still in there:

$$\text{DEQ:}\quad r(x,z) = z - g_\theta(z,x)
\qquad\qquad
\text{NeuralODE:}\quad r(z_0,z_1) = z_1 - \Phi_{t_0\to t_1}(z_0)$$

and $r = 0$ is [[README]]'s definition of an implicit learner, verbatim. Keeping $r$ rather
than the solved function is the whole intervention.

## 2. Three wrappers, and the number that distinguishes them

| wrapper | you supply | polarities | note |
|---|---|---|---|
| `LuxFactor` | the assembled layer | **1** | works on anything; see [[luxfactor]] |
| `DEQFactor` | the cell $g_\theta$ | **2** if $\dim x = \dim z$, else 1 | [[DEQ as a Relation]] |
| `NeuralODEFactor` | the dynamics $f_\theta$ | **2**, unconditionally | [[NeuralODE as an Invertible Factor]] |

The test suite makes the comparison on a single object: the same cell, wrapped as a
`LuxFactor`, has one polarity; as a `DEQFactor`, two — same network, same parameter count.
That pair of numbers is the shortest statement of what the package is for.

`LuxFactor` is not a straw man. It buys graph membership, parameter management, channel names
and free-energy accounting, and it works today on every model in the SciML ecosystem with no
dependency on any of them. It just does not buy bidirectionality, and bidirectionality is what
the rest of the project is about.

## 3. The two reverse directions are not equally cheap

This is the finding worth carrying away.

**A DEQ's reverse direction is a hard root-find.** Solving $z = g(z,x)$ for $x$ is a general
nonlinear problem with no contraction structure to lean on, it needs $\dim x = \dim z$ to be
square at all, and convergence is an open question on every call. `supports_polarity` refuses
it when the shapes disagree — so a DEQ whose input is an embedding of a different width is
*correctly* reported as unidirectional and gains nothing here.

**A NeuralODE's reverse direction is free.** A flow is a diffeomorphism, so integrating from
$t_1$ back to $t_0$ inverts it — same integrator, endpoints swapped, no solver, no dimension
condition, no convergence question. Ranking the project's bidirectional factors by the cost of
their second direction:

| factor | second direction |
|---|---|
| `NeuralODEFactor` | identical to the first |
| `LinearConstraintFactor` | identical (acausal by construction) |
| `GaussianFactor` | different arithmetic, both closed-form |
| `DEQFactor` | a root-find that may fail |

> A NeuralODE is the cheapest genuinely bidirectional factor in the project, and the reason is
> a theorem about ODEs rather than anything about neural networks.

## 4. Where the derivatives come back

[[The Diffusion Family]] got away without automatic differentiation: RED-Diff's stop-gradient
means the denoiser's Jacobian is never formed, so that package's entire inversion is forward
passes.

**The equilibrium family cannot do that.** Its backward pass *is* the implicit function
theorem,

$$\frac{\partial z^\ast}{\partial x} = (I - \partial_z g)^{-1}\,\partial_x g$$

which is a linear system built from the Jacobian. There is no version of this that skips the
derivative. `lib/ImplicitLayers.jl` computes it by dense finite differences and labels that a
test-scale tool — $O(n)$ forward passes where AD would be $O(1)$ — so the package does
inference at any scale and sensitivity analysis only at small scale. See [[solve]] §4.1.

That is a real difference between the families and the vault had not previously noticed it:
**one family's approximation buys it independence from AD; the other's exactness costs it.**

### A cross-family connection

`ift_sensitivity` fails when $I - \partial_z g$ is singular, and that is not a numerical
accident — it says the fixed point is not locally unique, i.e. **the relation branches there**.
The algebraic family calls that locus the discriminant
([[Branches and the Discriminant]]). Two of the three [[Implicit Learners]] families therefore
fail in the same place, for the same reason, in different vocabulary.

## 5. `GaussianBelief` is in the wrong package

Every inversion in `lib/ImplicitLayers.jl` returns a `DiracBelief`, because a root-find and an
ODE solve both produce a point. To return a *distribution* you would need somewhere to put
one — and `GaussianBelief` lives in the top-level `Lenticulum` package, which no `lib/`
package may depend on.

This is now the **third** factor package to hit it:

| package | what it can compute | what it must return |
|---|---|---|
| `VariationalDiffusion` | RED-Diff's variational $\sigma$ (derived in the paper, dropped) | `DiracBelief` |
| `ImplicitLayers` (DEQ) | $\partial z^\ast/\partial x$ — a covariance could be pushed through | `DiracBelief` |
| `ImplicitLayers` (NeuralODE) | `flow_logdet` — the exact change-of-variables correction | `DiracBelief` |

The last row is the sharpest: `flow_logdet` computes precisely the quantity a density
transport needs, it is tested and correct, and it is **dead code**, because there is no belief
type in scope to hold the result.

> [!important] The recommendation
> `GaussianBelief` and its canonical-form arithmetic belong in `LenticulumCore`, next to
> `DiracBelief`, `SampleBelief` and `TrivialBelief` — which are already there. Nothing about
> it depends on the top-level package; it ended up there because `Lenticulum` is where the
> first factor that needed it was written. Moving it would unblock Gaussian message passing
> in every `lib/` package at once.
>
> The same argument, from the other direction, is [[Acausal Composition is a Hypergraph Category]]
> §4: improper `GaussianBelief`s are what make the whole framework a hypergraph category. A
> load-bearing type should not be stranded at the top of the dependency graph.

## 6. Dirac-valued inversions have no Bethe free energy

Both factors here contribute an energy $\tfrac12\|r\|^2$ and **no entropy term**, because a
Dirac has none. So the counting correction of [[Bethe Free Energy]] — which adds back
$(1-d_v)$ copies of each variable's entropy — has nothing to correct, and a graph mixing these
factors with a `GaussianFactor` produces a total that is not $-\log p(y)$ for any model.

The identity [[The Linear Gaussian Chain]] §4 verifies to machine precision simply stops
holding. [[The Diffusion Factor]] §4.3 reached the same conclusion by a different route, and
together they suggest the general statement:

> **A factor whose inversion returns a point cannot participate in a Bethe free energy.** The
> formula is about entropies, and a point mass has none. Either the inversion returns a
> distribution (§5) or the factor is excluded from the accounting explicitly.

## Sources

- Bai, Kolter, Koltun, *Deep Equilibrium Models*, NeurIPS 2019 — the fixed-point layer and
  the $O(1)$-memory IFT backward pass.
- Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS
  2018 — the flow layer, the adjoint method, and the instantaneous change of variables.
- Grathwohl et al., *FFJORD*, ICLR 2019 — the Hutchinson trace estimator that makes the
  density correction affordable.
- [DeepEquilibriumNetworks.jl](https://docs.sciml.ai/DeepEquilibriumNetworks/stable/) and
  [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/layers/NeuralDELayers/) — the Julia
  implementations being wrapped.
- Broyden, *A class of methods for solving nonlinear simultaneous equations*, 1965.

Related: [[Implicit Learners]], [[DEQ as a Relation]],
[[NeuralODE as an Invertible Factor]], [[The Diffusion Family]],
[[Lux as a Parametric Lens]], [[Backpropagation by the Implicit Function Theorem]],
[[Bethe Free Energy]], [[ImplicitLayers]]
