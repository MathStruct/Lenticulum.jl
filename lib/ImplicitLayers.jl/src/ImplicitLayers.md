# ImplicitLayers.jl — package note

> SciML's implicit layers as Lenticulum factors: the **equilibrium** family of
> [[Implicit Learners]], alongside `VariationalDiffusion.jl`'s diffusion family.

## The thesis, in one table

`DeepEquilibriumNetworks.jl` and `DiffEqFlux.jl` both hand you an `AbstractLuxLayer` — a
function whose direction is fixed at construction. Both are built *out of* a relation. So
there are two ways to wrap them:

| wrapper | you supply | polarities | what you get |
|---|---|---|---|
| [[luxfactor\|`LuxFactor`]] | the assembled `DeepEquilibriumNetwork` / `NeuralODE` | **1** | graph membership; a Lux layer with extra steps |
| [[deq\|`DEQFactor`]] | the *cell* $g_\theta(z,x)$ | **2** (if square) | the relation, solvable either way |
| [[neuralode\|`NeuralODEFactor`]] | the *dynamics* $f_\theta(z,t)$ | **2**, always | the flow, invertible by construction |

The test suite makes the comparison on one object: the same cell wrapped as a `LuxFactor` has
one polarity and as a `DEQFactor` has two, with identical parameters.

## The files

| file | note |
|---|---|
| `solve.jl` | [[solve]] — Picard, Broyden, `SolveReport`, FD Jacobians, the IFT |
| `deq.jl` | [[deq]] — the fixed-point relation |
| `flow.jl` | [[flow]] — fixed-step RK forwards **and backwards**, CNF divergence |
| `neuralode.jl` | [[neuralode]] — the flow relation |
| `luxfactor.jl` | [[luxfactor]] — any Lux layer, one polarity |

Concept notes: [[The Equilibrium Family]], [[DEQ as a Relation]],
[[NeuralODE as an Invertible Factor]].

## Dependencies, and the AD line

`LuxCore`, `Random`, `LinearAlgebra` — no SciML, no `Lux`, no AD. The same choice
`VariationalDiffusion.jl` makes, but for a *weaker* reason, and the difference is worth
stating:

> RED-Diff's stop-gradient means the diffusion family needs **no derivative of the network at
> all**, so avoiding AD costs it nothing. The equilibrium family's backward pass **is** a
> linear system built from the Jacobian — the IFT is not an approximation that can be skipped.
> Avoiding AD here costs scalability: `fd_jacobian` is $O(n)$ forward passes and is honestly
> labelled a test-scale tool.

So this package can do inference at any scale (the solvers are derivative-free, as
`DeepEquilibriumNetworks.jl`'s are) and sensitivity analysis only at small scale.

## What it exposed

- **`GaussianBelief` is in the wrong package.** Every inversion here returns a `DiracBelief`,
  because a root-find and an ODE solve produce points and the belief type that could carry
  uncertainty lives in the top-level `Lenticulum`. `flow_logdet` computes exactly the
  correction a density transport needs and has nowhere to put it. Third factor package in a
  row; see [[The Equilibrium Family]] §5.
- **Dirac-valued inversions do not have Bethe free energies.** Both factors here contribute
  energy and no entropy, so the counting correction of [[Bethe Free Energy]] has nothing to
  correct and a mixed graph's total is not $-\log p(y)$. Same conclusion `VariationalDiffusion`
  reached.
- **The DEQ's singular Jacobian is the algebraic family's discriminant.** Two of the three
  [[Implicit Learners]] families fail in the same place for the same reason — see
  [[solve]] §3.

Related: [[Implicit Learners]], [[The Equilibrium Family]], [[Lux as a Parametric Lens]],
[[Statistical Game]]
