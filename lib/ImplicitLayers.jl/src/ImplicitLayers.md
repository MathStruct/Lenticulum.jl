#implementation

> SciML's implicit layers as Lenticulum factors: the **equilibrium** family of
> [[Implicit Learners]], alongside `VariationalDiffusion.jl`'s diffusion family.

> Sources: code: `ImplicitLayers.jl`, `deq.jl`, `flow.jl`, `luxfactor.jl`, `neuralode.jl`, `solve.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## The thesis, in one table

`DeepEquilibriumNetworks.jl` and `DiffEqFlux.jl` both hand you an `AbstractLuxLayer` — a
function whose direction is fixed at construction. Both are built *out of* a relation. So
there are two ways to wrap them:

| wrapper | you supply | polarities | what you get |
|---|---|---|---|
| [[luxfactor\|`LuxFactor`]] | the assembled `DeepEquilibriumNetwork` / `NeuralODE` | **1** | graph membership; a Lux layer with extra steps |
| [[deq\|`DEQFactor`]] | the *cell* $g_\theta(z,x)$ | **2** (if square) | the relation, solvable either way |
| [[neuralode\|`NeuralODEFactor`]] | the *dynamics* $f_\theta(z,t)$ | **2**, always | both endpoint integrations; numerical inverse depends on method |

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

There is **no hard SciML dependency**. The package depends on `LenticulumCore`, `Mycelium`,
`LuxCore`, `Random`, `LinearAlgebra` and `DispatchDoctor`. Built-in root solvers and Euler/RK4
remain available without SciML. Optional `SciMLIntegrator` loads through package extensions;
`OrdinaryDiffEqTsit5` supplies Tsit5, `OrdinaryDiffEqSDIRK` supplies midpoint/trapezoid, and
both bring `OrdinaryDiffEqCore` and `SciMLBase` without the umbrella's unrelated solvers.
See [[flow]] for loading, reversibility conditions, CNF augmentation and tests.

There is no SciMLSensitivity gradient implementation here; that remains LP-002. The base
package still uses finite differences for derivative information:

> RED-Diff's stop-gradient means the diffusion family needs **no derivative of the network at
> all**, so avoiding AD costs it nothing. The equilibrium family's backward pass **is** a
> linear system built from the Jacobian — the IFT is not an approximation that can be skipped.
> Avoiding AD here costs scalability: `fd_jacobian` is $O(n)$ forward passes and is honestly
> labelled a test-scale tool.

The built-in derivative tools are suitable for small systems; the optional integrators add
solver choices without establishing sensitivity-analysis or device support for this wrapper.

## What it exposed

- **Inversions still return points.** `GaussianBelief` now lives in `LenticulumCore`, but
  these factors still return `DiracBelief`. A general non-Gaussian pushforward belief is
  not implemented; `flow_logdet` computes the continuous density correction without
  transporting a belief. See [[The Equilibrium Family]] §5.
- **Dirac-valued inversions do not have Bethe free energies.** Both factors here contribute
  energy and no entropy, so the counting correction of [[Bethe Free Energy]] has nothing to
  correct and a mixed graph's total is not $-\log p(y)$. Same conclusion `VariationalDiffusion`
  reached.
- **The DEQ's singular Jacobian is the algebraic family's discriminant.** Two of the three
  [[Implicit Learners]] families fail in the same place for the same reason — see
  [[solve]] §3.

Related: [[Implicit Learners]], [[The Equilibrium Family]], [[Lux as a Parametric Lens]],
[[Factors are Parameterized Statistical Games]]
