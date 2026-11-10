#implementation #overview

> **MyceliumLearning** holds what changes the parameters of a factor graph: gradients of the
> free energy, expectation maximisation and the optimiser step. It was split from Mycelium in
> November 2026 ([[Refactor 2026-11]]); Mycelium infers, this package learns.

> Sources: code: `MyceliumLearning.jl`; tests: `lib/MyceliumLearning.jl/test/`, and the learning tests of `lib/Lenticulum.jl/test/` (`lp020.jl`, `lp021.jl`, `em.jl`, `gaussian_oracle.jl`)

## What is here

| file | what | note |
|---|---|---|
| `gradient.jl` | `free_energy_gradient` of a graph, `parameter_step` | [[gradient]] |
| `gradient_couplings.jl` | per-edge couplings, replay of a finite schedule, score estimates, `StraightThroughFactor` | [[gradient_couplings]] |
| `graph_adjoint.jl` | the implicit adjoint of a converged message fixed point | [[graph_adjoint]] |
| `em.jl` | `em_step`, `fit_em`, the `m_step` hook | [[em]] |
| `sample_gradient.jl` | the gradient averaged over the samples of a `SampleBuffer` | [[sample_gradient]] |

## The boundary

Mycelium does not depend on this package. This package imports the names of Mycelium it uses,
internal ones included, in one explicit list at the top of `MyceliumLearning.jl`; the list was
generated from the sources and is extended by hand when a new name is needed.

What stayed in Mycelium because inference needs it or factor types extend it there:

- the hooks `factor_belief`, `belief_free_energy`, `factor_beliefs`, `belief_free_energies`
  (a factor's belief and its free energy at a given belief);
- `learns_unnormalised` (a trait of a factor type);
- the checks `_check_energy_edges` and `_check_tied_state`;
- the update rules `GradientDescent`, `Momentum`, `Nesterov` and `OptimiserFactor` (a factor
  that emits a parameter value);
- the flat-vector helpers for frozen coordinates (`_gather_active`, `_mask_gradient`).

A factor type opts in to learning by implementing the two belief hooks in Mycelium (for the
gradient) and `MyceliumLearning.m_step` (for EM).

## What it does not do

- The methods `free_energy_gradient(store, graph, …)` extend a function of LenticulumCore on
  types of Mycelium, from a third package. Inside one repository that is a convenience; it is
  type piracy by the letter, and it means the gradient exists only once this package is loaded.
- The tests are not fully sorted: the tests of the graph gradient on Gaussian factors are in
  `lib/Lenticulum.jl/test/`, because the factors are there.
- The device (JLArrays) replay tests run only with `LENTICULUM_DEVICE_TESTS=true`.
