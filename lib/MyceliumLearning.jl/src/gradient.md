#implementation #derivation

> The parameter gradient of the graph free energy at fixed factor beliefs, and the optimiser
> step that uses it.

> Sources: code: `gradient.jl`; the derivation and its history are in Mycelium's [[free_energy]] (§"Fixed beliefs, not fixed messages") and in [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]], [[LP-021 The coupled gradient drivers cut diagonal edges at fixed messages]]; tests: `lib/Lenticulum.jl/test/lp020.jl`, `lp021.jl`, `gaussian_oracle.jl`

## What it computes

At a fixed point of message passing the beliefs are stationary points of the Bethe free energy,
so its derivative in the parameters is

$$
\frac{dF}{d\theta} \;=\; \sum_f \partial_\theta F_f(b_f, \theta),
$$

each factor's term differentiated at its own belief held fixed (`Mycelium.factor_belief`,
`Mycelium.belief_free_energy`). Tied nodes add. Frozen coordinates get zeros. A factor without
`factor_belief` is differentiated at fixed incoming messages, which agrees when those messages
are points and is otherwise not the free-energy gradient (a warning says so).

When this is the gradient of $-\log p$: on a tree whose messages are exact
([[Bethe Free Energy]], "What exact requires"). `gaussian_oracle.jl` checks it against central
differences of the dense evidence on a random tree.

## One rule for which driver runs

`_uses_coupled_driver(g, adjoint, schedule)` decides, for `free_energy_gradient` and for
`parameter_step` alike: the coupled drivers (`gradient_couplings.jl`, `graph_adjoint.jl`) run
for `adjoint = :implicit`, or when an edge has a non-diagonal coupling **and** a `schedule` is
given (or a coupling is a score estimator or a straight-through factor is present). Without a
schedule, exact and pathwise couplings do not change the gradient at a fixed point, and the
envelope gradient is returned. Before 2026-11-10 `parameter_step` made this decision by an older
test and threw where `free_energy_gradient` answered
([[LP-024 Findings of the review of 2026-11-09]]).

The implicit driver accepts dense LU through `:dense` and matrix-free GMRES
through `:gmres`. The `:krylov` spelling is an alias of `:gmres` for existing callers.
Both public drivers forward that choice. See [[graph_adjoint]] and
`lib/MyceliumLearning.jl/test/graph_adjoint.jl`.

## The free energy must be finite (I27)

At fixed beliefs the envelope gradient reads only each factor's `belief_free_energy`: neither the
variable terms (where the overlap of exact messages is refused, LP-071) nor a factor's own check of
its energy at its messages (a `HardConstraintFactor`'s rank on its cavity's support). A model whose
free energy is infinite, two exact observations of one direction ($Y_2 = 2Y_1$), so got finite,
nonzero gradients from every backend while `scalar_free_energy` threw. `free_energy_gradient` now
evaluates `scalar_free_energy` once at the store after the envelope driver (`_check_finite_free_energy`)
and refuses a value that is not finite; the refusals of the free energy itself stand. It runs after the
driver so that the driver's own refusals come first: an AD backend a factor does not support is refused
before any evaluation of the factor (ImplicitLayers' `lp002_followups.jl` counts the right-hand-side
calls). The coupled drivers differentiate their own stores, so they check what they differentiate: the
replay the value of its loss (each draw for a score edge; `scalar_free_energy` at the replay's store
would refuse tied factors with state, which the replay supports), the implicit adjoint the free energy
at its converged fixed point (review LP-075: a constant
`Inf` cost had given finite derivatives through both). Nesterov's `parameter_step` checks the
look-ahead parameters, where its gradient is taken. Test: `lib/Lenticulum.jl/test/hard_overlap.jl`,
"I27, I28: review of I26". The check costs one forward pass of the free energy per call (not measured).

## What it does not do

- It does not check that the store is at a fixed point. Away from one, the result is the
  derivative of the local terms at whatever beliefs the store holds.
- `parameter_step` flattens each parameter subtree to a vector several times per step and builds
  an index dictionary per call; for large networks that is measurable.
- The optimiser rules are `GradientDescent`, `Momentum` and `Nesterov` (in Mycelium); there is
  no adaptive rule and no driver loop.
