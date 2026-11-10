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

## What it does not do

- It does not check that the store is at a fixed point. Away from one, the result is the
  derivative of the local terms at whatever beliefs the store holds.
- `parameter_step` flattens each parameter subtree to a vector several times per step and builds
  an index dictionary per call; for large networks that is measurable.
- The optimiser rules are `GradientDescent`, `Momentum` and `Nesterov` (in Mycelium); there is
  no adaptive rule and no driver loop.
