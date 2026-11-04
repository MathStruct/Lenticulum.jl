#open-problem

> **Status: mitigated (2026-11-04).** An `output` channel can now be declared, and with it the factor is a normalised conditional density. Nothing refuses or warns about the degenerate parameterisation. Found from Daniel's question how a switching factor is kept from "accepting more points than are on the relation"; measured in `test/lp022.jl`. Related: [[Switching Constraint Factors]], [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]], [[constraint]], [[switching]], [[regime]].

> Sources: original repository analysis; code: `src/constraint.jl` (`_coeffs`, `_output_coeff`), `src/switching.jl`, `src/regime.jl`; tests: `test/lp022.jl`
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

**Problem.** A constraint factor (`LinearConstraintFactor`, `SwitchingConstraintFactor`,
`RegimeFactor`) is a Gaussian in its residual $r = \sum_i A_i x_i - b_k$:

$$
-\log f = \tfrac12 r^\top Q^{-1} r + \tfrac12 \log\det 2\pi Q - \log w_k .
$$

That density is normalised over $r \in \mathbb R^m$, not over any of the channels. Data enter a Mycelium
graph as point messages, and the free energy is $-\log$ of the product of factors at the data (latent
variables integrated out). No partition function over the observed variables normalises it.
So a factor is a density of the data only if its parameterisation makes it one. With every
coefficient learnable nothing does, and the free energy rewards relations that accept every point:

1. **$A = 0$, $b = 0$.** Every point is on the relation, $r = 0$, and $-\log f = \tfrac12 \log\det 2\pi Q$, the
   smallest value it can take. In `test/lp022.jl` (200 points of $y = 2x + 0.5 + 0.3\varepsilon$, $Q = 0.09$) the
   least-squares relation has $F = 32.34$, and $A = 0$ has $F = -57.01$. At the least-squares fit the
   gradient in $A_y$ is $\sum_i r_i^2 / Q = 178.69$, not 0, so gradient descent shrinks the relation.
2. **$A \to cA$, $Q \to c^2 Q$** (learnable noise, `RegimeFactor(...; learn_noise = true)`):
   $f \to c^{-m} f$, so $F$ changes by $N m \log c$ and has no lower bound. Measured in the same test: halving
   the scale lowers $F$ by $200 \log 2$ to $10^{-10}$.

Neither is a bug in the messages or the free energy, which are exact for the factor as defined.
The factor is a relation, not a model of its channels. The fits in the repository were well-posed
only because the coefficient of the data channel was held fixed by hand
(`examples/structure_edits.jl`: "the sum node's coefficients are not updated";
`examples/regime_lines.jl` held the coefficient of $y$ at 1). The diffusion models do not have the
problem, since their objective bounds the log-density of a normalised model.

**When it is fine.** If the data channel's coefficient $A_y$ is fixed and invertible, the change of
variables $y \mapsto r$ is fixed, and $f$ is the conditional density of $y$ given the other channels up to the
constant $|\det A_y|$. Mixtures then cannot add mass: each component is normalised, and the
weights are a softmax. `GaussianFactor` is this case by construction, with $p(y \mid x)$ and $A_y = I$. A
rank-deficient relation (fewer equations than observed coordinates) is flat along $\ker A$. That is
fine when those coordinates are conditioned on, but then $F$ is not a density of all the data, and two
graphs can be compared by $F$ only when they normalise the same observed coordinates the same way.

## Mitigation (2026-11-04)

`output = :y` on the three constraint factors (with `dims[:y] == m`) fixes the coefficient of `y` at
`output_coeff` (the identity, or an invertible $m \times m$ matrix; for `RegimeFactor` the same in every
regime). The coefficient is dropped from the parameters: `ps.A` omits it, and passing it is an
`ArgumentError`. Every method reads the coefficients through `_coeffs(f, ps)`, which merges the fixed
one in, so messages, free energies, cluster forms and `partial_sums` see the full relation.
`test/lp022.jl` checks:
- the free energy equals $-\sum_i \log \mathcal N(y_i; a x_i + c, Q)$;
- gradient descent from 0 reaches least squares to $10^{-8}$;
- a fixed coefficient of 2 gives the density up to $|\det A_y|$;
- a switching factor with an output in a cluster gives $-\log p(y)$ by the closed form, and its message to the outcome matches outside a cluster;
- the refusals.

## Open

- **No guard.** A factor without `output` whose coefficients are all learnable and which touches
  data is still accepted silently. A check would need to know which variables carry data (an
  `Emitting` data edge) and which parameters are learned (none are frozen). It could warn in
  `free_energy_gradient`. Not done: Daniel has not decided between a warning and a refusal.
- **Several data channels.** One output channel per factor. A factor on two data channels whose
  joint density is meant needs both coefficients fixed, or a model written as a chain of
  conditionals.
- `partial_sums` keeps a sum factor's output and refuses to move the output channel into a partial
  sum. This is not tested.
