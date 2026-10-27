#implementation

> Bethe free energy from incoming cavity messages, with an edge-direction guard for factors
> that read them and a per-variable entropy counting correction.

> Sources: code: `free_energy.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

Implements: the graph generalisation of [Theorem 23](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#composition-energies-add-entropies-chain-definition-22) — the
Bethe free energy, its counting numbers, and the graded per-factor breakdown.
Theory: [[Bethe Free Energy]].

## The formula

$$U = \sum_f U_f, \qquad H = \sum_f H_f - \sum_v (d_v - 1) H_v$$

**Energies add; entropies need a counting correction.** That is exactly the asymmetry
[Proposition 18](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) identifies, appearing again one level up — which is
the reason this file exists rather than a one-line `sum(local_free_energy, factors)`. The
one-liner is the energy part, and it is right; the entropy part is what a naive implementation
gets wrong.

## The invariant worth asserting

$$\texttt{total\_counting\_number(g)} \;=\; \sum_f 1 + \sum_v (1-d_v) \;=\; |F|+|V|-|E| \;=\; \chi(g)$$

`1` for a tree, `1-L` with `L` loops. One integer saying how badly the bookkeeping over-counts.
Asserted against [[graph|`euler_characteristic`]] in the tests, because it is the cheapest check
that the graph and the accounting have not drifted apart.

## Incoming messages and permitted edge directions

`messages_into` returns the variable-to-factor cavity messages, not the full variable
marginals. Each factor forms its own belief from its potential and those messages:

$$
b_a \propto f_a \prod_{i\in a} \mu_{i\to a}.
$$

Using the marginals here would include the factor's outgoing evidence a second time.
The variable counting correction, by contrast, uses the full variable marginals.

`uses_cavity(factor)` defaults to `false`. Every `local_free_energy` implementation that
reads its incoming messages must override it with `true`, even if it returns zero when
some message is absent. In `factor_free_energies`, such a factor's edges are checked with
`absorbs(edge.direction)` before its local evaluation. `Emitting()` raises `ArgumentError`
naming the factor, channel and variable and recommending `Bidirectional()`.
`Absorbing()` and `Bidirectional()` are permitted, including when their stored message is
`nothing`. Factors returning a constant zero without reading messages keep the default.
The check scans the existing edge indices without constructing a collection; error text
is constructed only on failure. Message passing and scheduling do not change.

Both `bethe_free_energy` and graph `scalar_free_energy` call `factor_free_energies`, so all
three graph entry points enforce the same requirement. Direct `local_free_energy` calls
have no graph directions to check. `chain_free_energy` combines already evaluated terms.

The Mycelium testset "LP-014: cavity messages and edge directions" checks the three entry
points, the diagnostic, missing messages on permitted edges, and an unchanged directed
DataFactor/RelayFactor/LossFactor graph. AlgebraicFactors' "LP-014: a prior on an Emitting
edge" checks exact enumerated marginals with either prior edge direction, rejection of
Emitting-prior free energy, and the Bidirectional result against $-\log Z$.
See [[LP-014 Emitting edges drop the cavity message from the free energy]].

## Two decompositions

| | `bethe_free_energy` | `chain_free_energy` |
|---|---|---|
| shape | factors + counting correction | Theorem 23 folded along an order |
| order-dependent? | no | **yes** |
| on a tree | exact | exact |

They agree when every belief is a `DiracBelief` — all expectations are evaluations, all variable
entropies vanish. That is the regime the test suite checks against a hand-computed 4.5.

## Implementation difficulties

### 1. Theorem 23 needs a "downstream", and a graph does not have one

This is a genuine gap in transporting the paper to graphs, not an implementation shortcut.
Theorem 23 averages the upstream loss under the *downstream* inversion; on a chain that is
unambiguous, on a graph "downstream" is a property of the **message schedule**, not the wiring.
Two schedules give two decompositions of the same total.

Bethe is the standard way around it — it has no order in it at all. `chain_free_energy` is
provided for reasoning about a specific message path, and the docstring says which to prefer for
reporting.

### 2. Variable entropy needs an explicit belief representation

`DiracBelief` and `TrivialBelief` contribute `0.0`; categorical beliefs and mixtures of
categoricals use exact Shannon entropy. `gaussian.jl` adds the proper Gaussian entropy.
Unsupported representations throw.

The Dirac convention deserves scrutiny: the differential entropy of a point mass is $-\infty$,
but the *correction term* it contributes is taken as zero, on the grounds that a clamped
variable carries no free bits. That is the convention BP uses for observed nodes and it is
right, but it is a convention and it is stated rather than derived.

The counting correction cannot be computed without variable entropies; unsupported belief
representations throw rather than silently approximate.

### 3. `chain_free_energy` folds with a singleton wrapper

`compose_free_energy([Fs[i]], acc)` passes a one-element sample vector, so the "expectation
under the downstream inversion" is an evaluation at a single point. That is exact for
deterministic (Dirac) inference and **degenerate otherwise** — it silently computes
$\sigma(\mathbb{E}[\cdot])$ with a one-sample mean.

Documented, but this is the weakest function in the file. A faithful implementation needs the
actual samples from the downstream inversion, which needs a belief type that can be sampled,
which is the same blocker as everywhere else.

### 4. `LenticulumCore.scalar_free_energy` is *extended*, not shadowed

The first version defined `Mycelium.scalar_free_energy` and the test suite immediately hit
`UndefVarError … two or more modules export different bindings with this name`. The scalar free
energy of a graph and of a factor are the same concept at two scales; two exported bindings would
force every downstream user to disambiguate. Extending is the right call and multiple dispatch
makes it free.

Third and fourth name collision in the project (`Channel`, `precision` were the first two),
which is why an exhaustive `names()` scan against `Base` and `LenticulumCore` is now part of the
routine rather than something discovered one test at a time.

### 5. The grading is nested and the arithmetic assumes it

`bethe_free_energy` returns `GradedEnergy((; factors = GradedEnergy(...), variables = ...))` —
a `GradedEnergy` whose leaves are themselves `GradedEnergy`s. `LenticulumCore`'s `+`, `-`, `*`
on `GradedEnergy` recurse correctly, so this works, but it is exactly the nested-grading
awkwardness [[energy]] §2 warns about: the names come from a fold rather than from the graph in
one step.

Here the names *do* come from the graph in one step (factor names, variable names), so the
`E_G = ⊕_f E_f` grading is built correctly. It is only the outer two-way split that is a fold.

Related: [[Bethe Free Energy]], [[Scalar and Multivariate Energy]], [[energy]], [[graph]]

## Parameter training at fixed beliefs (LP-002, 2026-10-26)

`free_energy_gradient(store, g, ps, st; ad)` differentiates each scalarised local free
energy and assembles a tree matching `ps`. Nonlearnable factors get zero subtrees.
Messages are snapshotted outside AD. Bethe variable corrections are constant under
this partial derivative, so the result is the parameter gradient of their sum plus
those corrections. The same cavity edge validation as reporting applies. Each learnable factor's
`_check_factor_ad(ad, factor, factor_state)` validation runs before its local derivative,
using that factor's state subtree. It rejects SciML AD with nonempty Lux state and
reports a missing Enzyme Euler/RK4 runtime-activity setting before entering AD
(see [[neuralode]]). `parameter_step` uses the same graph gradient check.

`parameter_step(store, g, ps, st; ad, rule, optimiser_state)` evaluates the rule's
look-ahead parameters, obtains that gradient, and applies `rule_put` to each learnable
factor's flattened vector. It returns `(ps_new, st_new, optimiser_state_new)`; Lux state
advances once at the look-ahead parameters. Pass the returned optimiser state on the
next call for Momentum/Nesterov. The existing `OptimiserFactor` still operates on exposed
parameter wires with an externally supplied gradient; schedules do inference only.
There was no graph `train`/`learn` driver supplying those gradients.

This is the smallest training driver: alternate inference and parameter steps as
appropriate for the model. It neither differentiates through inference schedules nor
implements per-edge couplings, parameter sharing, stochastic gradient estimators or
frozen cotangent semantics. `ImplicitLayers/test/parameter_ad.jl` learns the coefficient
of a discrete scalar ODE to its known generating value and checks the Nesterov
look-ahead formula, with ForwardDiff, Zygote, Mooncake and runtime-activity Enzyme
in Reverse and Forward modes.
