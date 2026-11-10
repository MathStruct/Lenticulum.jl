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

> [!note] The gradient code moved (2026-11-10)
> `free_energy_gradient` of a graph and `parameter_step` are now in `MyceliumLearning`
> ([[gradient]]). The sections below on parameter training describe that code; the free energy
> itself, the checks and the belief hooks are still in this file.

## Parameter training at fixed beliefs (LP-002, 2026-10-26)

> [!note] Superseded in part by LP-020 (2026-11-01)
> As first written, the derivative below held the *incoming messages* fixed. It now holds each
> factor's **belief** fixed, for factors that implement `factor_belief`; see the section
> [Fixed beliefs, not fixed messages](#fixed-beliefs-not-fixed-messages-lp-020-2026-11-01) below.

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

## Fixed beliefs, not fixed messages (LP-020, 2026-11-01)

**The defect.** The local free energy $F_c(\theta; \mu)$ recomputes the factor's belief
$b_c \propto f_\theta \prod_i \mu_{i \to c}$ from its incoming messages. Holding the messages fixed, the belief moves
with $\theta$, and the derivative picks up $\partial_\theta\, \mathbb E_{b_c}[\log \prod_i \mu_i]$, which is not zero. On the
two-variable tree of `test/lp020.jl` that derivative was $(-0.146, -0.441)$ against the true
$(0.679, -1.383)$, and alternating inference with it stopped at $a = 0.754$ instead of the
maximum-likelihood $1.179$. See [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]].

**The fix.** The Bethe free energy is a function of beliefs and parameters,
$F(b, \theta) = \sum_c \bigl(\mathbb E_{b_c}[-\log f_{c,\theta}] - H(b_c)\bigr) + \sum_v (d_v - 1) H(b_v)$, and belief propagation's fixed
points are its stationary points in $b$ under the normalisation and marginal-consistency
constraints (Yedidia–Freeman–Weiss). The constraints do not involve $\theta$, so by the envelope
theorem the derivative of the free energy at the fixed point is the partial derivative at fixed
beliefs:

$$
\frac{\mathrm d}{\mathrm d\theta} F(b^*(\theta), \theta) = \sum_c \mathbb E_{b^*_c}\bigl[-\partial_\theta \log f_{c,\theta}\bigr].
$$

The variable entropies drop out. Two hooks in `factors.jl` express it:

- `factor_belief(factor, msgs, ps, st)` returns the factor's belief in some form of the factor's
  choosing. The gradient evaluates it once, outside AD.
- `belief_free_energy(factor, b, ps, st)` is $F_c(b, \theta)$ at that fixed belief, graded like
  `local_free_energy`, and equal to it at the belief it was evaluated at.

`free_energy_gradient` differentiates `belief_free_energy` at the fixed belief. On a tree at the
fixed point this is the gradient of $-\log p$. On a loopy graph it is the gradient of the Bethe
approximation at the loopy fixed point, which is what training against that approximation means.

**Implementations.** In Lenticulum (`src/gaussian.jl`, `FrozenGaussianBelief`):
`GaussianFactor` (with learnable noise), `GaussianPrior` (learnable noise),
`LinearConstraintFactor`, `SwitchingConstraintFactor` (whose frozen belief includes the outcome
posterior $q$, so $A$, $b$ and the log-weights all get the envelope gradient), and the collapsed
Gaussian composite. Every one of these has an affine residual $r = \sum_i R_i x_i + r_0$, so one
function builds the frozen joint Gaussian of the non-point channels. `belief_free_energy`
re-evaluates the residual's mean and covariance under it at the current parameters.

In AlgebraicFactors (2026-11-02): `ToricFactor`, whose frozen belief is the joint table over its
states (`ToricBelief`), and the degree-one `PolynomialFactor`, whose frozen belief is the joint
Gaussian of its free coordinates (`PolynomialBelief`); `toric.md` and `polynomial.md` give the
formulas and the checks in `parked/AlgebraicFactors.jl/test/lp020.jl`.

**What it does not do.**
- A factor without `factor_belief` keeps the fixed-message derivative. That is the same thing
  when every incoming message is a point (the belief is then a point too), and in general it is
  not. A factor with `uses_cavity` whose messages are not all points logs a warning once. The
  hook-less learnable factors in the repository are the neural and diffusion factors, whose
  graph uses are with point messages. `PriorFactor` has no hook and needs none: neither its
  belief nor its `nlogp` reads the parameters, so its parameter derivative is zero either way.
- The argument is about the **summed** free energy. A nonlinear per-factor scalarisation changes
  the objective that belief propagation does not minimise, and the fixed-belief derivative is then
  not its gradient.
- The fixed point must be at the parameters the gradient is evaluated at. `parameter_step` with a
  look-ahead rule evaluates at look-ahead parameters with the current messages, as before.
- Non-diagonal couplings and `adjoint = :implicit` differentiate through inference
  ([[gradient_couplings]], [[graph_adjoint]]) and are unchanged. `sample_gradient` uses this
  gradient per sample by default ([[sampling]]).

**Checked** in `test/lp020.jl`. On four trees (the linear constraint above; a Gaussian
factor and a Gaussian prior, both with learnable noise; the collapsed composite; a switching
constraint) the graph gradient matches central differences of the converged Bethe free energy,
with inference re-run at every perturbed parameter, to $10^{-6}$ with reference differences,
ForwardDiff and Zygote. `belief_free_energy` equals `local_free_energy` at the frozen belief to
$10^{-12}$. On the linear constraint, alternating inference with gradient steps now converges to
the maximum-likelihood $a$.

**Difficulties.**
1. The switching test first put a Gaussian prior on the latent channel. That prior then receives
   the factor's Gaussian-mixture message, and no Gaussian factor's free energy accepts a mixture
   (its entropy has no closed form). The test uses a degree-1 latent instead. Its belief is
   proper because $A_x$ is invertible.
2. Even a degree-1 variable had its marginal's entropy evaluated, and multiplied by the
   counting number 0. That threw for the mixture marginal. Degree-1 variables are now skipped
   (`_variable_term`).
3. Zygote has no rule for `inv(::Cholesky)`. The constraint factor's fixed-belief energy uses
   `inv(Symmetric(Q))`, as the Gaussian factor already did.
4. A bounded review by Mistral (vibe) of the diff found no correctness bug. It made one change:
   trivial and missing messages are not points, so they trigger the fallback warning.

## The merged free energy of belief propagation and mean field (2026-11-02)

[[L-SPEC-005 The Hybrid Free Energy]]. A graph with `MeanField()` nodes is scored by Riegler et al.
(2013), §III, eq. (17), checked against the paper (arXiv:1112.0467v3):

$$
F_{BP,MF} = \sum_{a \in A_{BP}} F_a(b_a) + \sum_{a \in A_{MF}} \mathbb E_{\prod_i b_i}[-\log f_a]
 + \sum_i \bigl(|N_{BP}(i)| - 1\bigr) H(b_i).
$$

- `factor_free_energies` calls `mean_field_energy(factor, marginals_into(s, g, f), ps, st)` for a
  mean-field node: the marginals, which are what its messages read, not the cavities.
- `counting_number(g, v) = 1 - bp_degree(g, v)`. A variable whose neighbours are all mean-field
  nodes counts 1, its full entropy, as in naive mean field. So `total_counting_number` is the Euler
  characteristic only when every node is on belief propagation.
- The gradient differentiates `mean_field_energy` at fixed marginals. At a fixed point of the
  hybrid messages, a stationary point of $F_{BP,MF}$ (their Theorem 2), this is the derivative of
  the converged free energy, by the same envelope argument as LP-020.
- `_check_energy_edges(g; mean_field = false)` keeps the refusal on the paths written for
  belief propagation: the coupled and implicit adjoints, which differentiate through inference,
  and the sample buffer, which records residual moments from cavity messages. The cavity
  edge-direction check (LP-014) skips mean-field nodes, which read marginals.

**Difficulty.** Building the product of the marginals inside the differentiated function made
Zygote trace `push!`. Lenticulum's ChainRulesCore extension marks the builder non-differentiable:
it depends on the marginals only, which the gradient holds fixed.

**Checked** in Lenticulum's `test/mean_field_free_energy.jl`; the numbers are in L-SPEC-005.

## LP-011 implementation (2026-10-27)

The all-diagonal driver retained its fixed-cavity derivative (replaced by the fixed-belief one,
LP-020) and now masks frozen coordinates. Other couplings use the explicit replay implementation.
`parameter_step` applies look-ahead and updates only active coordinates; fully pinned
factors have no optimiser key. Score steps return their GradientEstimate as a fourth
entry. Factor free-energy evaluation uses a functional map with the same per-factor
state threading, so reverse AD avoids mutating a values array.

Evidence: `lib/MyceliumLearning.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].


## Device parameter driver (2026-10-27)

The GPUArraysCore extension implements active-coordinate gathering, replacement,
masking and pinning without host scalar reads. Default diagonal gradients and
coupled replay use the same helpers. CPU helpers retain their original arithmetic.
Device parameter steps cast GradientDescent/Momentum/Nesterov coefficients to
parameter precision, preserving Float32 even with the default Float64 learning
rate. Optimiser vectors still contain only active coordinates. Tests in
`parked/ImplicitLayers.jl/test/device_arrays.jl` check closed-form Gaussian/Dirac
objectives, cut/freeze behavior, score estimates and actual DEQ/ODE factor scoring.
See [[gradient_couplings]] for the supported device subset and its limits.


## Batch C graph parameter adjoints (2026-10-28)

`free_energy_gradient` and `parameter_step` forward `damping` and support
`adjoint=:implicit`, `maxsweeps`, `tol`, `linear_solver=:dense/:krylov`,
`linear_tol` and `linear_maxiter`. Defaults retain local all-diagonal
training (fixed-belief since LP-020) and finite replay for non-diagonal graphs. The implicit opt-in starts
reference inference from empty messages and includes both local losses and
Bethe corrections in its adjoint RHS. The resulting tree gradient has a dense
Gaussian negative-log-partition oracle; loopy tests reconverge the Bethe
objective at every perturbation. See [[graph_adjoint]] for the equation, costs,
cuts, CPU/empty-message-state restriction and residual checks.

Flooding/residual/damped finite derivatives and their realised-schedule semantics
are specified in [[gradient_couplings]] and tested in `test/graph_adjoint.jl`
and `test/graph_adjoint_devices.jl`. Native Enzyme/Mooncake replay is still
rejected; the reproducer and backend findings are recorded in LP-002.

## Tied parameters (2026-11-02)

`free_energy_gradient` returns one entry per parameter key (`parameter_keys(g)`). For a key read
by several nodes it evaluates each node's term outside AD as before (its belief, its mean-field
marginals, or its messages for a factor without `factor_belief`) and differentiates the sum of
their scalars in one AD call (`_gradient_item`, `_tied_scalar`); one node is differentiated as
before. `parameter_step` keeps one optimiser state per key. `factor_free_energies` still reports
one energy per node, keyed by node name, and threads state by key. `scalar_free_energy` sums
the same terms as a vector (`_factor_terms`), so it does not build a type with one field per node,
which recompiled for every plate size (`plates.md`). Checked in `test/plates.jl`
against the sum of separate graphs' gradients. The coupled drivers differentiate the whole of
`ps` and sum tied entries without change.

## Batched energies over plates (2026-11-03)

With message blocks (`blocks.md`, [[L-ADR-0010 Plates get batched message blocks]]), a plate factor
row whose incoming messages are blocks is evaluated in one call: `_factor_terms` and
`scalar_free_energy` use `local_free_energies(factors, msgs, ps, st)`, and `free_energy_gradient`
gives the row one item, `_beliefs_scalar` over `factor_beliefs` (evaluated outside AD) and
`belief_free_energies` (differentiated). Rows are found by `_energy_rows`: belief-propagation nodes
on one key, edges in block rows in one channel order, replica $i$ in column $i$, none detached. A
variable row's Bethe terms are the entropies of its marginal block (`_variable_terms_rows`). A
store without rows (the coupled drivers' replay, or one built from vectors) takes the per-node path
unchanged, which also keeps that path free of mutation for Zygote. The scalar sum over a row is a
function of the concretely typed factor vector, so `scalarisation` is not dispatched per node.
Tested in `test/plates.jl` against the same messages per edge; timings in `blocks.md`.


## Expectation-propagation and mixture-reduction nodes (2026-11-03)

An `ExpectationPropagationRule()` node is counted like a belief-propagation node: its
`local_free_energy` at its incoming messages, and its edges in each variable's counting number
(`bp_degree` counts both rules, `_cavity_rule` in `graph.jl`). At an EP fixed point the factor
belief's marginals match the variables' Gaussian marginals in their moments rather than
exactly, and the same expression is the free energy whose stationary points those fixed points
are: `test/mixture_rules.jl` checks it numerically, the fixed-belief gradient against the
derivative of the re-converged free energy (`passing.md`). The default drivers accept EP nodes;
the paths that differentiate through inference (non-diagonal couplings, `adjoint = :implicit`)
and the residual-moment recorder still refuse anything but belief propagation. Energy rows of a
plate (`_energy_row`) accept EP nodes too, when every node of the row has the same rule.

A `MixtureReduction(M)` node is refused by every energy path (`_check_energy_edges`). Its fixed
points are not the stationary points of a free energy we can name, and Lenticulum's local free
energies collapse mixture inputs to their moments, so the number would describe other beliefs
than the ones the node sent.

The Bethe free energy of a belief-propagation graph whose marginal is a Gaussian mixture is not
available either: `variable_entropy` has no method for a Gaussian mixture, which has no closed-form
entropy. A latent mixture on a plain BP graph therefore has messages but no free energy; on EP
nodes its marginals are Gaussian and it has both.

## Learned relations that are not densities of their data (2026-11-04, LP-022)

`free_energy_gradient` warns once per session when a learnable node whose factor
`learns_unnormalised` (a hook, `false` by default; Lenticulum's constraint factors without an
`output`, AlgebraicFactors' unnormalised `ToricFactor`) has no frozen `A` and a channel on a variable that a `DataFactor` observes; inside a
cluster node it looks at the clustered factors through their wiring. Such a factor is normalised in
its residual, so its free energy on data can fall without fitting (LP-022). It is a warning, not a
refusal: a fit may hold the coefficients fixed in its own loop. Detection only covers data from
`DataFactor`s. Test: Lenticulum `test/lp022.jl`.

Since 2026-11-05 the hook also has a form by channel, `learns_unnormalised(factor, observed)` (the
channels a `DataFactor` observes), which by default is the one-argument form when any channel is
observed; `DEQFactor` overrides it to fire only with its state observed.

