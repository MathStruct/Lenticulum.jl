# free_energy.jl — implementation note

Implements: the graph generalisation of [[Composition of Statistical Games|Theorem 23]] — the
Bethe free energy, its counting numbers, and the graded per-factor breakdown.
Theory: [[Bethe Free Energy]].

## The formula

$$U = \sum_f U_f, \qquad H = \sum_f H_f - \sum_v (d_v - 1) H_v$$

**Energies add; entropies need a counting correction.** That is exactly the asymmetry
[[Variational Free Energy|Proposition 18]] identifies, appearing again one level up — which is
the reason this file exists rather than a one-line `sum(local_free_energy, factors)`. The
one-liner is the energy part, and it is right; the entropy part is what a naive implementation
gets wrong.

## The invariant worth asserting

$$\texttt{total\_counting\_number(g)} \;=\; \sum_f 1 + \sum_v (1-d_v) \;=\; |F|+|V|-|E| \;=\; \chi(g)$$

`1` for a tree, `1-L` with `L` loops. One integer saying how badly the bookkeeping over-counts.
Asserted against [[graph|`euler_characteristic`]] in the tests, because it is the cheapest check
that the graph and the accounting have not drifted apart.

## `beliefs_at` uses the **full** marginal, not the excluded one

Exclusion is a rule about *messages* — it stops a factor's own claim returning to it as
independent evidence. The free energy is evaluated at the graph's **actual beliefs**, and
excluding there would be wrong. Easy to get backwards; called out on the docstring.

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

### 2. `variable_entropy` is implemented for exactly two types

`DiracBelief` and `TrivialBelief`, both `0.0`. Everything else throws.

The Dirac convention deserves scrutiny: the differential entropy of a point mass is $-\infty$,
but the *correction term* it contributes is taken as zero, on the grounds that a clamped
variable carries no free bits. That is the convention BP uses for observed nodes and it is
right, but it is a convention and it is stated rather than derived.

For any real belief type this is the blocking gap: **the counting correction cannot be computed
without variable entropies**, so on a graph with non-Dirac beliefs `bethe_free_energy` currently
throws rather than approximates. Throwing is the right failure mode; the gap is real.

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
