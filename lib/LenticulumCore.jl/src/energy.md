# energy.jl — implementation note

Implements: [[Scalar and Multivariate Energy]] — the two-energy design, and the algebra that
makes AutoBayes' Definition 22 a *corollary* of ours rather than a competitor.

## What is implemented

| concept | type |
|---|---|
| $(E, K)$ energy space | `ScalarEnergySpace`, `EuclideanEnergySpace`, `GradedEnergySpace` |
| $E_{dc} = E_c \oplus E_d$ | `oplus` / `⊕`, `GradedEnergySpace` |
| an element of $E_G$ | `GradedEnergy` (a `NamedTuple` keyed by factor) |
| $\sigma : E \to \mathbb{R}$ | `IdentityScalarisation`, `WeightedSum`, `SquaredNorm`, `GradedScalarisation` |
| linear vs convex | `islinear` |
| the Jensen gap | `jensen_gap` |

`GradedEnergy` carries `+`, `-`, scalar `*` and `zero` because
[[Composition of Statistical Games|Definition 22]]'s entropy law takes an **expectation** of
energies, which a bare monoid cannot support.

## The two theorems, in code

**Strict for linear $\sigma$.** `jensen_gap(σ, samples) == 0.0` short-circuits on
`islinear(σ)`, and the test asserts
`scalarise(σ, mean(Fc)) ≈ mean(scalarise.(Ref(σ), Fc))`.

**Lax for convex $\sigma$, by exactly a variance.** For `SquaredNorm()` the test asserts

$$\texttt{jensen\_gap} \;=\; \tfrac12 \operatorname{tr}\operatorname{Cov}(\mathbf{F}^c)$$

against `Statistics.cov(...; corrected=false)`, and

$$F^{dc}_{\text{scalar}} \;=\; \sigma\bigl(\mathbf{F}^{dc}\bigr) \;+\; \texttt{chain\_rule\_defect}$$

with $\sigma(\mathbf{F}^{dc}) \le F^{dc}$. These are the two claims the whole design rests
on, so they are tested numerically rather than argued.

## Recovering the paper exactly

```julia
energyspace(f)    = ScalarEnergySpace()
scalarisation(f)  = IdentityScalarisation()
```

are the defaults on `AbstractLenticulumFactor`. A factor that ignores the vector machinery
behaves exactly as AutoBayes Definition 20 specifies. **The generalisation is opt-in**,
which is the only acceptable way to ship it — a reader of the paper must be able to write a
factor without learning our extension first.

## Implementation difficulties

### 1. Eager accumulation is only valid for linear $\sigma$

This is the difficulty that motivates `islinear` existing at all. Walking a graph and
adding up scalar losses as you go computes $\mathbb{E}[\sigma(\mathbf{F})]$. The
multivariate chain rule asks for $\sigma(\mathbb{E}[\mathbf{F}])$. For linear $\sigma$ these
coincide and eager accumulation is free; for convex $\sigma$ they do not, and the vector
loss must be assembled first.

There is no way to detect this at runtime after the fact — both are finite numbers, both
look plausible, and the wrong one is *smaller*, so it will look like your model is doing
better. Hence the trait.

### 2. `oplus` on unnamed operands invents `:left` / `:right`

Folding a binary `⊕` gives nested rather than flat gradings, so associativity holds only up
to renaming. Acceptable (composition of open models is likewise bicategorical) but a trap.
**Build $E_G$ in one step from a node-keyed `NamedTuple` once the graph layer exists**;
treat binary `oplus` as a test-and-prototyping convenience.

### 3. The entropy has to be told *which coordinate* it regularises

$\mathbf{H}^c$ is typed into $K_c$, but entropy is naturally a scalar. The idiom is to
return `h * u` for a fixed direction `u ∈ K_c`. This is not busywork — it is what makes it
possible to regularise one block of a factor and not another, which is the whole point of
per-coordinate precisions. But it *is* an extra decision imposed on every factor author,
and a `ScalarEntropy` wrapper that broadcasts against a default direction would be worth
adding once there is evidence about what direction people actually pick.

### 4. `SquaredNorm(M)` has an unconstrained `M`

`dot_quadratic(M, e) = sum(e .* (M * e))` assumes `M` is symmetric positive semidefinite;
nothing checks it. If `M` is indefinite, $\sigma$ is not convex, `islinear` still reports
`false`, and `jensen_gap` may go **negative** — silently invalidating the "multivariate is a
lower bound" claim. Either validate `M` at construction or document the precondition
loudly. Currently: documented here, not enforced. This is the sharpest outstanding
correctness gap in the file.

### 5. `scalarise` on `GradedScalarisation` requires matching names

`scalarise(σ::GradedScalarisation{names}, e::GradedEnergy{names})` is constrained to
identical `names`. A mismatch is a `MethodError` — which is the right failure (loud, at the
right place), but the message will be unhelpful. A custom error explaining "the
scalarisation's grading does not match the energy's" is worth adding.

### 6. Not yet implemented: the Jacobian side

[[Scalar and Multivariate Energy]] §6 is the *reason* for the vector energy —
$J^\top J$ for Gauss–Newton, the Fisher metric of Definition 27, and the implicit function
theorem. None of that is here. `energy.jl` currently gives you the *values* and the
*algebra*; extracting $D_\theta \mathbf{F}$ needs an AD backend, which `LenticulumCore`
deliberately does not depend on. It belongs in `Lenticulum.jl` proper, as an extension per
backend. **Until that exists, the design's main payoff is unrealised** — worth saying
plainly rather than letting the abstraction stand in for the result.

Related: [[Scalar and Multivariate Energy]], [[Statistical Game]], [[statistical_game]]
