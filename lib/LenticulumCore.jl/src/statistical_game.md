# statistical_game.jl — implementation note

Implements: [[Statistical Game]] (Def. 20), [[Composition of Statistical Games]] (Defs. 22,
25), [[Parameterized Statistical Game]] (Defs. 27, 28) — i.e. the factor interface itself.

## The composition laws, side by side

|  | AutoBayes Def. 22 | here |
|---|---|---|
| energy | $l^{dc} = l^c + l^d$ | `compose_energy` → a `GradedEnergy` pair |
| entropy | $H^{dc} = \mathbb{E}_{(y,b)\sim d'_{c_*\pi}(z)}[H^c] + H^d(c_*\pi,z)$ | `compose_entropy`, same shape, into a summand |
| loss | Thm 23: $F^{dc} = \mathbb{E}[F^c] + F^d$ | `compose_free_energy`, same |
| defect | — | `chain_rule_defect` = the Jensen gap |

The paper's `+` for energies has become a direct sum. Everything else is unchanged in shape.
Applying a `GradedScalarisation` recovers the paper exactly when $\sigma$ is linear, and
differs by `chain_rule_defect` when it is convex. See [[Scalar and Multivariate Energy]].

## Two adjectives that are the whole content of Definition 22

`compose_entropy(Hc_samples, Hd)` takes samples of $\mathbf{H}^c(\pi, y)$ at $(y,b)$ drawn
from the **downstream** inversion, at the **pushforward** prior. Get either adjective wrong
and you have a plausible-looking number that is not the free energy. The docstring says so;
there is no way to check it from inside this function, because it receives the samples
already drawn. **The correctness burden sits in the caller** — i.e. in the graph walker that
does not exist yet. Flagging it now so it is not discovered later.

## `ComposedFactor` is a `ContainerFactor`

```julia
struct ComposedFactor{A,B,C} <: AbstractLenticulumContainerFactor{(:first, :second)}
```

so `initialparameters` produces `(; first = ..., second = ...)` — the product
$\Phi \times \Theta$ of Definition 28, realised as literally the same `NamedTuple`
structure Lux uses for `Chain`. That is not a coincidence worth hiding; it is the clearest
evidence that both libraries are doing `Para`.

## Inheriting the LuxCore interface

The `for op in (:initialparameters, :initialstates)` loop extends `LuxCore.$op` for the
three factor abstract types, mirroring LuxCore's own definitions. Deliberately **not**
subtyping `AbstractLuxLayer` — see [[abstract_types]] §"Why factors are not a subtype".

## Implementation difficulties

### 1. Almost everything here is a stub, and that is the honest state

`free_energy`, `energy`, `entropy`, `assemble`, `invert`, `forward`, `pushforward` are
`function f end` declarations. The composition **combinators** are real and tested; the
things they combine are not implemented, because there is no concrete factor yet.

This is the right place to stop, but it means the design is currently unfalsified by
contact with a real model. The next step should be **one worked example end to end** — a
Gaussian factor with an exact inversion, composed with a prior, reproducing
[[Examples from the Paper|Example 1]]'s claim that $F^{c\pi}(\ast,y) = -\log p_{c_*\pi}(y)$ —
rather than more abstraction.

### 2. `compose_entropy` and `compose_free_energy` take a plain collection of samples

They average with `reduce(+, samples) * (1/length(samples))`, which assumes an unweighted
Monte Carlo estimate. A weighted particle set, a quadrature rule, or an analytic expectation
in the conjugate case all need something else. The signature should eventually take a belief
plus a functional, not a vector of evaluations — but that requires the belief interface
that [[open_model]] §4 says is still an open question. Interim: the current signature is
adequate for sampling-based inversions and wrong for the others.

### 3. `TensorFactor` has no energy/entropy law implemented

Definition 25 says both halves add across summands. `TensorFactor` currently only derives
`energyspace` and `scalarisation`; there is no `compose_tensor_energy`. Straightforward, but
absent — and it inherits the mean-field laxness of [[lens]] §3, so it should not be added
without deciding how to report that.

### 4. `chain_rule_defect` measures one edge, not a path

`chain_rule_defect(σ, Fc_samples)` gives the Jensen gap across a single composition. For a
chain of $n$ factors the total discrepancy is a sum of per-edge gaps, each taken under a
different downstream posterior — it does **not** telescope into one variance. Any "total
laxness" diagnostic must accumulate per edge during the walk, not be computed once at the
end. Currently there is no walker, so this is a note for whoever writes it.

### 5. `setup` is defined here rather than re-exported

`LuxCore.setup` would work verbatim, but defining `LenticulumCore.setup` for
`AbstractLenticulumFactor` keeps the factor interface self-contained and avoids a user
needing `using LuxCore` to set up a factor. The cost is a name that shadows nothing but
duplicates something. Reconsider if `Lenticulum.jl` ends up re-exporting LuxCore anyway.

### 6. `AbstractGradientCoupling` is declared but unused

The type exists and `ComposedFactor` stores one, but nothing dispatches on it — the gradient
machinery is not written. It is here so that the edge annotation is part of the data model
from the start rather than retrofitted, since retrofitting a per-edge choice into a global
one is much harder than the reverse. But it is currently inert, and should not be mistaken
for a working feature.

Related: [[Statistical Game]], [[Composition of Statistical Games]], [[Composition of Gradients]], [[energy]], [[LenticulumCore]]
