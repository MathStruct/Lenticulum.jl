# deq.jl — implementation note

> `DEQFactor`: the fixed-point condition $z = g_\theta(z,x)$ kept as a **residual**, so the
> polarity decides which channel to solve for.

## 1. What SciML gives you, and what it keeps

`DeepEquilibriumNetworks.jl` builds a Lux layer. You hand it a cell and a
`NonlinearSolve.jl` solver, and you get `x ↦ z★(x)` — the relation solved in one direction,
with the solve sealed inside a function and the solve's diagnostics tucked into `st` as a
`DeepEquilibriumSolution`.

Everything in that sentence is good engineering and none of it is a relation any more. The
object that *is* a relation is the residual:

$$r(x,z) = z - g_\theta(z,x), \qquad R_\theta = \{(x,z) : r(x,z) = 0\}$$

which is [[README]]'s definition of an implicit learner, verbatim. `DEQFactor` keeps it, and
therefore gets both directions out of one solver:

| polarity | solves | who can do this |
|---|---|---|
| `x` observed, `z` unobserved | $z = g(z,x)$ for $z$ | SciML's DEQ, and this |
| `z` observed, `x` unobserved | $z = g(z,x)$ for $x$ | **only this** |

Same residual, same `BroydenSolver`, different variable held fixed. The test suite checks
both against the closed form for a linear cell, and checks that they really are inverse to
each other.

## 2. The energy is a proper vector energy

`energyspace` is `EuclideanEnergySpace(dim z)` and the scalarisation is `SquaredNorm` — the
residual is returned as an $\mathbb{R}^n$-valued object, not collapsed to a number.

This is what [[Scalar and Multivariate Energy]] asks every implicit factor to do and what
`VariationalDiffusion.jl`'s factor could not deliver (its `predictor.md` §4.3 records the
inconsistency). Here it comes for free, and it is exactly what the IFT needs — that note's
§6.3 is about needing the Jacobian of the *vector* residual.

**So the equilibrium family is the first one to satisfy the vault's own energy design.**

## 3. The reverse direction needs a square residual

Solving $z = g(z,x)$ for `x` is $\dim z$ equations in $\dim x$ unknowns. Unless
$\dim x = \dim z$ it is over- or under-determined, and a root-find is not asking a well-posed
question.

`supports_polarity` refuses that polarity when the dimensions differ, and
`supported_polarities` returns one element rather than two — so a non-square DEQ correctly
reports `isunidirectional == true`, i.e. *it really is just a Lux layer*. Asserted in the
tests.

> The honest reading: the bidirectionality of a DEQ factor is conditional on a shape
> coincidence. A DEQ whose input and state have different widths — which is the common case
> in practice, since the input is usually an embedding — is unidirectional and gains nothing
> from this package over `LuxFactor`. What would help is a least-squares solve for the
> over-determined case and a regularised one for the under-determined case; neither is
> implemented.

## 4. Implementation difficulties

### 4.1 Every message is a `DiracBelief`

A root-find returns a point. Propagating a *distribution* would need the linearisation
`deq_sensitivity` already computes — push a covariance through $\partial z^\ast/\partial x$ —
and somewhere to put it, i.e. a `GaussianBelief`.

**`GaussianBelief` lives in the top-level `Lenticulum` package**, which a `lib/` package must
not depend on. So this factor cannot return one, `VariationalDiffusion`'s cannot either, and
the belief type that makes message passing work is unreachable from every package that needs
it. Three factor packages, one wall — see [[The Equilibrium Family]] §5 for the argument that
`GaussianBelief` belongs in `LenticulumCore`.

### 4.2 The message is a posterior, not a likelihood

The same problem `VariationalDiffusion`'s `factor.md` §5.1 records, for the same reason: there
is no way to divide a neural network's contribution out of the answer. The message is correct
when the target variable has degree 1 and double-counts otherwise, and nothing detects it.

The `prior` argument is used as the solver's **warm start** and nothing else. That is a
genuinely good use — warm-starting from the current belief is what makes iterated message
passing over implicit layers affordable, and the test suite checks that starting at the
answer costs zero iterations — but it is *not* conditioning, and calling the argument `prior`
invites the confusion.

### 4.3 No entropy, hence nothing for the Bethe correction to correct

`local_free_energy` returns $\tfrac12\|r\|^2$ and no entropy term, because a Dirac has none.
So a `DEQFactor` contributes energy only. In a graph mixing it with a `GaussianFactor`, the
counting correction of [[Bethe Free Energy]] is applied to variables whose entropy this factor
never charged — the total is not $-\log p(y)$ for anything. Same conclusion as the diffusion
factor reached, and for the same underlying reason: **Dirac-valued inversions do not have
free energies in the sense the Bethe formula wants.**

### 4.4 The initial guess is zeros

`solve_state` starts from $z_0 = 0$ unless warm-started. `SkipDeepEquilibriumNetwork` exists
precisely because a *learned* initial guess $z_0 = h_\phi(x)$ converges much faster and
regularises training. That is a second parameter tree on the inversion — the
`AmortisedInversion` situation of [[Bayesian Lens]] — and implementing it would make this the
first factor in the project with genuinely two parametrised halves. Not done.

### 4.5 `deq_sensitivity` costs $O(\dim x + \dim z)$ forward passes

Two dense finite-difference Jacobians. See [[solve]] §4.1: the real answer is a VJP, and this
family cannot avoid needing derivatives the way the diffusion family did.

Related: [[solve]], [[luxfactor]], [[neuralode]], [[DEQ as a Relation]],
[[The Equilibrium Family]], [[Scalar and Multivariate Energy]], [[Implicit Learners]]
