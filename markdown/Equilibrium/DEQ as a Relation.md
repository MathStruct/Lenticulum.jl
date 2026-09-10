# DEQ as a Relation

> A deep equilibrium model is defined by a relation and shipped as a function. This note is
> about the difference, and about what it costs to get the relation back.
>
> Implemented as `DEQFactor` in `lib/ImplicitLayers.jl/src/deq.jl`; see [[deq]].

## 1. The model

$$z^\ast = g_\theta(z^\ast, x)
\qquad\Longleftrightarrow\qquad
r(x,z) = z - g_\theta(z,x) = 0$$

Bai, Kolter and Koltun's observation was that an infinitely deep weight-tied network is a
fixed point, that you can find it with a root solver instead of unrolling, and that you can
differentiate it with the implicit function theorem instead of storing activations — $O(1)$
memory regardless of iteration count.

For this vault the interesting sentence is the first equality. $R_\theta = \{(x,z) : r = 0\}$
is a **relation**, and [[README]]'s table is about exactly that object:

| | explicit | implicit |
|---|---|---|
| approximator | $f_\theta : X \to Y$ | $R_\theta \subseteq X_1\times\cdots\times X_n$ |
| inference | forward evaluation | **root-finding** |
| symmetry | fixed output direction | **no distinguished input/output** |

A DEQ scores "root-finding" on row two. It scores "fixed output direction" on row three,
because `DeepEquilibriumNetwork(cell, solver)` solves for $z$ and only for $z$.

## 2. Both directions, one residual

```julia
solve_state(f, x, ps, st)     # solve r(x, z) = 0 for z  — what SciML does
solve_input(f, z, ps, st)     # solve r(x, z) = 0 for x  — what it cannot be asked
```

Same residual, same `BroydenSolver`, different variable held fixed. Checked against the
closed form for a linear cell $g = Wz + Ux + b$:

$$z^\ast = (I-W)^{-1}(Ux+b)
\qquad\qquad
x^\ast = U^{-1}\bigl((I-W)z - b\bigr)$$

and checked to be mutually inverse.

What does the reverse direction *mean*? It is the question "which input would have produced
this equilibrium?" — inversion of a trained implicit layer. In a factor graph it is what lets
evidence at the output travel back to the input, which is the whole reason
[[Messages are Inversions]] exists.

> [!warning] It is conditional on a shape coincidence
> $\dim z$ equations in $\dim x$ unknowns is only square when $\dim x = \dim z$.
> `supports_polarity` refuses the reverse polarity otherwise, so a DEQ whose input is an
> embedding of a different width reports `isunidirectional == true` and gains nothing over
> [[luxfactor|`LuxFactor`]]. The common practical case is the unidirectional one.
>
> Least-squares for the over-determined case and a regularised solve for the under-determined
> one would fix this and are not implemented.

## 3. The contraction caveat, made testable

[[Implicit Learners]] §Equilibrium records the caveat and leaves it as prose:

> *this only works if the iteration converges, and unconstrained DEQs need not.*

With a linear cell it becomes arithmetic. Picard iteration $z \leftarrow Wz + Ux + b$
converges iff $\rho(W) < 1$. So:

| $\rho(W)$ | fixed point exists? | Picard | Broyden |
|---|---|---|---|
| $0.5$ | yes, unique | converges | converges, faster |
| $1.6$ | **yes, unique** | **diverges** | converges |

The middle column is the point. At $\rho(W) = 1.6$ the fixed point $(I-W)^{-1}c$ exists and
is unique — nothing is wrong with the *model*. What fails is the *naive solver*, and a
quasi-Newton method has no trouble at all. Asserted in the test suite.

This is why `DEQFactor` defaults to `BroydenSolver` and why `PicardSolver` is present mainly
as the thing that demonstrates the problem. It is also what
`DeepEquilibriumNetworks.jl` concluded — Broyden and limited-memory Broyden are its
workhorses, and the whole point of routing through `NonlinearSolve.jl` is to get better
solvers than "apply the layer repeatedly".

> [!note] Convergence is reported, not enforced
> `SolveReport(converged, iterations, residual, nfe)` is threaded out and nothing throws.
> That is [[Bayesian Lens]]'s position taken literally: *a solver that stopped early is simply
> an inexact inversion, and the loss records the cost.* A non-convergent DEQ message is a bad
> message, not an error — which is a far more graceful failure than a divergent unroll.

## 4. The backward pass is where AD comes back

$$(I - \partial_z g)\frac{\partial z^\ast}{\partial x} = \partial_x g$$

One linear solve, no unrolling — see [[Backpropagation by the Implicit Function Theorem]].
And unlike [[The Diffusion Family]], which escaped automatic differentiation entirely via
RED-Diff's stop-gradient, **there is no version of this that skips the Jacobian.** The IFT is
the Jacobian.

`lib/ImplicitLayers.jl` computes both Jacobians by dense finite differences and says plainly
that this is a test-scale tool: $O(n)$ forward passes where a VJP would be $O(1)$. It exists
so the IFT can be checked against $(I-W)^{-1}U$, not so it can be used.

### The singular case is the discriminant

`ift_sensitivity` throws when $I - \partial_z g$ is singular. That is not a numerical edge
case — it says the fixed point is not locally unique, i.e. the relation **branches**. The
algebraic family has a name for that locus: [[Branches and the Discriminant]].

So two of the three [[Implicit Learners]] families fail in the same place, for the same
reason, and had different vocabulary for it. In the algebraic case the discriminant is
computable in advance by elimination; in the equilibrium case you find out when the linear
solve fails.

## 5. What the factor cannot do

- **Return anything but a point.** The message is a `DiracBelief`. `deq_sensitivity` computes
  exactly the linearisation a covariance would be pushed through, and there is nowhere to put
  the result — see [[The Equilibrium Family]] §5.
- **Divide out its prior.** So the message is a posterior rather than a likelihood, correct at
  degree 1 and double-counting beyond. The same wall [[The Diffusion Factor]] §4.1 hits, for
  the same reason: you cannot subtract a neural network.
- **Contribute an entropy.** Energy only, so the Bethe correction has nothing to correct
  ([[The Equilibrium Family]] §6).
- **Start from a learned guess.** `SkipDeepEquilibriumNetwork` exists because $z_0 = h_\phi(x)$
  converges much faster. That is a second parameter tree on the inversion — the
  `AmortisedInversion` case of [[Bayesian Lens]] — and would make this the first factor in the
  project with genuinely two parametrised halves. Not implemented.

Related: [[The Equilibrium Family]], [[NeuralODE as an Invertible Factor]],
[[Implicit Learners]], [[Backpropagation by the Implicit Function Theorem]],
[[Branches and the Discriminant]], [[Messages are Inversions]], [[deq]], [[solve]]
