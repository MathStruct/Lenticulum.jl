# Loopy Message Passing

> What happens off a tree. Short answer: no guarantees, several partial remedies, and one
> problem specific to this project that I have not seen treated anywhere.

## What is lost

On a tree, [[Schedules|two sweeps]] give the true marginals and the true free energy. Off a
tree, all three of these fail:

1. **Convergence is not guaranteed.** Flooding BP can oscillate indefinitely, and does, on
   graphs with strong couplings around short loops.
2. **Fixed points are not unique.** Multiple fixed points can exist, and which one you reach
   depends on the initialisation and the [[Schedules|schedule]].
3. **A fixed point is not the true marginal.** It is a stationary point of the
   [[Bethe Free Energy|Bethe free energy]] (Yedidia–Freeman–Weiss), which equals the true free
   energy only on a tree — the deficit being measured by the Euler characteristic
   $\chi = 1 - L$.

None of these is a defect of the implementation. Exact inference on a loopy graphical model is
#P-hard; loopy BP is the price of tractability, and knowing exactly what you traded is better
than not running it.

## The remedies, and what each actually does

### Damping

$$\mu^{\text{new}} \;\leftarrow\; \alpha\,\mu^{\text{new}} + (1-\alpha)\,\mu^{\text{old}}$$

The standard cure for oscillation: it cannot create a fixed point that was not there, but it
can stop the iteration overshooting one that was. Costs convergence *speed* when the iteration
was already fine.

Implemented only where a convex combination of beliefs is defined — for `DiracBelief` with
numeric payloads, interpolation of the values. Elsewhere it is a no-op, and `can_damp` reports
that rather than letting a caller believe damping is active when it is not.

### Residual scheduling

Send whichever pending message would change the most, rather than sweeping uniformly
(Elidan, McGraw & Koller). Converges on many graphs where flooding does not, because it spends
its effort on the parts that have not settled.

Requires a usable `belief_distance`; with the current belief types most pairs return `Inf` and
it degrades to an arbitrary order. Recorded, not hidden.

### Reporting rather than pretending

`ConvergenceReport` carries `converged`, `sweeps`, `residual` and a `reason`. Two deliberate
choices:

- **`belief_distance` returns `Inf` for unknown pairs, not `0`.** An unverifiable residual makes
  the loop run to `maxsweeps` and report "convergence cannot be verified". A false "converged"
  is much worse than a wasted sweep.
- **`converged == false` is not an error.** On a loopy graph it is the expected outcome; the
  caller decides whether the marginals are good enough.

## What this framework adds to the standard story

Three things that are not in the ordinary loopy-BP literature, because they come from the
[[Statistical Game|statistical game]] structure:

### 1. The approximation is quantified in two independent ways

- **Bethe deficit**: $\chi(g) = 1 - L$, an integer, computed from the graph alone
  ([[Bethe Free Energy]]).
- **Mean-field laxness**: whenever a factor sends separate messages about coupled channels, the
  discarded correlation is the mutual information between them
  ([[Composition of Bayesian Lenses|Remark 16]], [[Composition of Statistical Games|Remark 26]]).

Both are reportable. The framework's recurring instruction — *track the laxness, do not hide
it* — has a concrete meaning here.

### 2. Approximate inversions are already legal

A solver that stopped early, a message computed with a damped or truncated inversion, a
moment-matched pushforward: all of these are just **inexact $c'$**, which
[[Bayesian Lens|Definition 9]] permits outright. The loss gets worse; nothing breaks.

That is a much more graceful failure mode than an unrolled iterative solver that diverges, and
it is a genuine architectural advantage of building on Bayesian lenses rather than on
differentiable fixed points.

### 3. Monodromy: branch labels do not survive a loop

This one is specific to Lenticulum and I have not seen it discussed in the implicit-layer
literature.

[[Branches and the Discriminant]] shows that for an implicit factor the latent space
$\llbracket c \rrbracket$ is the **set of solution branches**, and that transporting the
observed input around a loop can **permute** them (the monodromy group). Consequences:

> **Message passing around a cycle in the factor graph can return to a different branch than
> it started on.** If the monodromy is transitive there is no continuous global choice of
> branch at all, so any implementation that caches "which branch we took last time" is making a
> *local* choice that cannot be made global.

This is a correctness issue, not a convergence issue: the iteration can converge perfectly to a
consistent assignment that is on the wrong sheet. It affects exactly the graphs Lenticulum
exists to allow — loopy ones with multi-valued factors — and it has no known remedy. Recorded
in [[Open Problems in Algebraic Implicit Learning]] §7b, and it lands here.

## Practical guidance

| situation | do |
|---|---|
| tree | `tree_schedule`; one sweep; exact; done |
| DAG, unidirectional | `forward_backward_schedule`; this is a Lux forward pass |
| few short loops | `flooding_schedule` with damping ≈ 0.2–0.5 |
| many loops, slow convergence | `ResidualSchedule` |
| oscillating regardless | accept it; report the residual; do not average the oscillation and call it a marginal |

And: **check `istree(g)` before believing any number the graph produces.** It is one integer
subtraction and it decides whether you have an answer or an approximation.

Related: [[Schedules]], [[Bethe Free Energy]], [[Branches and the Discriminant]], [[passing]]
