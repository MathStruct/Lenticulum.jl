#implementation

> Closed-form M-steps from cluster beliefs: `em_step` collects each clustered factor's moment
> statistics from the exact beliefs and asks the factor for its maximiser (`m_step`); `fit_em`
> alternates inference and M-steps, optionally SQUAREM-accelerated. With every learned factor in a
> cluster node this is EM, and it reproduces hand-written EM step for step.

> Sources: Varadhan & Roland, *Simple and Globally Convergent Methods for Accelerating the Convergence of Any EM Algorithm*, Scand. J. Statist. 35(2), 2008 (SQUAREM, scheme S3). code: `em.jl`; Lenticulum `src/m_steps.jl` (the constraint factors' `m_step`); tests: Lenticulum `test/em.jl`
>
> Bibliography: [[Bibliography#^varadhan2008squarem|Varadhan & Roland 2008]]

## 1. What an M-step needs

At a fixed belief $b$ the free energy is $F(b, \theta) = \sum_f \mathbb E_b[-\log f(\theta_f)] - H(b)$, and its
parameter-dependent part separates over parameter keys. For a factor stated by a `cluster_form`
(a residual $r = \sum_c A_c x_c - C_{:,k}$, noise $Q$, weight $w_k$), $\mathbb E_b[-\log f]$ depends on $b$ only
through the moments of its continuous channels $x$ (stacked) per outcome $k$:

$$
n_k = \sum_a q_a [k(a) = k], \qquad m_k = \sum_a q_a\, \mathbb E[x \mid a]\, [k(a) = k], \qquad S_k = \sum_a q_a\, \mathbb E[x x^\top \mid a]\, [k(a) = k].
$$

`MomentStats` holds them. `_moment_stats` reads them from a `FrozenClusterBelief`: the means
per assignment, the covariance of the assignment's precision class (points have none), and the
weights $q_a$. `em_step` sums them over the tied nodes and the plate's replicas of a parameter key,
then calls `m_step(factor, stats, ps, frozen)`, the factor's minimiser of $\mathbb E_b[-\log f]$, with the
key's frozen paths (a cluster node's paths with the factor's key stripped). A factor without a
method keeps its parameters. A node outside a cluster takes part when its `factor_belief` is a
`FrozenClusterBelief` (Lenticulum's `RegimeFactor`, a one-factor cluster).

With exact beliefs (every learned factor in a cluster node, inference converged) one `em_step` is
one step of EM: the free energy is $-\log p$ of the data, and it does not increase.

## 2. `fit_em` and acceleration

`fit_em(g, ps, st; schedule)` alternates `sweeps` sweeps (the E-step), the free energy and
`em_step`, until $F$ falls by less than `tol`. It ends with an E-step at the returned parameters, so
the store and `Fs[end]` belong to them.

**EM is slow when a latent variable is almost determined by the parameters.** A switch with
noise $10^{-3}$ feeding a sum of noise 0.09 pins its $u$ to the current offset $b_k$, and each M-step moves
$b$ by little. Measured in a scratch run (one data set, 60 points, two switches, the
`test/em.jl` model):

| switch noise | plain EM | SQUAREM |
|---|---|---|
| $10^{-3}$ | 2306 steps to $\Delta F < 10^{-9}$, contraction 0.990 per step, 18.3 s | 86 cycles, 2.3 s, the same $F$ (70.534082) |
| 0.05 | 57 steps, contraction 0.650, 0.5 s | 7 cycles |

`accelerate = true` runs SQUAREM's S3 cycle on the learnable subtrees (θ₁ = M(θ₀), θ₂ = M(θ₁),
θ′ = θ₀ − 2αr + α²v with α = −‖r‖/‖v‖ ≤ −1, one stabilising step from θ′), falling back to θ₂ when the cycle
raises $F$. `test/em.jl` checks that it reaches plain EM's fixed point in under 200 cycles where plain
EM takes over 1000.

## 3. Measured on the structure search

`examples/structure_edits.jl` with `STRUCTURE_FIT=em` (`structure_edits.md`) uses `fit_em` with
SQUAREM in place of 300 Momentum steps. One fit of the generating structure (three switches, 200
points) from a perturbed start, measured in a scratch run after compilation: 1.0 s against 7.7 s,
both reaching $F = 504.6574$. Per call: an E-step (two sweeps and $F$) 0.017 s, `em_step` 0.03 s, a
ForwardDiff gradient 0.01 s. `em_step` builds every replica's belief on its own; the gradient uses
the plate's batched beliefs.

The whole search improved less at first, 177 s against 192–200 s on the same day with the same path
and scores. Fitting was not the bottleneck there: the first fit of every new structure spent 11–22 s
compiling (each `L` is a new parameter type, and the message loop was specialised on it), and the
over-parameterised `L = 4` candidates needed 130–170 SQUAREM cycles to $\Delta F < 10^{-6}$ (30–50 s; their $F$,
504.879, is lower than the gradient fit's). Per-fit timings from an instrumented scratch copy.
The message loop no longer specialises on the parameter tree (`passing.md`, "Compilation per
structure"), and the example stops EM at $\Delta F < 10^{-4}$ (BIC to $10^{-3}$): the search takes **58.0 s**
with EM against 163.8 s with gradient steps, the same path and scores.

**`examples/structure_clusters.jl`** (`structure_clusters.md`): the per-factor M-step built on
`moment_statistics` crawled (the same small-noise switches; 1 143 s, a worse structure), and a
collapsed M-step, the $u$'s integrated out given the assignment, takes the search to 181 s against
1 121 s with gradient steps, on another path into a worse local optimum of the generating
structure. `fit_em(...; mstep)` takes such a caller-supplied M-step.

## 4. What it does not do

- **Batched statistics.** `em_step` computes each replica's belief separately; a batched version
  over `FrozenClusterBeliefs` would cut its cost (0.03 s against the gradient's 0.01 s).
- **Factors outside clusters**, except one-factor clusters, have no statistics: a learned factor
  on approximate (loopy) beliefs is not fitted by EM here.
- **Only closed forms.** Factors without `m_step` keep their parameters; mixing M-steps with
  gradient steps for the others is the caller's (generalised EM).
- **Compilation per structure** dominates a structure search's first fits; that is a property of
  the typed inference code, not of EM (LP-023's "typed execution plans").

## 5. Difficulties

1. **The first test compared one step too many.** `fit_em` runs an M-step after every E-step, so
   `iters = steps + 1` performed `steps + 1` M-steps, and the graph looked a step ahead of the hand
   EM. With the count fixed the iterates agree to $10^{-10}$.
2. **Plain EM did not converge in the switch test** (500 steps, $F$ still falling by 0.002): the
   tiny switch noise, §2. Hence SQUAREM.
3. **The first accelerated fit took 12 s.** The extrapolation rebuilt the parameters from a flat
   vector as views (`ReshapedArray`), a new parameter type that recompiled inference; they are
   materialised as plain arrays now (1.6 s for the first fit, 1.0 s after).
