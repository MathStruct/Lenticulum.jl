#implementation

> `AndersonSchedule(inner; m = 3)` (`anderson.jl`): Anderson acceleration of any static schedule.
> A sweep of `inner` is a map $G$ on the stored messages; the accelerated sweep replaces $G(x_k)$ by
> the combination of the last $m + 1$ images that best cancels their residuals $G(x) - x$. Same
> fixed points as `inner`, one sweep of `inner` per sweep, no convergence guarantee.

> Sources: Walker & Ni, *Anderson Acceleration for Fixed-Point Iterations*, SIAM J. Numer. Anal.
>
> Bibliography: [[Bibliography#^walker2011anderson|Walker & Ni 2011]]
> 49(4), 2011 (the update used here, and its relation to GMRES on linear problems). code: `anderson.jl`, `bench/bp_anderson.jl`;
> vault: Composing Optimisers §4.3.

## Theory

Write the stored messages as one vector $x$ and one sweep as $x \mapsto G(x)$, with residual
$f = G(x) - x$. With the last $m + 1$ iterates, let $\Delta F$ and $\Delta G$ hold the differences of
consecutive residuals and images. Anderson (type II, undamped) steps to

$$
x^+ = G(x_k) - \Delta G\,\gamma, \qquad \gamma = \arg\min_\gamma \lVert f_k - \Delta F\,\gamma\rVert_2 .
$$

At a fixed point $f_k = 0$, so $\gamma = 0$ and $x^+ = G(x_k) = x_k$: the fixed points are $G$'s. On a
linear map with unlimited memory it is essentially GMRES on $(I - J)x = b$ (Walker & Ni), which is
why a short memory can cancel a Jacobian with few distinct eigenvalues; with a finite memory and off
the linear regime nothing is guaranteed.

## Implementation

- **What is mixed.** Every stored Gaussian message in host `Float64` arrays, in natural parameters:
  $\eta$ and the lower triangle of $\Lambda$. If any stored message is of another kind (categorical,
  particles, a mixture, a device array, dual numbers) the sweep is `inner`'s unchanged. Messages not
  yet sent are left out; when the set of stored messages changes (a cold start, a message that
  appears) the memory is cleared.
- **Scaling.** The least squares are taken in coordinates scaled by the largest $\lvert\eta\rvert$ and the
  largest $\lvert\Lambda\rvert$, fixed when the memory starts, so that the precisions (about $10^3$ on the
  divider, the means about 1) do not decide the fit alone. Unscaled mixing was not measured.
- **Ridge.** $\gamma$ solves $(\Delta F^\top \Delta F + r\,\mathrm{tr}(\Delta F^\top \Delta F)\,I)\gamma = \Delta F^\top f_k$ with
  $r = 10^{-14}$ by default: near convergence the columns of $\Delta F$ are nearly dependent.
- **Restarts.** A step that is not finite, or would make a precision lose positive
  semidefiniteness (its smallest eigenvalue below $\min(0, \lambda_{\min}$ of $G$'s$)$ minus a rounding
  margin), is replaced by $G(x_k)$ and the memory cleared. `anderson_restarts(sched, store)` counts
  them. The improper flag of each message is kept.
- **Memory per store.** The history lives in a `WeakKeyDict` keyed by the `MessageStore`, so the
  schedule is reusable across stores (each `infer!` starts afresh) and a dropped store frees it.
- **Residual.** `sweep!` returns `inner`'s residual, the distance from the stored messages to their
  images, which is the fixed-point residual $\lVert G(x) - x\rVert$ in `belief_distance`. So `propagate!`'s
  `tol` means what it means for `inner`, and the default criterion is `inner`'s.
- `ResidualSchedule` is refused: its sweep depends on the order of residuals and is not a fixed map.

## Measured

On the resistive divider (`bench/bp_anderson.jl`, by `infer!` with `tol = 1e-11`):

| $R_1, R_2$ | flooding | spanning tree | + Anderson(2) | + Anderson(3) | flooding + Anderson(3) |
|---|---|---|---|---|---|
| 2, 3 | 463 | 82 | 51 | 51 | 1 168 |
| 1, 3 | 187 | 34 | 25 | 25 | 307 |
| 3, 2 | 463 | 80 | 32 | 34 | 920 |
| 2, 2 | 116 509 | 20 881 | 31 | 24 | > 20 000 |

On Gaussian grids up to 20 × 20 (`bench/bp_grid.jl`), Anderson(3) around the spanning-tree order
needs 67–296 sweeps where the plain order needs 894–1 425 ($p_v = 100$), and 124 where it does not
converge in 20 000 ($p_v = 10^4$). Why a short memory suffices: the spanning-tree sweep factors
through its off-tree messages, so its Jacobian has at most two nonzero eigenvalues per independent
loop (vault: Composing Optimisers §4.4).

The means agree with the plain spanning-tree order's to $10^{-14}$ (to $8\cdot 10^{-12}$ for $2, 2$, where the
plain order's stopping point is the less accurate one). The bench's own implementation, with a
reference fixed point, gives the same picture in message evaluations (vault: Composing Optimisers
§4.3).

## What it does not do

- **Wrap flooding reliably.** Around flooding it was slower in every divider setting above, and with
  $m \ge 5$ it can fail (§4.3); on Gaussian grids it speeds flooding up (0.08–0.73 of its sweeps,
  `bench/bp_grid.jl`, §4.4). Around `spanning_tree_schedule` it helped everywhere measured, with no
  restarts on the grids. Why the divider differs is not known (its flooding spectrum is a ring of
  equal moduli).
- **Gradients.** Schedule replay and the implicit adjoints do not know about the mixing; parameter
  gradients through an `AndersonSchedule` are not supported.
- **Non-Gaussian messages**, plates batched on a device, and damping of the mixed step (Walker &
  Ni's $\beta$): not implemented. `damping` is passed to `inner`.
- **Safeguards beyond positivity.** No residual-decrease check (safeguarded Anderson); a step that
  increases the residual is kept.

## Difficulties

The bench scales by the reference fixed point's block maxima, which a library does not have; the
scales are taken from the first image $G(x)$ and frozen, because changing them between iterations
would make the stored differences inconsistent. The library and the bench therefore mix in slightly
different coordinates; both give the picture above.
