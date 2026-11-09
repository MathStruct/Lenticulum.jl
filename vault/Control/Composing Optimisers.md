#design

> The optimisers in this repository act on three kinds of state: parameters (`OptimiserFactor`,
> `fit_em` with SQUAREM), messages (a BP sweep, with `damping`) and, not yet, beliefs. Most proposals
> for faster or safer inference are **modifiers** of one of these maps: damping, momentum, Anderson
> or SQUAREM extrapolation, a mirror map, noise. This note asks how such pieces compose.
>
> Two results. **Fixed points compose; convergence does not.** Each modifier keeps the base map's
> fixed points, so any stack of them still satisfies condition (O) of
> [[The Regulariser and the Loss]]. But whether the stack converges, and how fast, depends on the
> spectrum of the linearised map, and two convergent pieces can compose into a divergent one.
> The spectrum also says which modifier to use: momentum where the slow eigenvalues are real and
> near 1, damping or Anderson where they are complex or near −1. That makes a **spectral
> diagnostic** the first tool to build, and a benchmark the way to decide the rest.
>
> Measured on the divider graph (§4.1, `bench/bp_spectrum.jl`): every slow eigenvalue of a flooding
> sweep is complex, on a ring. Heavy ball diverges there for every β ≥ 0.07, damping gains at most
> 2% in rate, and the gain is in the **schedule**: one that goes round the loop in a sweep converges
> at ρ⁶ instead of ρ per sweep (38 sweeps instead of 326). A fixed order that goes once round the
> loop each way, or one built from a spanning tree with no knowledge of the loop, does that at
> flooding's cost per sweep: about six times fewer message evaluations in every setting tried
> (§4.2, `bench/bp_schedules.jl`; now `spanning_tree_schedule` in Mycelium). Anderson acceleration
> composes well with that order and badly with flooding: on top of the spanning-tree order it saves
> up to half again, and where the loop gain is near 1 it turns 297 000 evaluations into 414; on top
> of flooding it costs 1.4 to 3.8 times more in the ordinary cases, or fails (§4.3,
> `bench/bp_anderson.jl`).

> Sources: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE TIT 51(7), 2005 (BP fixed points are the stationary points of the Bethe free energy); Polyak, *Some Methods of Speeding Up the Convergence of Iteration Methods*, 1964 (heavy ball); Walker & Ni, *Anderson Acceleration for Fixed-Point Iterations*, SIAM J. Numer. Anal. 49(4), 2011; Varadhan & Roland, SQUAREM, 2008; Raskutti & Mukherjee, *The Information Geometry of Mirror Descent*, IEEE TIT 61(3), 2015; Krichene, Bayen & Bartlett, *Accelerated Mirror Descent in Continuous and Discrete Time*, NeurIPS 2015; Wibisono, Wilson & Jordan, *A Variational Perspective on Accelerated Methods in Optimization*, PNAS 113(47), 2016; Andrychowicz et al., *Learning to Learn by Gradient Descent by Gradient Descent*, NeurIPS 2016; Cruttwell, Gavranović, Ghani, Wilson & Zanasi, *Categorical Foundations of Gradient-Based Learning*, ESOP 2022 (Def. 3.14, update lenses). The composition laws and the spectral reading are original to this vault; the table in §4 is a closed-form check, §4.1 is measured by `bench/bp_spectrum.jl` and §4.2 by `bench/bp_schedules.jl`, §4.3 by `bench/bp_anderson.jl`; Elidan, McGraw & Koller, *Residual Belief Propagation*, UAI 2006 (the priority queue of §4.2); Wainwright, Jaakkola & Willsky, *Tree-Based Reparameterization Framework for Analysis of Sum-Product and Related Algorithms*, IEEE TIT 49(5), 2003 (updates along spanning trees). code: `lib/Mycelium.jl/src/anderson.jl` (`AndersonSchedule`), `bench/bp_spectrum.jl`, `bench/bp_schedules.jl`, `bench/bp_anderson.jl`, `bench/divider.jl`, `lib/Mycelium.jl/src/schedules.jl` (`spanning_tree_schedule`), `lib/Mycelium.jl/src/passing.jl` (`sweep!`, `damping`), `lib/Mycelium.jl/src/schedules.jl` (`AbstractSchedule`), `lib/Mycelium.jl/src/factors.jl` (`Momentum`, `Nesterov`), `lib/Mycelium.jl/src/em.jl` (SQUAREM)
>
> Bibliography: [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^polyak1964heavyball|Polyak 1964]] · [[Bibliography#^walker2011anderson|Walker & Ni 2011]] · [[Bibliography#^varadhan2008squarem|Varadhan & Roland 2008]] · [[Bibliography#^raskutti2015mirror|Raskutti & Mukherjee 2015]] · [[Bibliography#^krichene2015accelerated|Krichene et al. 2015]] · [[Bibliography#^wibisono2016variational|Wibisono et al. 2016]] · [[Bibliography#^andrychowicz2016learning|Andrychowicz et al. 2016]] · [[Bibliography#^cruttwell2022gradient|Cruttwell et al. 2022]] · [[Bibliography#^elidan2006residual|Elidan et al. 2006]] · [[Bibliography#^wainwright2003trp|Wainwright et al. 2003]]
>
> Theory (CT-ML wiki): [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## 1. Three maps to accelerate

| state | base map $T$ | its fixed points | exists |
|---|---|---|---|
| parameters $\theta$ | a gradient step $\theta - \eta\nabla_\theta F$, or an M-step | $\nabla_\theta F = 0$ (for EM: stationary points of the likelihood) | `OptimiserFactor`, `fit_em` |
| messages $m$ | one BP sweep (flooding, sequential, residual) | stationary points of the Bethe free energy (Yedidia et al. 2005) | `sweep!`, `infer!` |
| beliefs $b$ | a step on $\nabla_b F$ in some geometry | stationary points of $F$ on the constraint set | missing (no $\nabla_b F$, no retraction: [[Feature Backlog]], the Langevin row) |

The second row is the one that matters most and is easiest to get wrong. A BP sweep **is not a
gradient step** on $F$. It is a fixed-point map whose fixed points happen to be $F$'s stationary
points. So BP satisfies (O), but nothing guarantees it descends $F$ on the way, and modifiers built
for gradient descent (momentum, mirror maps) carry over only as fixed-point modifiers, with
none of their descent guarantees.

## 2. The pieces: modifiers of a map

Each piece takes a map $T$ (with or without state) and returns another. Write $r(x) = T(x) - x$
for the residual.

| modifier | the new map | state | fixed points |
|---|---|---|---|
| **relaxation** (damping) | $x' = x + \alpha\, r(x)$, $0 < \alpha \le 1$ | none | $\mathrm{Fix}(T)$, exactly |
| **heavy ball** (Polyak 1964) | $v' = \beta v + \alpha\, r(x)$, $x' = x + v'$ | $v$ | $\mathrm{Fix}(T)$ with $v = 0$, exactly |
| **Anderson** (Walker & Ni 2011) | $x' = \sum_i \gamma_i T(x_{k-i})$, $\sum_i \gamma_i = 1$, $\gamma$ minimising $\lVert \sum_i \gamma_i r(x_{k-i}) \rVert$ | the last $m$ iterates | contains $\mathrm{Fix}(T)$; it can also stall at a non-fixed point, so convergence is judged by $\lVert r \rVert$ |
| **SQUAREM** (Varadhan & Roland 2008) | a step along $r$ and $T(T(x)) - 2T(x) + x$, from one $T(T(x))$ | none | contains $\mathrm{Fix}(T)$ (with a guard for $r = 0$) |
| **mirror map** $\psi$ (for a gradient map) | $\nabla\psi(x') = \nabla\psi(x) - \eta\nabla F(x)$ | none | $\nabla F = 0$ (on a constraint set, its KKT points) |
| **acceleration** of a mirror step (Krichene et al. 2015) | a mirror step on a coupled pair of sequences | a second sequence | as for the mirror map |
| **temperature** (Langevin) | add noise $\sqrt{2\gamma T}\,\xi$ to a momentum step | $v$, the noise | none for $T > 0$: an invariant law $\propto e^{-F/T}$, on the global minimisers as $T \to 0$ |
| **learned** $U_\psi$ (Andrychowicz et al. 2016) | anything | anything | not guaranteed: (O) must be checked or built in |

The heavy-ball row in one line: a fixed point needs $x' = x$, so $v' = 0$; then $v' = v$ gives
$v = 0$, and $\alpha\, r(x) = 0$. The same argument as for `Momentum` in
[[The Regulariser and the Loss]] §3, applied to a residual in place of a gradient.

The mirror rows apply only where $T$ is a gradient step, so for BP they need the belief-row map
of §1 first. Their use is geometric. With the log-partition as $\psi$, mirror descent in the mean
parameters is natural-gradient descent in the natural ones (Raskutti & Mukherjee 2015), so every
iterate is a valid belief. Leaving that set is what broke the scratch heavy-ball test
([[Feature Backlog]], the Langevin row: a negative precision, then NaN).

## 3. Three ways to compose, and what each keeps

**Wrapping** (a modifier around a map, or a stack of them). Every modifier above except
temperature and a learned rule keeps $\mathrm{Fix}(T)$, so a stack of them does too. Damping inside
Anderson inside a schedule still satisfies (O). That is the good news, and it holds whatever the
order.

**Alternation on separate blocks** (inference sweeps and parameter steps, as in EM or the
proposed `train!`). If $T_b$ updates the beliefs given $\theta$ and $T_\theta$ the parameters given $b$,
the alternation $(b, \theta) \mapsto \bigl(T_b(b,\theta),\, T_\theta(T_b(b,\theta), \theta)\bigr)$ is at rest exactly when
$b = T_b(b,\theta)$ and $\theta = T_\theta(b,\theta)$: the joint fixed points. When both blocks descend the same
$F$ (condition (I)), those are its stationary points. When they descend different objectives, the
fixed points are a game's equilibria instead ([[The Two-Part Diagram]] §3).

**Nesting** (an inner solve to a tolerance inside each outer step). The outer fixed points are
stationary only up to the inner tolerance, and the outer gradient is either unrolled through the
inner iterations or taken at the inner fixed point by the implicit function theorem, which
Mycelium already does (`adjoint = :implicit` in `free_energy.jl`). Inner tolerance becomes a
hyperparameter of the composite.

**What does not compose: convergence.** Two maps that each converge can compose into one that
diverges. For linear maps the rate is the spectral radius, and it is not submultiplicative:

$$
J_1 = \begin{pmatrix} 0 & 2 \\ 0 & 0 \end{pmatrix},\quad
J_2 = \begin{pmatrix} 0 & 0 \\ 2 & 0 \end{pmatrix},\qquad
\rho(J_1) = \rho(J_2) = 0,\quad \rho(J_2 J_1) = 4 .
$$

Each converges in two steps; alternating them diverges. There is one positive law: if $T_1$ and $T_2$
are contractions **in the same norm** with factors $c_1, c_2$ and share a fixed point, their composite
contracts with $c_1 c_2$. Loopy BP is not a contraction in general, which is why no stack of
modifiers on it comes with a guarantee for free.

## 4. The spectrum decides which modifier helps

Near a fixed point every map is its linearisation, $e' = J e$. For a Gaussian graph the mean
update is exactly linear once the precisions have converged, so $J$ can be computed. Then each
modifier acts on each eigenvalue $\lambda$ of $J$ separately:

- **relaxation** maps $\lambda \mapsto 1 - \alpha + \alpha\lambda$: it pulls everything towards 1, so it cures
  eigenvalues near $-1$ (oscillation) and complex ones, and slows those near $+1$;
- **heavy ball** (with $\alpha = 1$) turns each $\lambda$ into the two roots of $\mu^2 - (\lambda + \beta)\mu + \beta = 0$.
  For **real** $\lambda$ it is stable exactly when $-1 - 2\beta < \lambda < 1$, so it never destabilises a convergent
  real spectrum. At $\lambda = 0.99$ the best $\beta = 0.81$ gives $|\mu| = 0.9$, about 10.5 times fewer iterations
  than 0.99. At $\lambda = -0.99$, $\beta = 0.3$ gives $|\mu| = 0.55$;
- but for **complex** $\lambda$ momentum can destabilise a convergent map:

| $\lambda$ | $\lvert\lambda\rvert$ | $\beta = 0$ | $0.3$ | $0.6$ | $0.9$ |
|---|---|---|---|---|---|
| $0.95i$ | 0.95 | 0.95 | 1.22 | 1.43 | 1.61 |
| $-0.5 + 0.8i$ | 0.94 | 0.94 | 1.09 | 1.27 | 1.44 |
| $0.3 + 0.9i$ | 0.95 | 0.95 | 1.23 | 1.44 | 1.63 |

(spectral radius of the heavy-ball iteration; closed-form roots, computed in scratch.)

A BP sweep's Jacobian is not symmetric, so complex eigenvalues are possible. Gradient descent on a
smooth $F$ has a symmetric Jacobian and real eigenvalues, which is the setting momentum was designed
for. On the divider graph a sweep's slow eigenvalues are all complex (§4.1), and the linear analysis
alone predicts the momentum failure.

**Anderson** needs no such case split. It fits the best combination of the last $m$ residuals, so on
a linear map it is a Krylov method (GMRES-like) and handles complex spectra. That makes it the
natural first accelerator for BP sweeps, and the cheapest to try: it wraps `sweep!` through
`AbstractSchedule` and needs none of the belief-geometry additions.

### 4.1 Measured: the divider (experiment 1)

`bench/bp_spectrum.jl` linearises one flooding sweep of the resistive divider (`test/runtests.jl`:
three variables, three constraints, one loop of six edges, weak priors of variance $10^4$) at its
fixed point, by central differences over every message's $(\eta, \Lambda)$. The precisions do not depend
on the means ($\partial\Lambda'/\partial\eta = 0$ exactly), so the spectrum is the mean block's and the precision
block's together.

| $R_1, R_2$ | mean eigenvalues: $\lvert\lambda\rvert$ | precision eigenvalues: $\lvert\lambda\rvert$ | $\rho^6$ | flooding sweeps to an error $< 10^{-10}$ |
|---|---|---|---|---|
| 2, 3 (the test's) | 0.9347 | 0.8736 | 0.6667 | 326 |
| 1, 3 | 0.8327 | 0.6934 | 0.3333 | 122 |
| 3, 2 | 0.9347 | 0.8736 | 0.6667 | 322 |
| 2, 2 | 0.99984 | 0.99968 | 0.99905 | 97 224 (§4.2) |

(12 nonzero eigenvalues in each block, all of the same modulus to $10^{-10}$; the error is the
largest of the mean and precision errors, each relative to its block's largest entry.)

- **All the slow eigenvalues are complex, on a ring.** The mean eigenvalues sit at $\pm 30°$, $\pm 90°$ and
  $\pm 150°$: the sixth roots of $-\rho^6$. The precision eigenvalues sit at $0°$, $\pm 60°$, $\pm 120°$ and $180°$, the sixth
  roots of $+\rho_\Lambda^6$, with $\rho_\Lambda = \rho^2$ in every case. So one trip round the six-edge loop takes six flooding sweeps,
  and a mean comes back multiplied by $-\rho^6$. In the three unequal cases $\rho^6 = R_{\min}/R_{\max}$; with equal
  resistors it is $1 - 9.5\cdot 10^{-4}$. That is an observation on four settings, not derived. The
  test's divider is a benign case of a graph that flooding solves badly.
- **The prediction is exact.** Undamped, the measured asymptotic rate is 0.9346, 0.8324 and 0.9347,
  against $\rho$ = 0.9347, 0.8327 and 0.9347.
- **Momentum is ruled out on this graph, not merely untuned.** Heavy ball's spectral radius on the
  test's divider is 0.9864 at $\beta = 0.05$, 0.9964 at 0.06, 1.0062 at 0.07 and 1.2043 at 0.3. With
  equal resistors it diverges for every $\beta \ge 0.05$. The scratch failure for every $\beta \ge 0.3$
  ([[Feature Backlog]], the Langevin row) is what the linear analysis predicts, before any precision
  goes negative.
- **Damping can barely help.** Relaxation pulls a ring towards 1 and shrinks some eigenvalues at the
  expense of others: the best uniform $\alpha = 0.75$ gives 0.9260 against 0.9347. Mycelium's `damping`
  relaxes only the factor-to-variable messages (`passing.jl`, `_step!`), so its Jacobian is
  $DJ + (I - D)$ with $D = 1 - \text{damping}$ on those messages and 1 elsewhere, not $1 - \alpha + \alpha\lambda$ on every eigenvalue.
  That prediction matches the measured rate at damping 0.5 (0.9167 and 0.9164) and 0.8 (0.9591 and
  0.9591). At 0.25, and at 0.5 for $R = 1, 3$, the runs converge faster than predicted, as a start
  that barely excites the leading modes would (not checked). And **the rate is not the sweep count**:
  damping 0.5 has the better rate (0.9167 against 0.9347) but needs 410 sweeps against 326, because
  the transients differ.
- **The schedule is the modifier that works.** A residual schedule (the largest change first)
  converges at exactly $\rho^6$ per sweep, a trip round the loop in each sweep: 38 sweeps instead of 326,
  15 instead of 122. Its sweeps cost more, though: it evaluates every task's candidate before each
  commit, $18^2$ evaluations against flooding's 18, so it needs about twice flooding's message
  evaluations (12 312 against 5 868). Its order, replayed as a plain sequential sweep (with the tasks
  it skipped appended), keeps the rate at flooding's cost per sweep for $R = 1, 3$ (462 evaluations
  against 2 196), partly for $2, 3$ (1 620 against 5 868), and not for $3, 2$. A sequential sweep in
  flooding's task order gets $\rho^2$ per sweep (two hops), at half flooding's evaluations.

So on this graph the slow modes are a ring, momentum diverges, damping gains a few percent, and the
gain is in the order. The expectation was $\rho^6$ per sweep at flooding's cost for a schedule that
follows the loop; the replays above reach it on one of three settings, and §4.2 tests it directly.
Anderson stays the open candidate: the mean block has six distinct nonzero eigenvalues, so on the
linear part a short memory should already cancel them (an expectation, experiment 3).

### 4.2 Measured: the order of a sweep (experiment 2)

`bench/bp_schedules.jl` runs the same 18 messages of the divider six ways and counts **message
evaluations** (calls of the update rule, committed or not) until the error against the fixed point
falls below $10^{-10}$ (the error of §4.1, from a cold start):

- *flooding*, double-buffered, 18 evaluations per sweep;
- *sequential*, in place, in `all_tasks` order (edge by edge);
- *loop*: the messages off the loop, then once round the loop each way (the cycle found by a
  depth-first search);
- *spanning*: the messages off a spanning tree, then the tree's inward and outward sweeps
  (`_tree_sweeps` from a variable; it needs no knowledge of the loop);
- *residual*: `ResidualSchedule`, every candidate evaluated before each commit;
- *queue*: residual BP with a priority queue (Elidan, McGraw & Koller, 2006): after a commit, only
  the messages that read the committed one are evaluated again (the dependencies of belief
  propagation; mean-field and EP nodes read more and are not covered).

Evaluations to an error $< 10^{-10}$, and as a fraction of flooding's:

| $R_1, R_2$ | flooding | sequential | loop | spanning | residual | queue |
|---|---|---|---|---|---|---|
| 2, 3 | 5 868 | 2 970 (0.51) | 1 026 (0.17) | 1 008 (0.17) | 12 312 (2.10) | 1 418 (0.24) |
| 1, 3 | 2 196 | 1 134 (0.52) | 414 (0.19) | 396 (0.18) | 4 860 (2.21) | 562 (0.26) |
| 3, 2 | 5 796 | 2 862 (0.49) | 972 (0.17) | 972 (0.17) | 12 312 (2.12) | 1 383 (0.24) |
| 2, 2 | 1 750 032 | 875 016 (0.50) | 291 672 (0.17) | 297 144 (0.17) | not run | 538 712 (0.31) |

- **A fixed order round the loop is the whole gain.** *Loop* and *spanning* converge at $
ho^6$ per
  sweep (measured 0.6666 and 0.6667 for $R = 2, 3$; 0.3335 and 0.3326 for $1, 3$), the residual
  schedule's rate, at flooding's 18 evaluations per sweep: about six times fewer evaluations than
  flooding, in all four settings. The sequential sweep in edge order gets $
ho^2$ (0.8736), half.
- **The spanning-tree order needs no knowledge of the loop** and matches the loop order to within 3%.
  It is the schedule to try first on graphs with several loops: every off-tree edge's message is
  refreshed, then information crosses the whole tree in one sweep.
- **The priority queue removes the residual schedule's quadratic cost** (from 2.1 to 0.24 times
  flooding's evaluations) but stays behind the static orders, at 0.24 to 0.31: its commits follow
  the loop like the residual schedule's (690 commits, 38 sweeps' worth, for $R = 2, 3$), and each
  commit costs about one more evaluation for its dependents. On this graph a fixed order is cheaper.
  Where the slow part of a graph is not known in advance, or moves, the queue is the adaptive
  option; that is untested.
- **With equal resistors flooding needs 97 224 sweeps**, not the 145 000 that $
ho$ alone gives from an
  error of 1: the cold start puts little into the slowest modes (as for damping 0.25 in §4.1).
  The run's rates over its last six sweeps are not asymptotic and are not reported; the fixed point
  is the loop order's run to its rounding floor. With $1 - \rho^6 \approx 10^{-3}$ a residual at the floor
  could leave an error a thousand times larger; checked in §4.3, the same procedure's reference is
  within $7.5\cdot 10^{-13}$ of a fixed point with residual $4\cdot 10^{-17}$, so the criterion of $10^{-10}$ holds.

So the cheapest acceleration of BP on this graph is a change of schedule, no modifier at all: a
spanning-tree order, a sixfold saving that keeps the fixed points (§3: a schedule changes the map,
not its fixed points). Whether it holds on graphs with several loops, where no one tree carries
every loop's information in one sweep, is the next measurement (a Gaussian grid, in the benchmark).
The order is now a library schedule, `spanning_tree_schedule(g)` (Mycelium, `schedules.md`); with
`infer!` on the test's divider it converges in 82 sweeps against flooding's 463, to the same
marginals (`test/runtests.jl`).

### 4.3 Measured: Anderson around a sweep (experiment 3)

`bench/bp_anderson.jl` wraps a sweep $G$ (flooding or the spanning-tree order) in Anderson
acceleration with memory $m$ (Walker & Ni, 2011, type II, undamped):
$x^+ = G(x_k) - \Delta G\,\gamma$ with $\gamma$ the least-squares fit of the newest residual
$f_k = G(x_k) - x_k$ by the differences of the last $m$. It keeps $G$'s fixed points. The least
squares are taken in coordinates scaled per block (precisions are about $10^3$), with a ridge of
$10^{-14}$ of the trace; a step that would make a precision non-positive is replaced by the plain
sweep and the memory cleared (a *restart*). Cost: 18 message evaluations per iteration, as before.
The reference fixed point, the plain spanning-tree sweep run to its rounding floor, is within
$10^{-12}$ of Anderson(3) after 300 iterations in every setting.

Message evaluations to an error $< 10^{-10}$, as a fraction of plain flooding's:

| $R_1, R_2$ | base | $m = 0$ | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|---|
| 2, 3 | flooding | 1 (5 868) | 1.46 | 1.79 | 3.19 | fails | fails |
| | spanning | 0.17 | 0.19 | **0.13** | 0.13 | 0.13 | 0.15 |
| 1, 3 | flooding | 1 (2 196) | 1.46 | 1.95 | 2.23 | 2.38 | 3.79 |
| | spanning | **0.18** | 0.20 | 0.19 | 0.19 | 0.19 | 0.22 |
| 3, 2 | flooding | 1 (5 796) | 1.44 | 1.93 | 3.39 | fails | 152.9 |
| | spanning | 0.17 | 0.11 | **0.09** | 0.09 | 0.10 | 0.12 |
| 2, 2 | flooding | 1 (1 749 924) | 1.21 | 1.56 | 0.28 | 0.02 | 0.0019 (3 312) |
| | spanning | 0.17 (297 108) | 4.5e-4 | 3.2e-4 | **2.4e-4 (414)** | 3.4e-4 | 4.0e-4 |

("fails": not below $10^{-10}$ within 400 000 iterations, restarting tens of thousands of times.)

- **Anderson composes with the order, not with flooding.** Around the spanning-tree order it loses
  at most a fifth ($R = 1, 3$, $m = 10$: 0.22 against 0.18) and gains up to half ($R = 3, 2$: 0.09 against 0.17). Around flooding it
  loses in all three ordinary settings, by 1.4 to 3.8 times, and with $m = 5$ or 10 it can fail outright.
  Two pieces that each converge compose into one that does not: §3's warning, measured.
- **Where the loop gain is near 1, Anderson is the cure.** With equal resistors ($\rho^6 = 0.99905$) the
  spanning-tree order needs 297 108 evaluations and Anderson(3) around it 414, about 700 times fewer
  (and 4 200 times fewer than flooding). Around flooding it needs a longer memory: $m = 10$ gives 3 312.
  That fits §4.1's expectation (a short memory cancels a few distinct eigenvalues; flooding's ring
  has six per block, a spanning sweep's linear part is expected to have fewer), which is not checked
  here: the spectrum of the spanning-tree sweep is not computed.
- **Why Anderson hurts flooding in the ordinary cases is not known.** It is not the positivity guard:
  with $m = 1$ there are no restarts for $R = 2, 3$ and $3, 2$, and it still costs 1.46 and 1.44 times
  flooding. A candidate is
  the precision block, which is nonlinear and converges at $\rho^2$ from far away; untested.

So the composition to use on this graph is **the spanning-tree order, with Anderson(2–3) on top when
convergence is slow**; flooding should not be accelerated by Anderson at all. That composition is
now in the library: `AndersonSchedule(spanning_tree_schedule(g); m = 3)` (Mycelium, `anderson.md`),
which by `infer!` converges in 51, 25, 34 and 24 sweeps where the plain spanning-tree order needs 82,
34, 80 and 20 881, and flooding 463, 187, 463 and 116 509 (`bench/bp_anderson.jl`, last table; the
test's divider in `test/runtests.jl`). Open: the spectrum of the spanning-tree sweep, and a graph with
several loops.

## 5. The design space, and why it needs a benchmark

The choices multiply: the state (θ, m, b) × the schedule × the stack of modifiers × their
hyperparameters (α, β, $m$, temperature) × nesting tolerances. Each claim so far was tried on one
graph. Three consequences:

1. **A spectral diagnostic first.** Report the leading eigenvalues of the linearised sweep at a
   fixed point (exact for Gaussian means; by finite differences or AD of `sweep!` otherwise). It
   says before any tuning whether momentum, damping or Anderson should help, and it can be
   checked against the observed iteration counts.
2. **A benchmark to compare stacks** ([[Feature Backlog]], Tooling): problems with reference
   answers where today's methods are slow or fail, run by one script recording sweeps, wall time,
   error, the free energy reached and failures.
3. **A learned optimiser last.** A learned $U_\psi$ trained on the benchmark can overfit it (its outer
   criterion is the benchmark), and must keep (O): the simplest way is to learn only the
   hyperparameters of a stack whose fixed points are fixed by construction (§3), not a free map.

A proposed order of experiments, cheapest first:

| | experiment | needs | question |
|---|---|---|---|
| 1 | spectrum of the linearised flooding sweep on the divider | finite differences of `sweep!` | **done** (§4.1): all complex, on a ring; heavy ball diverges for $\beta \ge 0.07$; the schedule is what helps |
| 2 | a loop-following sequential order, and the residual schedule with a priority queue | the cycle basis of the graph; LP-023's queue | **done on the divider** (§4.2): loop and spanning-tree orders need 0.17–0.19 of flooding's evaluations, the queue 0.24–0.31; graphs with several loops remain |
| 3 | Anderson ($m = 3, 5, 10$) around flooding and around experiment 2's order | a schedule type wrapping `sweep!` | **done on the divider** (§4.3): around the spanning-tree order 0.09–0.19 of flooding, and 414 evaluations instead of 297 000 with equal resistors; around flooding worse, or failing |
| 4 | SQUAREM and Anderson on `fit_em` for the switch learning | `em.jl` | does Anderson beat SQUAREM on the parameter block? |
| 5 | mirror and accelerated mirror descent on beliefs | $\nabla_b F$, a mirror map, primal beliefs | do valid-by-construction iterates fix the momentum failure? |
| 6 | Langevin with annealed temperature | experiment 5's interface | does it leave poor fixed points on multimodal graphs? |

## 6. Open questions

1. **What is a composite's (O) when the pieces descend different objectives?** Damping a sweep
   that minimises the Bethe $F$ and an M-step that maximises a likelihood are both fine alone; the
   alternation's fixed points are the joint stationary points only if the two objectives are one
   $F$ (§3).
2. **Is there a norm in which BP contracts on a given graph?** Where there is, composition laws
   become rates (§3); the walk-summable Gaussian graphs are the candidates.
3. **Do accelerations compose in continuous time?** Wibisono et al. (2016) derive accelerated
   methods as discretisations of one Lagrangian flow. If the modifiers above are discretisations of
   flows, composing them is adding terms to the flow, and the question becomes which
   discretisation keeps the rate.
4. **Optimiser composition as lens composition.** An update rule is a lens (Cruttwell et al.,
   Def. 3.14), and wrapping is a map from lenses to lenses that keeps the fixed points. Whether the
   modifiers of §2 form a category acting on update lenses, with (O) as a property it preserves, is
   unworked.

Related: [[The Regulariser and the Loss]], [[The Inferencer and the Optimizer]],
[[The Two-Part Diagram]], [[Schedules]], [[Loopy Message Passing]], [[Feature Backlog]]
