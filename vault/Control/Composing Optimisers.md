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

> Sources: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE TIT 51(7), 2005 (BP fixed points are the stationary points of the Bethe free energy); Polyak, *Some Methods of Speeding Up the Convergence of Iteration Methods*, 1964 (heavy ball); Walker & Ni, *Anderson Acceleration for Fixed-Point Iterations*, SIAM J. Numer. Anal. 49(4), 2011; Varadhan & Roland, SQUAREM, 2008; Raskutti & Mukherjee, *The Information Geometry of Mirror Descent*, IEEE TIT 61(3), 2015; Krichene, Bayen & Bartlett, *Accelerated Mirror Descent in Continuous and Discrete Time*, NeurIPS 2015; Wibisono, Wilson & Jordan, *A Variational Perspective on Accelerated Methods in Optimization*, PNAS 113(47), 2016; Andrychowicz et al., *Learning to Learn by Gradient Descent by Gradient Descent*, NeurIPS 2016; Cruttwell, Gavranović, Ghani, Wilson & Zanasi, *Categorical Foundations of Gradient-Based Learning*, ESOP 2022 (Def. 3.14, update lenses). The composition laws and the spectral reading are original to this vault; the spectral numbers in §4 are closed-form checks, not measurements on a graph. code: `lib/Mycelium.jl/src/passing.jl` (`sweep!`, `damping`), `lib/Mycelium.jl/src/schedules.jl` (`AbstractSchedule`), `lib/Mycelium.jl/src/factors.jl` (`Momentum`, `Nesterov`), `lib/Mycelium.jl/src/em.jl` (SQUAREM)
>
> Bibliography: [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^polyak1964heavyball|Polyak 1964]] · [[Bibliography#^walker2011anderson|Walker & Ni 2011]] · [[Bibliography#^varadhan2008squarem|Varadhan & Roland 2008]] · [[Bibliography#^raskutti2015mirror|Raskutti & Mukherjee 2015]] · [[Bibliography#^krichene2015accelerated|Krichene et al. 2015]] · [[Bibliography#^wibisono2016variational|Wibisono et al. 2016]] · [[Bibliography#^andrychowicz2016learning|Andrychowicz et al. 2016]] · [[Bibliography#^cruttwell2022gradient|Cruttwell et al. 2022]]
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
for. That is a plausible reading of why momentum works for parameters and failed on messages,
**not checked**: the scratch failure left the domain (a negative precision) before any linear
analysis applied. Measuring the spectrum on the divider graph would settle it.

**Anderson** needs no such case split. It fits the best combination of the last $m$ residuals, so on
a linear map it is a Krylov method (GMRES-like) and handles complex spectra. That makes it the
natural first accelerator for BP sweeps, and the cheapest to try: it wraps `sweep!` through
`AbstractSchedule` and needs none of the belief-geometry additions.

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
| 1 | spectrum of the linearised flooding sweep on the divider (433 sweeps today) | AD or finite differences of `sweep!` | are its slow eigenvalues real or complex? |
| 2 | Anderson ($m = 3, 5, 10$) around flooding, with and without damping | a schedule type wrapping `sweep!` | sweeps to tolerance vs flooding and damped flooding |
| 3 | SQUAREM and Anderson on `fit_em` for the switch learning | `em.jl` | does Anderson beat SQUAREM on the parameter block? |
| 4 | mirror and accelerated mirror descent on beliefs | $\nabla_b F$, a mirror map, primal beliefs | do valid-by-construction iterates fix the momentum failure? |
| 5 | Langevin with annealed temperature | experiment 4's interface | does it leave poor fixed points on multimodal graphs? |

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
