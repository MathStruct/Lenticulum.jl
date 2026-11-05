#open-problem

> **Status: open (2026-11-05).** Area: probabilistic graphical models, tropical geometry, topological data analysis · Prerequisites: [[Switching Constraint Factors]], [[Why Learned Relations Do Not Collapse]] · Related: [[Feature Backlog]] (overlaid factors), `lib/VariationalDiffusion.jl/examples/calibration.md`, [[LP-022 Constraint factors are normalised in their residual, not their channels]].

> Sources: Zhang, Naitzat & Lim, *Tropical Geometry of Deep Neural Networks*, ICML 2018 (ReLU networks are tropical rational maps); original analysis of this repository. code: `src/switching.jl`; tests: `test/rq008.jl`
>
> Bibliography: [[Bibliography#^zhang2018tropical|Zhang et al. 2018]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Question

Several factors overlaid on the same variables multiply, so their energies add. Which relations
can such overlays of simple factors (linear constraints, fixed nonlinear energies, switches)
represent? Which of them stay tractable, and normalised in their data channels? How does the
combinatorics of a switching factor's low-noise limit (a tropical polynomial) relate to the
topology of the learned relation (its branches and their connections)? The motivation is a
program in which a large learned model is distilled step by step into a factor graph of small
learners. There, each step needs to know how many branches a piece has and how the pieces connect.

## Setting

- **Overlay.** Factors $f_1, \dots, f_m$ on one variable set $z$ form the overlay $f(z) = \prod_i f_i(z)$, with energy
  $E = \sum_i E_i$. Each $f_i$ keeps its own parameters.
- **Linear constraint.** $E_{\mathrm{lin}}(z) = \tfrac12 (Az - c)^\top Q^{-1} (Az - c)$, convex in $z$.
- **Switching factor** (`SwitchingConstraintFactor`, the location-shift case: $A$, $Q$ shared by the
  outcomes) with its outcome $k$ summed out:

$$
E_{\mathrm{sw}}(z) = -\log \sum_{k=1}^K w_k\, \mathcal N(Az - b_k;\ 0, Q).
$$

- **Tropical polynomial** (max-plus): $T(y) = \max_k (\langle a_k, y\rangle + c_k)$, convex and piecewise affine. Its
  tropical hypersurface is the set where the maximum is attained at least twice; it partitions $y$-space
  into the cells where each term wins.

## What is known

1. **Convex overlays stay unimodal** (elementary). A sum of convex energies is convex, so an overlay
   of log-concave factors (a linear constraint and a convex $\varphi(z)$) is log-concave: its superlevel
   sets are convex and its mode is unique if any term is strictly convex. Such an overlay cannot
   represent a mixture, and it cannot replace a switching factor.
2. **The summed-out switch is a soft-min, and a difference of convex functions** (derived here;
   checked in `test/rq008.jl` against the graph's exact free energy to $10^{-12}$). With $y = Az$,

$$
E_{\mathrm{sw}}(z) = \tfrac12 y^\top Q^{-1} y \;-\; \operatorname{logsumexp}_k\Bigl(b_k^\top Q^{-1} y - \tfrac12 b_k^\top Q^{-1} b_k + \log w_k\Bigr) + \tfrac12 \log\det 2\pi Q .
$$

   That is the linear constraint's convex quadratic minus a convex log-sum-exp of affine functions
   of $y$.

3. **Its low-noise limit is tropical** (derived here; checked in `test/rq008.jl`). With $Q = sQ_0$,

$$
s\,E_{\mathrm{sw}}(z) \;\longrightarrow\; \tfrac12 y^\top Q_0^{-1} y - \max_k\Bigl(b_k^\top Q_0^{-1} y - \tfrac12 b_k^\top Q_0^{-1} b_k\Bigr), \qquad s \to 0,
$$

   a convex quadratic minus a tropical polynomial in $y$. The error is $O(s \log s)$, dominated by
   $s \cdot \tfrac12 \log\det 2\pi s Q_0$. Measured: 0.035, $8.1 \cdot 10^{-4}$ and $1.3 \cdot 10^{-5}$ at $s = 10^{-2}, 10^{-4}, 10^{-6}$, that is $0.76$–$0.92\, s\lvert\log s\rvert$. The
   weights drop out of the limit; they set where the cells' boundaries lie at finite $s$.
   So a switching factor *is* an overlay: a linear constraint plus a fixed concave, piecewise-linear
   factor (minus a tropical polynomial), with learnable offsets $b_k$. The tropical hypersurface
   separates the regions each outcome wins.

4. **ReLU networks are tropical rational maps** (Zhang, Naitzat & Lim 2018), differences of two
   tropical polynomials. The same max-plus language describes switches and any piecewise-linear
   learner distilled into them.
5. **Collapse.** A linear constraint with every coefficient learned collapses on data (LP-022). A
   plain overlay with a convex $\varphi$ does not prevent it, because at $A = 0$ the linear factor is still
   at its maximum. An overlay node that normalises its product over the data channel,
   $Z(\theta) = \int \prod_i f_i(z)\, dz_{\text{data}}$, would: it makes the overlay a conditional density. For
   log-concave members this is a log-concave integral, so it is tractable by Laplace or quadrature in
   low dimension.
6. **Branch counts are measured by hand today.** `examples/calibration.md` counts recalled branches
   of learned implicit relations against analytic ones.

## What would count as an answer

- **A representation theorem.** Characterise the energies an overlay of linear constraints and
  fixed piecewise-linear factors represents (all differences of convex piecewise-quadratic
  functions?), and which of them are normalisable in a given channel.
- **A tractability result.** For which overlays are exact or provably good messages available
  (closed form, EP with convergence, a cluster node)? And what does the normalised overlay node cost?
- **A topology statement.** Relate the tropical hypersurface of a switching factor (its cells and
  their adjacency) to the homology of the learned relation's zero set at small noise. For instance,
  the number of branches of a query against the number of cells its line crosses.
- **A measurement protocol.** Persistent homology of the root sets of `implicit_roots` over a grid
  of queries ($\beta_0$, the components, as a branch count; $\beta_1$, the loops), compared with the analytic
  relation on the calibration benchmark, and used to choose the number of outcomes $K$ of a distilled
  switch.

## Why it matters here

- **Overlaid factors** (the Feature Backlog's "mirrored factors"): the identity in 2 says which
  overlays add expressive power (a concave, piecewise-linear member) and which do not (a convex one).
  It also suggests building the overlay as one node, exact like a cluster node, with an optional
  normaliser.
- **Distillation into a factor graph** (the LenticulumLM moonshot, [[Language Models]]; a diffusion model distilled
  subgraph by subgraph): each distilled piece needs a branch count and a connectivity. The tropical
  form gives the pieces, and persistent homology measures the target.
- **The calibration defects** (spurious and off-domain roots, merged branches) are questions about
  the topology of the smoothed relation against the data's.

## Leads

- **Verify 2 and 3 in more cases:** outcome-dependent $A_k$ or $Q_k$ (`RegimeFactor`). There the pieces are
  different quadratics, and the limit is a minimum of quadratics, not a quadratic minus a tropical
  polynomial.
- **The overlay node** exists since 2026-11-05 for members with a cluster form (`OverlayFactor`,
  `lib/Mycelium.jl/src/overlay.md`), with the normaliser per joint outcome; it equals a cluster node
  of the members and the closed forms (`test/overlay.jl`). Open: members with nonlinear energies
  (EP or Laplace), and the normalisation of the whole product, which gates the outcomes by $x$.
- **Dequantisation.** The passage from log-sum-exp to max as $s \to 0$ is Maslov's dequantisation. The
  temperature $s$ is the switch's noise, so annealing the noise is a path from a smooth mixture to its
  tropical skeleton, as annealing the field levels is for implicit relations (`calibration.md`).
- **Persistent homology** on the calibration benchmark's root sets: Ripser-style Vietoris–Rips over
  the found roots, per query line and over the whole grid.
- **Known not to work:** convex members alone (1). A product of experts with log-concave experts is
  unimodal.
