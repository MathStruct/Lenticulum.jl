#implementation

> `coupling_weights(g, ps)`: one weight per factor, the strongest coupling it adds between two of its
> variables, for `spanning_tree_schedule(g; weights = coupling_weights(g, ps))` (I25). Linear-Gaussian
> factors report their precision over their channels (`coupling_precision`); the weight is the
> largest off-diagonal block of it, measured in the variables' total precision, so it lies in
> $[0, 1]$ and does not depend on the variables' units.

> Sources: Wainwright, Jaakkola & Willsky (2003), tree-based reparameterisation (`wainwright2003trp`); Malioutov, Johnson & Willsky (2006), walk-sums (`malioutov2006walksums`), for the normalised coupling $|J_{uv}| / \sqrt{J_{uu} J_{vv}}$; both motivation only, no theorem used. code: `coupling.jl`; Mycelium `lib/Mycelium.jl/src/schedules.jl` (`spanning_tree_schedule`); tests: `test/i25_coupling_weights.jl`; benchmark: `bench/bp_trees.jl`
>
> Bibliography: [[Bibliography#^wainwright2003trp|Wainwright et al. 2003]] · [[Bibliography#^malioutov2006walksums|Malioutov et al. 2006]]
>
> Theory (CT-ML wiki): [[Composing Optimisers]]

## 1. What it computes

A factor with a linear-Gaussian form adds to the joint precision of the graph the matrix
$J^f = C^\top Q^{-1} C$ over its channels, $C = [C_1 \cdots C_n]$ the stacked coefficients and $Q$ its
noise. `coupling_precision(f, ps)` returns `(dims, J)`:

| factor | $J^f$ |
|---|---|
| `GaussianFactor` ($y = Ax + b + \varepsilon$) | $[-A\ \ I]^\top Q^{-1} [-A\ \ I]$, $Q$ = `noise_covariance(f, ps)` |
| `GaussianPrior` | $\Sigma^{-1}$ (one channel) |
| `LinearConstraintFactor`, `SwitchingConstraintFactor` | $A^\top Q^{-1} A$ over the continuous channels; the switch shifts only the constant |
| `RegimeFactor` | $\sum_k w_k A_k^\top Q_k^{-1} A_k$, $w = \operatorname{softmax}(\text{logw})$ |
| `HardConstraintFactor` | `:hard` (infinite along the directions it fixes) |
| anything else (`NonlinearFactor`, `LikelihoodFactor`, other packages) | `nothing` |

With $D_v = \sum_{(f, c) \to v} J^f_{cc}$ the diagonal blocks every linear-Gaussian factor puts on
variable $v$, the weight of factor $f$ is

$$
w_f = \max_{a \ne b,\ u(a) \ne v(b)} \bigl\lVert D_u^{-1/2} J^f_{ab} D_v^{-1/2} \bigr\rVert_2
$$

(`normalize = true`, $D^{-1/2}$ the pseudo-inverse square root) or $\lVert J^f_{ab} \rVert_2$
(`normalize = false`). Since $J^f \succeq 0$ and $D_u \succeq J^f_{aa}$, the normalised weight is at
most 1. For a scalar constraint $x_a - x_b = \varepsilon$, $\varepsilon \sim \mathcal N(0, q)$, the raw
weight is $1/q$, the weight measured in I3.

Hard factors get the weight `hard`: 1 normalised (the largest possible), otherwise one more than the
largest finite weight. Factors without a form get `default` (0). Neither adds to $D$; a hard
factor's infinite precision would make every other weight on its variables zero.

## 2. Why this weight

The tree heuristic (I3, `schedules.md` "Choice of tree") puts strong couplings inside the tree. For a
Gaussian graph, how strongly $u$ and $v$ are tied is a property of the joint precision normalised by
each variable's own precision: $|J_{uv}| / \sqrt{J_{uu} J_{vv}}$ is the edge weight of the walk-sum
analysis of Gaussian BP, and it is invariant when a variable is rescaled. The raw $\lVert J_{ab} \rVert$
changes with the units of $u$ and $v$: rescaling $x_2 \mapsto 10 x_2$ divides every raw weight on $x_2$
by 10 and changes no message. Normalisation needs the whole graph ($D_v$), which is why the weights
are computed per graph and not per factor.

This is a heuristic. A maximum-weight tree under these weights does not minimise the spectral
radius of a BP sweep; it only keeps the strongest normalised couplings inside the tree.

## 3. Measured

`julia --project=bench bench/bp_trees.jl` (2026-11-18), the I3 grids (10 × 10, prior variance 100;
see `lib/Mycelium.jl/src/schedules.md` §"Choice of tree" for the setting), rows "coupling" (normalised)
and "coupling raw":

Message evaluations (sweeps) to a residual of `1e-11`; output in `~/.cache/lenticulum/i25/bp_trees.log`.
"precision" is I3's hand-made `1/q`.

| | isotropic | horizontal | vertical | random |
|---|---:|---:|---:|---:|
| BFS | 1 017 520 (1 106) | 5 882 480 (6 394) | 2 186 840 (2 377) | 4 063 640 (4 417) |
| precision = coupling raw | 1 017 520 (1 106) | 2 139 000 (2 325) | 2 186 840 (2 377) | 2 369 920 (2 576) |
| coupling (normalised) | 1 181 280 (1 284) | 2 139 000 (2 325) | 2 186 840 (2 377) | 2 571 400 (2 795) |
| BFS + A(3) | 107 640 (117) | 792 120 (861) | 161 920 (176) | 196 880 (214) |
| precision + A(3) = coupling raw + A(3) | 107 640 (117) | 147 200 (160) | 161 920 (176) | 67 160 (73) |
| coupling + A(3) | 110 400 (120) | 147 200 (160) | 161 920 (176) | 65 320 (71) |

Every run converged, mean errors below `3e-10` against the dense solve. The raw weights reproduce the
hand-made `1/q` rows exactly (the same forest; the test checks the masks). The normalised weights pick
the same tree where the anisotropy is uniform and a different one elsewhere: on the isotropic grid
(all raw weights equal, so Kruskal's tie-break gives BFS; normalised, boundary edges weigh more)
16 % more evaluations plain and 3 % with Anderson; on the random grid 8 % more plain and 3 % fewer
with Anderson. So on these grids, whose variables share one scale, **normalisation does not pay**;
its case is graphs whose variables have different units or scales, where raw weights rank couplings
by units (measured next).

### Normalised or raw: variables of mixed scales

`julia --project=bench bench/bp_trees_scale.jl` (2026-11-18; log `~/.cache/lenticulum/i25b/bp_trees_scale.log`)
states the same grids in mixed units, $y_v = x_v / s_v$: "blocks" puts the right half of the grid in
units $10^{-3}$ (numbers 1000 times larger), "random" draws $s_v = 10^{U(-3,3)}$. The model is the
same; the raw weight of a constraint becomes $s_u s_v / q$. Stopping is relative (`tol = 0`,
`rtol = 1e-10`), so it does not depend on the units either. Every run converged, with errors at most
`6e-9` in the original units.

Gaussian BP is equivariant under such a rescaling, and the plain runs confirm it: a schedule with a
fixed tree (BFS, `1/q`) takes the same number of sweeps at every scale. Message evaluations, plain
(with Anderson(3) in brackets); "1/q" is the tree of the physical precisions, the target:

| noise, scale | 1/q | coupling (normalised) | coupling raw |
|---|---:|---:|---:|
| isotropic, none | 959 560 (103 040) | 1 120 560 (101 200) | 959 560 (103 040) |
| isotropic, blocks | 959 560 (101 200) | 1 104 920 (115 920) | 959 560 (101 200) |
| isotropic, random | 959 560 (92 000) | 990 840 (83 720) | 1 043 280 (92 920) |
| horizontal, none | 1 758 120 (122 360) | 1 758 120 (122 360) | 1 758 120 (122 360) |
| horizontal, blocks | 1 758 120 (47 840) | 1 758 120 (47 840) | 2 241 120 (115 920) |
| horizontal, random | 1 758 120 (54 280) | 1 753 520 (57 960) | 2 655 120 (143 520) |
| random, none | 1 901 640 (61 640) | 2 068 160 (59 800) | 1 901 640 (61 640) |
| random, blocks | 1 901 640 (44 160) | 2 068 160 (62 560) | 2 088 400 (71 760) |
| random, random | 1 901 640 (48 760) | 2 068 160 (54 280) | 2 530 000 (92 000) |

- The normalised weights choose the same tree at every scale where the couplings differ (horizontal
  and random noise: 379 and 369 of the 379 edges of the `1/q` tree, in every row). On the isotropic
  grid many normalised weights are equal in exact arithmetic, and rounding breaks the ties
  differently at each scale (357, 351, 339 shared edges), so the tree does change there, though not
  because of the units. The 378 against 379 in "horizontal, random" is the same kind of tie.
- The raw weights follow the units. On mixed scales where the noise singles out strong edges
  (horizontal and random noise) they need 1 to 51 % more plain evaluations than normalised
  (horizontal/blocks 2.24 M against 1.76 M, horizontal/random 2.66 M against 1.75 M), and with
  Anderson 1.15 to 2.5 times as many. On the isotropic grid with blocks, raw happens to equal `1/q`:
  its three weight classes (inside the left half, across, inside the right half) are each tied,
  and Kruskal's tie-break by edge index then gives the BFS tree, which is the `1/q` tree here.
- Normalised against the `1/q` target on mixed scales: 0 to 15 % more plain evaluations; with
  Anderson from 9 % fewer to 42 % more (random/blocks 62 560 against 44 160).
- With one shared scale (the "none" rows) raw equals `1/q`, and normalised costs up to 17 % more plain
  (isotropic) and at most 2 % more with Anderson.
- Anderson acceleration is not equivariant: with the tree fixed, its counts vary with the scale
  (BFS + A(3) on the horizontal grid: 616 400, 879 520, 882 280), because its least-squares mixing
  weights depend on the message parameters' units.

**Default: `normalize = true`** (kept 2026-11-18 on this measurement). Its largest loss in these
eighteen settings is 17 % plain against raw (isotropic, one scale); raw's largest loss against it is
51 % plain and 2.5 times with Anderson (mixed scales). Pass `normalize = false` when the variables are
known to share one scale. Only 10 × 10 scalar grids were measured.

## 4. What it does not do

- Nonlinear and non-Gaussian factors get a constant (`default`); their coupling depends on where
  they are linearised, which a schedule built before inference does not know.
- A hard constraint is ranked above every soft one, whatever its coefficients.
- One weight per factor: a factor with three or more channels gives all its edges its strongest
  pair's weight.
- Parameters are converted to `Float64`: dual or device parameters are not supported (the weights
  choose a schedule; they are not differentiated). Untested: plates, tied parameters across nodes,
  `MeanField()` nodes (the weights are computed from the parameters alone, whatever the rule).

Tests (`test/i25_coupling_weights.jl`): the weights against an independently assembled dense joint
precision on random loopy graphs with variables of dimensions 1–3; invariance of the normalised
weights and scaling of the raw ones under rescaling a variable; each factor type's
`coupling_precision` against its formula; hard, nonlinear and degenerate factors; on the anisotropic
grids the raw weights give the `1/q` forest, the normalised ones keep every strong edge, and `infer!`
reaches the dense solution.


## Element type

Computation follows Julia promotion of the parameters, noise, incoming messages and data it reads;
data and parameters are not converted to force a result type. Constants and numerical
hyperparameters are cast at use.

Precision blocks and weights retain their promoted scalar type. Default and hard weights are cast at
use, while pseudoinverse zeros use the eigenvalue type. P31 compares Float32 and Float64 weights
with and without normalisation. An entirely unrecognised/hard-only graph falls back to the floating
type of `default` because it has no finite precision from which to select a type. Ordering and
differentiation support are otherwise unchanged.

Checks: `lib/Lenticulum.jl/test/p31_element_type.jl`. The implementation is not a claim of coverage
for every mixed precision, derivative, or device configuration.
