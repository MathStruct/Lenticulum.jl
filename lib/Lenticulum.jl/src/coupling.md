#implementation

> `coupling_weights(g, ps)`: one weight per factor, the strongest coupling it adds between two of its
> variables, for `spanning_tree_schedule(g; weights = coupling_weights(g, ps))` (I25). Linear-Gaussian
> factors report their precision over their channels (`coupling_precision`); the weight is the
> largest off-diagonal block of it, measured in the variables' total precision, so it lies in
> $[0, 1]$ and does not depend on the variables' units.

> Sources: Wainwright, Jaakkola & Willsky (2003), tree-based reparameterisation (`wainwright2003trp`);
> Malioutov, Johnson & Willsky (2006), walk-sums (`malioutov2006walksums`), for the normalised
> coupling $|J_{uv}| / \sqrt{J_{uu} J_{vv}}$; both motivation only, no theorem used. code: `coupling.jl`; Mycelium
> `lib/Mycelium.jl/src/schedules.jl` (`spanning_tree_schedule`); tests: `test/i25_coupling_weights.jl`;
> benchmark: `bench/bp_trees.jl`
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
by units. That case is not measured. The default is `normalize = true` for that invariance; pass
`normalize = false` when the variables share a scale.

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
