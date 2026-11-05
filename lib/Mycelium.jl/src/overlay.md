#implementation

> `OverlayFactor`: several different factors on the same variables, each with its own
> parameters, as one node. Unnormalised it is their product, equal to a cluster node of the
> members. With `normalise = (:y, …)` it is, for each joint outcome of the members' switches, the
> product normalised as a density of `y`: a product of Gaussian experts, and a mixture of experts
> with switches. A learned relation on data cannot collapse then.

> Sources: original to this repository ([[RQ-008 Overlays, Mixtures and Tropical Limits]], the Feature Backlog's "mirrored factors"). code: `overlay.jl`; tests: Lenticulum `test/overlay.jl`

## 1. What it is

`OverlayFactor(:a => f₁, :b => f₂, …)` holds members stated by a `cluster_form`. Its channels are the
union of theirs; members with a channel of one name share that variable, switch channels included.
Its parameters are a `NamedTuple` keyed by the member names, each member with its own.

**Unnormalised**, the energy is $\sum_i E_i$. In cluster-form terms the overlay is one form:
- the members' residuals are stacked;
- their noises form a block-diagonal $Q$;
- its outcome is the joint outcome of all the members' switch channels, column-major in their order;
- its log-weights are the sum of the members' for that joint outcome.

If any member's coefficients or noise depend on its outcome, so do the overlay's.

**Normalised** in channels $y$, for each joint outcome $k$:

$$
f(y, x, k) = w_k\, \mathcal N\bigl(y;\ P_k^{-1} A_y^\top Q^{-1}(C_k - A_x x),\ P_k^{-1}\bigr), \qquad P_k = A_y^\top Q^{-1} A_y,
$$

with $A$, $C$, $Q$ the stacked form and $w_k$ the product of the members' weights. It is
$\prod_i f_i / \int \prod_i f_i\, dy$ outcome by outcome: the $y$-quadratic of the summed energies, completed. So
members that do not involve $y$ cancel. $P_k$ must be invertible (the members must determine $y$), or
the form refuses. A common scaling of the coefficients and the noise leaves $P_k^{-1}$ and the mean
unchanged.

**Outside a cluster** its messages and free energy are those of a one-factor cluster (`as_cluster`).
**Inside one** it is its cluster form.

## 2. The normaliser and LP-022

A constraint factor with every coefficient learned collapses on data (A = 0). Normalised in its
data channel it is a conditional density there: the normalised relation is
$\mathcal N(y;\ -A_y^{-1}(A_x x - c),\ A_y^{-1} Q A_y^{-\top})$, so its coefficients and its scale are the conditional mean and the
variance. `learns_unnormalised` is false for a normalised overlay, so no warning fires.
`test/overlay.jl` checks this with one fully learned `LinearConstraintFactor` on 100 points of
$y = 2x + 0.5 + 0.3\varepsilon$:
- $A = 0$ is refused (singular $P$);
- at the maximum-likelihood point (least squares, its residual variance) the free energy equals
  $-\sum_i \log \mathcal N(y_i;\ a x_i + c, v)$ to $10^{-10}$;
- it is lower there than at perturbed parameters;
- the gradient vanishes there.

## 3. Normalising outcome by outcome

With switches the normalisation is per joint outcome: $\sum_k w_k\, p_k(y \mid x)$, a mixture of experts whose
weights do not depend on $x$. The exact normalisation of the whole product,
$\prod_i f_i / \sum_k w_k Z_k(x)$, gates the outcomes by $x$. It is a different model and not a cluster form,
because of the division by a sum. Without switches the two agree.

## 4. Checked by

`test/overlay.jl`:
- **Unnormalised overlay** of a linear constraint and a switch: equals a cluster node of the same
  members as separate nodes (free energy and outcome posterior to $10^{-12}$). Its message to the outcome
  outside a cluster matches the closed form.
- **Normalised:**
  - two linear experts give the precision-weighted Gaussian;
  - a linear expert and a switch give the mixture of experts $\sum_k w_k \mathcal N(y; m_k, 1/P)$;
  - a prior on $x$ in the overlay cancels;
  - all to $10^{-12}$.
- **The LP-022 case** of §2.

## 5. What it does not do

- **Members without a `cluster_form`** (a nonlinear energy, a convex $\varphi$) are not supported. They would
  need Laplace or EP messages (RQ-008, leads).
- **No closed-form M-step** for the overlay (`m_step`); EM keeps its parameters, and gradients
  learn them.
- **Only per-outcome normalisation** with switches (§3).
- **No batching over plates** beyond what its cluster form gets as a cluster node.
