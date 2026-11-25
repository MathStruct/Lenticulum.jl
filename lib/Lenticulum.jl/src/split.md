#implementation

> `SplitFactor(whole, pieces; dim)` (`split.jl`): a channel $z$ and pieces of it, $x_k = z[I_k]$,
> imposed exactly. Concatenation, splitting, permutation and selection of coordinates in one factor,
> with [[extended_gaussian|extended Gaussian]] messages and `RelayFactor`'s free
> energy. Backlog row U2.

> Sources: Stein & Samuelson, *A Category for unifying Gaussian Probability and Nondeterminism*, CALCO 2023 (the belief algebra). code: `split.jl`; `lib/LenticulumCore.jl/src/extended_gaussian.jl` (`preimage`, `marginalise`), `lib/Mycelium.jl/src/gaussian.jl` (`combine`, `variable_entropy`). tests: `test/split.jl`.
>
> Bibliography: [[Bibliography#^stein2023gaussian|Stein & Samuelson 2023]]
>
> Theory (CT-ML wiki): [Gaussian Relations](https://mathstruct.org/CategoryTheory-ML-Wiki/Gaussian-Relations)

## Theory

The factor is $f(z, x_1, \dots, x_K) = \prod_k \delta(x_k - S_k z)$, with $S_k$ the selection matrix
of the index set $I_k \subseteq \{1, \dots, d\}$ ($S_k z = z[I_k]$). Each piece is a function of $z$,
so every sum-product message is an operation on $z$:

$$
\mu_{f \to z}(z) = \prod_k \mu_{k \to f}(S_k z), \qquad
\mu_{f \to x_k}(x_k) = \int \mu_{z \to f}(z) \prod_{j \ne k} \mu_{j \to f}(S_j z)\; \delta(x_k - S_k z)\, dz .
$$

The first is a product of **preimages** $z \mapsto \mu_j(S_j z)$ (flat off $I_j$); the second is the
marginal on $I_k$ of $\mu_{z\to f}$ times the other pieces' preimages. Preimage, product
(`combine`) and marginal are exact for extended Gaussians, so the messages are exact for points,
Gaussians, extended Gaussians and no information (improper Gaussians as far as they are extended
Gaussians; see the limits below). Nothing in this needs the
$I_k$ to be disjoint or to cover $\{1, \dots, d\}$: overlapping pieces meet in the product on $z$
(two points that disagree on a shared coordinate are a contradiction, an `ArgumentError`), and an
uncovered coordinate is flat in $\mu_{f\to z}$.

## Free energy

The factor is the linear relation $G\,(z, x) = 0$ with $G = [S \;\; -I]$ ($S$ the stacked $S_k$, $m$
rows), and its Bethe term is `HardConstraintFactor`'s for that relation: the limit of the noisy
relation's term as the noise vanishes,

$$
F_f \;=\; \tfrac12 \log\det\bigl(\tilde G \tilde G^\top\bigr) \;-\; H_R(b_f),
$$

with $\tilde G$ the relation on the coordinates the inputs leave free (points and hard directions
substituted out) and $H_R$ the entropy of the factor belief relative to Lebesgue measure on the
solution set $R$ (`hard_constraint.md`, "Free energy").

**Without hard directions in the inputs** this is $-H(q)$, with $q$ the normalised product
$\mu_{z\to f}(z)\prod_k \mu_{k\to f}(S_k z)$ on $z$: `RelayFactor`'s term (LP-019). $R$ is the image of
$z \mapsto (z, Sz)$, whose Jacobian $J$ has $J^\top J = I + S^\top S = \operatorname{diag}(1 + c_i)$, $c_i$
the number of pieces holding coordinate $i$; so $H_R = H(q) + \tfrac12 \sum_i \log(1 + c_i)$. And
$GG^\top = I + SS^\top$ has the same determinant (Sylvester: $\det(I_m + SS^\top) = \det(I_d + S^\top S)$),
so the two half-log-determinants cancel, for any overlap and any uncovered coordinates. The code
takes this path then, from $q$ in covariance form.

**With hard directions** (a point, or an extended Gaussian with a hard direction) the cancellation
can fail: a hard direction oblique to the coordinates, in a $z$ whose coordinates are held by
different numbers of pieces, leaves a Jacobian that $-H(q)$ does not have. The review found one
(LP-069, finding 8): a prior on the line $z = (T, T)$, $x = z_1$ clamped at 0, where $-H(q)$ gave
$-\log Z + \log\sqrt2$. The code then calls `HardConstraintFactor`'s `local_free_energy` with
$A_z = S$, $A_{x_k} = -I$ on its rows, $c = 0$. That term is infinite, and throws, where two inputs
fix the same direction (two points on a shared coordinate, or a point on a piece and a hard
direction of $z$ there): the evidence has a $\delta(0)$. A factor belief flat in some direction is
improper and throws too. The grading is `HardConstraintFactor`'s (`fit`, `complexity`,
`negentropy`); on the first path `complexity` is 0.

## Implementation

`_split_on_whole` converts every input that takes part to an extended Gaussian (`_as_extended`,
which collapses a mixture to its moments as `HardConstraintFactor` does) in the promoted float type
of those inputs (`Float64` when none is informative, P30; the input left out does not count).
Pieces without information are skipped. When the remaining pieces' index sets are pairwise
disjoint, their messages are joined block-diagonally in covariance form, with a flat block for the
coordinates they do not cover, and permuted into place by `pushforward`. This multiplies nothing
and so cuts no precision. Overlapping pieces are pulled back with `preimage(e, S_k)` and multiplied
with `combine`. $z$'s own input, unless it is the target, is then multiplied in. A piece's message is
the marginal of the product on $I_k$.

That marginal (`_split_marginal`) has its hard directions judged against $q$, not against itself.
`marginalise` alone uses the scale of the marginal, and a piece reached only by coordinates an
observation fixes then has a covariance of pure rounding, about $10^{-31}$ for a $q$ of scale 1. A
relative tolerance takes that for variances (the oracle test met it, tree 6 of `test/split.jl`).
The rounding is second order (a basis of $q$'s free directions off by $\varepsilon$ leaves
$\varepsilon^2$ on its hard coordinates), so eigenvalues of the marginal below the square of $q$'s
tolerance over its scale are set to zero, on values only. A first version cut at $q$'s tolerance
itself, which also removed a true variance of $10^{-16}$ against 1 (LP-069, finding 2).

The tests found one defect outside the factor: `LenticulumCore.pushforward` made its default
integer offset a `Float64` vector, so `marginalise` of a `Float32` belief returned `Float64`. An
integer offset of `pushforward` and `preimage` now takes the promoted float type of the belief and
the map (P30; regression tests in LenticulumCore's `test/extended_gaussian.jl`).

The factor has no parameters, is not learnable, declares `exact_messages` (P13) and
`coupling_precision(f, ps) === :hard` (I25). Every constructor checks its arguments.

## What it does not do

- **Cost.** Every message goes through $d$-dimensional extended-Gaussian operations (eigen- and
  SVD-based normalisation), not through index arithmetic on means and covariances alone. The row's
  motivation was that a selection-matrix `HardConstraintFactor` would be more expensive. Neither is
  measured.
- **Plates.** There is no batched `factor_messages` method: a plate sweep calls `factor_message`
  per replica (Mycelium's default).
- **Mean field.** There is no `mean_field_message`; a `SplitFactor` on a `MeanField` node throws
  Mycelium's default error.
- **Derivatives.** Not tested under ForwardDiff. Not differentiable with Zygote: the message path
  mutates (the `combine` fold, `marginalise`'s selection matrix), and the reviewer's probe failed in
  the pullback (LP-069, finding 12). Where `_split_marginal` cuts rounding eigenvalues, the tangent
  of the cut part is dropped.
- **`RelayFactor` equivalence** holds for proper, non-mixture inputs. `RelayFactor` passes a mixture
  through unchanged where this factor collapses it, and gives 0 where every input is uninformative,
  where this factor throws (an improper factor belief).
- **Limits of the extended-Gaussian core**, met through this factor and not changed (LP-069):
  - Contradictory points are detected with a tolerance relative to the scale of the means, so a
    disagreement of 1 next to a coordinate of $10^{10}$ passes as agreement (finding 1).
  - An improper Gaussian site that is not an extended Gaussian (a linear potential with zero
    precision and nonzero information, or a negative precision) loses information or is refused in
    the conversion (finding 4).
  - A `Float32` extended Gaussian takes a variance below $32\,\varepsilon_{32} \approx 3.8\cdot10^{-6}$
    of the largest for a hard direction: two pieces with variances 1 and $10^6$ give the right
    message to $z$, but its entropy, and so $F$, drops the unit direction (`@test_broken` in
    `test/split.jl`).
- **Non-Gaussian inputs.** Categorical and other non-Gaussian beliefs are refused by the
  conversion to an extended Gaussian.

## Device arrays (Q7)

Selection indices stay on the host. The selection operators and flat messages follow the inputs
or target cavity. The rank cut for the covariance of a marginal uses the core host spectral
helper.
