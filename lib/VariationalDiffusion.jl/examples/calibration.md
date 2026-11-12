# Calibration of implicit relations

Run `julia -t auto --project=test examples/calibration.jl` from `lib/VariationalDiffusion.jl`
(about 45 minutes on 8 CPU threads since the queries run in parallel and the remedies were added,
nearly all of it in the root queries; `CALIBRATION_SMOKE=1` runs it at small size in a few minutes,
to check the script). This answers the request of
both reviews of 2026-11-04 ([[LP-023 Findings of the external reviews of 2026-11-04]]): measure how
far the stable roots of a learned relation are from the relation, how many are spurious, and how
many branches are recalled.

## Question

`ImplicitDiffusion` answers a query (clamp some coordinates, solve for the rest) by the stable
zeros of a trained noise predictor's field, smoothed over a range of noise levels. Those zeros are
modes of the smoothed density $p_t$, not points of the data's support. How close do they come, on
relations whose branches are known?

## Setup

- **Relations** in $\mathbb R^2$, data noise 0.02:
  - the unit circle;
  - the lemniscate of Gerono $y^2 = x^2 - x^4$, two branches crossing at the origin;
  - two concentric circles of radius 1 and 2 with weights 0.9 and 0.1, a minority branch.
- **Model**: the 3-layer MLP of `circle_mlp.jl` (time embedding, about 5k parameters), trained by
  denoising score matching for 6000 Adam steps at batch 256, $t \le 0.2$, one seed per relation.
- **Queries**: $x$ clamped on a grid (step 0.1, from −1.4 to 1.4, and from −2.4 to 2.4 for the two
  circles), every stable $y$ by `implicit_roots` (33 starts, spread 1.5 or 2.5).
- **Metrics**, with tolerance $\tau = 0.1$:
  - *recall*: the fraction of true branches $y(x)$ matched by a root within $\tau$;
  - *spurious*: the fraction of roots farther than $\tau$ from every branch at their $x$, including
    the roots of queries with no branch;
  - *off-domain*: queries with no branch at their $x$ (outside the relation's $x$ range) that still
    return a root;
  - *dist*: the mean $\lvert y - y(x)\rvert$ of matched roots;
  - *radial*: for circles, the mean $\lvert z\rvert - r$ of matched roots (negative is inward);
  - *minority recall*: recall of the radius-2 branches alone.
- **Field levels**, without retraining: low ($t \le 0.01$), the default of `field_nodes` ($t \le 0.05$),
  high ($t \le 0.2$).

## Measured (2026-11-05)

| relation | levels | recall | spurious (roots) | off-domain | dist | radial | minority recall |
|---|---|---|---|---|---|---|---|
| circle | low | 0.857 | 0.234 (47) | 8/8 | 0.0207 | −0.0099 | |
| circle | default | 0.833 | 0.255 (47) | 8/8 | 0.0285 | −0.0196 | |
| circle | high | 0.048 | 0.973 (37) | 8/8 | 0.0020 | 0.0000 | |
| lemniscate | low | 0.524 | 0.621 (58) | 8/8 | 0.0543 | | |
| lemniscate | default | 0.286 | 0.786 (56) | 8/8 | 0.0593 | | |
| lemniscate | high | 0.190 | 0.828 (29) | 8/8 | 0.0446 | | |
| two circles | low | 0.282 | 0.739 (134) | 8/8 | 0.0151 | +0.0034 | 0.000 |
| two circles | default | 0.282 | 0.703 (118) | 8/8 | 0.0143 | −0.0104 | 0.000 |
| two circles | high | 0.000 | 1.000 (59) | 8/8 | | | 0.000 |

## What it shows

1. **Every query outside the relation's range returns a root** (8 of 8 in each case). Clamping $x$
   beyond the circle and asking for $y$ finds the smoothed density's ridge, a nearest-point
   projection, and nothing marks it as off the relation. This is the reviewers' "on-manifold roots
   against projections". A check of the full score (clamped coordinates included) at a small $t$
   would separate them; it is not implemented.
2. **Smoothing shifts and merges.** Matched circle roots sit inside the circle, by 0.010 at the low
   levels and 0.020 at the default ones, so the bias grows with $t$, the curvature shift
   ($O(\sigma_t^2\kappa)$). At high levels the two branches of the circle merge, and recall falls to 0.05.
3. **A crossing is hard.** On the lemniscate, recall is at best 0.52, and most roots are spurious
   (62% at low levels). Near the crossing the two branches are within a smoothing width of each
   other, and the field's zeros lie between them.
4. **A minority branch is lost.** The radius-2 circle, 10% of the data, is never recalled at any
   level. The roots the queries find there are spurious or belong to the inner circle. Either the
   model did not learn the outer circle at this training budget, or its modes are too shallow to be
   stable roots; this run does not separate the two.
5. **Low levels are better here on every relation.** The default levels do not help any of them.
   The high levels are worse throughout.

## Caveats

- One seed per relation, a small model, an untuned training budget (each model trained in seconds),
  and a fixed $\tau$. These numbers are a baseline, not the best the method can do. In particular, the
  minority branch may only need more training or more data.
- Recall counts a branch found if any root is within $\tau$, so spurious roots near a branch can count
  as recall. Precision is the spurious fraction.
- Starts are random (33 per query); a branch missed by every start is counted as not recalled.

## Remedies (F3, measured 2026-11-12; the ridge residual 2026-11-13)

The same models, queries and seeds; the "all roots" rows reproduce the table above exactly. Each root
then gets three verdicts, and the metrics are recomputed for the roots each one keeps:

- **round trip** (`round_trip`, `implicit.md` §5): clamp the root's $y$, free $x$, solve back from the
  root; keep the root if $x$ moves by at most $\tau$. Needs the model only.
- **ridge** (`ridge_residual`, `implicit.md` §5): keep the root if its first-order distance to the
  ridge of codimension 1, from the field and its Jacobian at the root, is at most $\tau$. Needs the
  model only, and no further solve.
- **coverage** (`coverage`, [[coverage]]): keep the root unless flagged, $p < 0.05$, against 300
  held-out points, with the nearest-neighbour distance to 2000 more and the reconstruction score at
  $t = 0.01$. Needs data from the relation.
- **annealed**: the low levels with `continuation = (16, 4)` in every start, i.e. solved at
  $t \le 0.16$, then $0.04$, then $0.01$ (the reviews' "anneal the levels from high to low").

Low levels ($t \le 0.01$), the best setting of every relation above:

| relation | filter | recall | spurious (roots) | off-domain |
|---|---|---|---|---|
| circle | all roots | 0.857 | 0.234 (47) | 8/8 |
| circle | round trip | 0.738 | 0.088 (34) | 0/8 |
| circle | ridge | 0.857 | 0.077 (39) | 0/8 |
| circle | coverage | 0.857 | 0.053 (38) | 0/8 |
| lemniscate | all roots | 0.524 | 0.621 (58) | 8/8 |
| lemniscate | round trip | 0.119 | 0.375 (8) | 0/8 |
| lemniscate | ridge | 0.524 | 0.421 (38) | 0/8 |
| lemniscate | coverage | 0.310 | 0.000 (13) | 0/8 |
| two circles | all roots | 0.282 | 0.739 (134) | 8/8 |
| two circles | round trip | 0.161 | 0.130 (23) | 0/8 |
| two circles | ridge | 0.282 | 0.435 (62) | 0/8 |
| two circles | coverage | 0.282 | 0.079 (38) | 0/8 |

Annealed, against the low levels unannealed (all roots): circle recall 0.762 against 0.857, spurious
0.238 against 0.234; lemniscate **no root at all** (0 of 58); two circles recall 0.218 against 0.282,
spurious 0.518 against 0.739. The default and high levels, and both filters together, are in the log
of the runs (`~/.cache/lenticulum/f3/calibration.log` and, with the ridge rows,
`calibration-ridge.log`, not in the repository); they change none of
the conclusions below.

What it shows:

1. **Both checks remove every off-domain root** (0 of 8 left on each relation, at every level).
2. **Coverage is the better filter where data are at hand.** On the circle and the two circles it
   removes 9 of 11 and 96 of 99 spurious roots and no matched root (36 and 35 matched roots before and after). On the lemniscate it removes every
   spurious root, and recall falls from 0.52 to 0.31: roots near the crossing are both matched (within
   $\tau = 0.1$) and atypical of the data, and coverage cannot tell them apart.
3. **The round trip rejects true roots where the relation is parallel to the clamped coordinate.**
   On the circle, the true roots it rejects all lie at $\lvert x\rvert \le 0.2$, at the top and bottom
   (a diagnostic run, not kept: reversed solves that did not converge in 200 iterations, or converged
   0.14 to 0.24 away). There "$x$ given $y$" is ill-conditioned: the $x$-slice through a point near
   the top is flat along the circle, so the reversed query has no well-determined answer. The round trip
   thus tests that the relation is locally invertible in the reverse direction, as well as that the
   root is on it.
4. **The ridge residual tests membership alone, and loses no recall at the low levels.** It keeps
   every matched root on all three relations (36, 22 and 35 of them) and removes every off-domain
   root, against the round trip's recall losses (0.86 to 0.74, 0.52 to 0.12, 0.28 to 0.16). It removes
   fewer spurious roots than coverage: 8 of 11 on the circle, 20 of 36 on the lemniscate, 72 of 99 on
   the two circles, against coverage's 9, 36 and 96. On the lemniscate it does not cost recall at the
   crossing, where coverage does (0.52 to 0.31); combined with coverage it gives coverage's rows
   exactly at the low levels, so there coverage's rejections include all of the ridge's. At the
   default levels it costs the lemniscate one branch (recall 0.29 to 0.26), and the combination
   differs from coverage: on the lemniscate it keeps 6 roots instead of 11 (spurious 0.455 to 0.167,
   recall 0.14 to 0.12), on the two circles one spurious root fewer (0.079 to 0.054). At the high levels,
   where the branches have merged, it keeps no matched root. Why the remaining spurious roots pass
   (a ridge of the learned field that is not on the relation, or within $\tau$ of one) was not
   examined.
5. **Annealing as tried here hurts.** At $t \le 0.16$ the branches merge (the "high" rows above), so
   the starts reach fewer basins, and the low-level solves from there recover fewer branches. On the
   lemniscate no start ends converged and stable; why was not examined (each stage takes at most 200
   iterations, and the last one starts at the merged blob near the crossing). A schedule with more,
   smaller steps, or one that tracks every root as it splits, was not tried.
6. **No remedy finds what the queries miss.** The minority circle is still never recalled, and the
   lemniscate's recall is at best 0.52 before any filter.

Not checked: one seed per relation, one $\tau$, the round trip's and the ridge's tolerances equal to $\tau$, the ridge's codimension fixed at 1, coverage's
$\alpha = 0.05$, and one annealing schedule.

## What would improve it (not tried here)

Tried above: annealing, the round trip, the ridge residual, and rejection by coverage (reconstruction
and distance; these models have no energy). Not tried: an energy threshold (needs an energy network), or a relation trained as a codimension-$k$
residual. Each is a row in [[Feature Backlog]]; this example is the benchmark to measure them against.
