# Calibration of implicit relations

Run `julia -t auto --project=test examples/calibration.jl` from `lib/VariationalDiffusion.jl`
(about 70 minutes on a CPU, nearly all of it in the root queries). This answers the request of
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

## What would improve it (not tried here)

From the reviews: annealing the levels from high to low $t$ (track roots as $t$ decreases), rejecting
roots by the full score's norm or an energy threshold, a round-trip check (clamp the root's $y$ and
solve back for $x$), ridge residuals (only the score's component normal to the relation), or a
relation trained as a codimension-$k$ residual. Each is a row in [[Feature Backlog]]; this example
is the benchmark to measure them against.
